from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualChannelMixMLP(nn.Module):
    """
    LayerNorm -> Linear(C -> expansion*C) -> GELU -> Dropout ->
    Linear(expansion*C -> C) -> Residual.

    Operates on the last dimension, so the caller must present input as
    (B, L, C).
    """

    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return residual + h


class ConvBranch(nn.Module):
    """Single strided Conv1d branch: (B, C_in, L) -> (B, hidden_dim, L_out)."""

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int, stride: int):
        super().__init__()
        padding = max(kernel_size // 2, 0)
        self.conv = nn.Conv1d(
            in_channels, hidden_dim, kernel_size=kernel_size,
            stride=stride, padding=padding,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        h = self.conv(x)                # (B, hidden, L_out)
        h = h.transpose(1, 2)           # (B, L_out, hidden)
        h = self.norm(h)
        h = self.act(h)
        return h


class MultiScaleConvEmbedding(nn.Module):
    """
    Three parallel branches with distinct kernel/stride, each producing a
    sequence of `hidden_dim`-dimensional tokens at a different temporal
    resolution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernels: Tuple[int, int, int],
        strides: Tuple[int, int, int],
    ):
        super().__init__()
        assert len(kernels) == len(strides) == 3
        self.branches = nn.ModuleList([
            ConvBranch(in_channels, hidden_dim, k, s)
            for k, s in zip(kernels, strides)
        ])

    def forward(self, x: torch.Tensor):
        # x: (B, C_in, L) -> list of 3 tensors (B, L_i, hidden_dim)
        return [branch(x) for branch in self.branches]


class DropPath(nn.Module):
    """Per-sample stochastic depth (drop the residual branch outright)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SelectiveSSM(nn.Module):
    """
    Pure-PyTorch re-implementation of the selective SSM (S6) mechanism
    used by Mamba. This favors clarity and portability (no custom CUDA
    kernel required) over raw throughput; the sequential scan over time
    is done with a Python loop, which is perfectly fine at the token
    counts produced by the conv embedding (tens to a few hundred steps).

    Input / output: (B, L, d_model) -> (B, L, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: "int | str" = "auto",
        conv_kernel: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=self.d_inner,
        )
        self.conv_kernel = conv_kernel

        # Projects the (post-conv) hidden state to per-timestep dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # A is parameterized as -softplus/-exp(A_log) to stay negative
        # (stable / decaying state dynamics), one row per inner channel.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)  # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        x_and_res = self.in_proj(x)                      # (B, L, 2*d_inner)
        x_in, res = x_and_res.chunk(2, dim=-1)            # each (B, L, d_inner)

        # Depthwise causal-ish conv over time
        x_in_t = x_in.transpose(1, 2)                     # (B, d_inner, L)
        x_in_t = self.conv1d(x_in_t)[..., :L]
        x_in = x_in_t.transpose(1, 2)                      # (B, L, d_inner)
        x_in = F.silu(x_in)

        x_dbl = self.x_proj(x_in)                          # (B, L, dt_rank + 2*d_state)
        dt, Bmat, Cmat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))                  # (B, L, d_inner)

        A = -torch.exp(self.A_log)                          # (d_inner, d_state)

        y = self._selective_scan(x_in, dt, A, Bmat, Cmat, self.D)
        y = y * F.silu(res)
        return self.out_proj(y)

    @staticmethod
    def _selective_scan(
        u: torch.Tensor,      # (B, L, d_inner)
        delta: torch.Tensor,  # (B, L, d_inner)
        A: torch.Tensor,      # (d_inner, d_state)
        Bmat: torch.Tensor,   # (B, L, d_state)
        Cmat: torch.Tensor,   # (B, L, d_state)
        D: torch.Tensor,      # (d_inner,)
    ) -> torch.Tensor:
        Bsz, L, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize: deltaA_t = exp(delta_t * A)   (B, L, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        # deltaB_u_t = delta_t * B_t * u_t          (B, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * u.unsqueeze(-1)

        state = torch.zeros(Bsz, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            state = deltaA[:, t] * state + deltaB_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", state, Cmat[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + u * D
        return y


class BiMambaBlock(nn.Module):
    """
    LayerNorm -> Linear projection -> Depthwise conv -> Forward Mamba ->
    Backward Mamba -> Gate -> Residual -> FeedForward -> Residual.
    """

    def __init__(
        self,
        d_model: int = 128,
        expand: int = 2,
        d_state: int = 16,
        dt_rank: "int | str" = "auto",
        ffn_expansion: int = 4,
        ffn_dropout: float = 0.1,
        drop_path: float = 0.1,
        conv_kernel: int = 3,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model)

        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_model,
        )
        self.conv_kernel = conv_kernel

        self.mamba_fwd = SelectiveSSM(d_model, d_state=d_state, expand=expand, dt_rank=dt_rank)
        self.mamba_bwd = SelectiveSSM(d_model, d_state=d_state, expand=expand, dt_rank=dt_rank)

        self.gate = nn.Linear(2 * d_model, d_model)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)
        ffn_hidden = d_model * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        L = x.shape[1]
        residual = x

        h = self.norm1(x)
        h = self.in_proj(h)

        h_t = h.transpose(1, 2)
        h_t = self.dwconv(h_t)[..., :L]
        h = h_t.transpose(1, 2)

        fwd = self.mamba_fwd(h)
        bwd_in = torch.flip(h, dims=[1])
        bwd = self.mamba_bwd(bwd_in)
        bwd = torch.flip(bwd, dims=[1])

        gate = torch.sigmoid(self.gate(torch.cat([fwd, bwd], dim=-1)))
        combined = gate * fwd + (1 - gate) * bwd

        x = residual + self.drop_path1(combined)

        residual2 = x
        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        x = residual2 + self.drop_path2(h2)
        return x


class MultiScaleFusion(nn.Module):
    """
    Aligns the three branches to a common sequence length via adaptive
    average pooling, concatenates along the feature dimension, then
    projects: Linear -> GELU -> LayerNorm.
    """

    def __init__(self, hidden_dim: int, fusion_dim: int, num_branches: int = 3):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * num_branches, fusion_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, branch_outputs) -> torch.Tensor:
        # branch_outputs: list of (B, L_i, hidden_dim), L_i differ per branch
        common_len = min(t.shape[1] for t in branch_outputs)
        aligned = []
        for t in branch_outputs:
            if t.shape[1] == common_len:
                aligned.append(t)
            else:
                t_t = t.transpose(1, 2)  # (B, hidden, L_i)
                t_t = F.adaptive_avg_pool1d(t_t, common_len)
                aligned.append(t_t.transpose(1, 2))
        fused = torch.cat(aligned, dim=-1)  # (B, common_len, hidden*num_branches)
        fused = self.proj(fused)
        fused = self.act(fused)
        fused = self.norm(fused)
        return fused  # (B, common_len, fusion_dim)


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over the time dimension:
      score_t = w^T tanh(W h_t)
      alpha   = softmax(score) over t
      pooled  = sum_t alpha_t * h_t
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.score_proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1),
        )
        self.out_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_dim)
        scores = self.score_proj(x)                 # (B, L, 1)
        alpha = torch.softmax(scores, dim=1)          # (B, L, 1)
        pooled = torch.sum(alpha * x, dim=1)          # (B, in_dim)
        return self.out_proj(pooled)                  # (B, out_dim)
