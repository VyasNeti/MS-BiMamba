from __future__ import annotations

import torch
import torch.nn as nn

from config import Config
from modules import (
    ResidualChannelMixMLP,
    MultiScaleConvEmbedding,
    BiMambaBlock,
    MultiScaleFusion,
    AttentionPooling,
)


class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MedMambaReg(nn.Module):
    """MedMamba-inspired network for SBP/DBP regression from PPG-derived signals."""

    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model

        self.channel_mix = ResidualChannelMixMLP(
            channels=m.in_channels,
            expansion=m.channel_mix_expansion,
            dropout=m.channel_mix_dropout,
        )

        self.conv_embed = MultiScaleConvEmbedding(
            in_channels=m.in_channels,
            hidden_dim=m.hidden_dim,
            kernels=m.conv_branch_kernels,
            strides=m.conv_branch_strides,
        )

        # Independent stack of BiMamba blocks per branch (3 branches total)
        self.branch_encoders = nn.ModuleList([
            nn.Sequential(*[
                BiMambaBlock(
                    d_model=m.hidden_dim,
                    expand=m.mamba_expand,
                    d_state=m.mamba_d_state,
                    dt_rank=m.mamba_dt_rank,
                    ffn_expansion=m.ffn_expansion,
                    ffn_dropout=m.ffn_dropout,
                    drop_path=m.drop_path,
                    conv_kernel=m.mamba_conv_kernel,
                )
                for _ in range(m.num_mamba_blocks_per_branch)
            ])
            for _ in range(3)
        ])

        self.fusion = MultiScaleFusion(
            hidden_dim=m.hidden_dim,
            fusion_dim=m.fusion_dim,
            num_branches=3,
        )

        self.attn_pool = AttentionPooling(
            in_dim=m.fusion_dim,
            out_dim=m.pooled_dim,
        )

        self.head = RegressionHead(
            in_dim=m.pooled_dim,
            hidden_dim=m.head_hidden,
            out_dim=m.num_outputs,
            dropout=m.head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, seq_len) - channels are [PPG, VPG, APG]
        returns: (B, 2) - [SBP, DBP]
        """
        # Channel mixing operates on the channel dimension -> present as (B, L, C)
        h = x.transpose(1, 2)              # (B, L, C)
        h = self.channel_mix(h)            # (B, L, C)
        h = h.transpose(1, 2)              # (B, C, L)

        branch_tokens = self.conv_embed(h)  # list of (B, L_i, hidden_dim)

        encoded = [
            encoder(tokens)
            for encoder, tokens in zip(self.branch_encoders, branch_tokens)
        ]

        fused = self.fusion(encoded)        # (B, common_len, fusion_dim)
        pooled = self.attn_pool(fused)       # (B, pooled_dim)

        out = self.head(pooled)              # (B, 2)
        return out


def build_model(cfg: Config) -> MedMambaReg:
    """Factory function used by train.py / test.py / predict.py."""
    return MedMambaReg(cfg)


if __name__ == "__main__":
    # Quick shape sanity check
    from config import cfg as _cfg

    model = build_model(_cfg)
    dummy = torch.randn(4, _cfg.model.in_channels, _cfg.model.seq_len)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: (4, 2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
