from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

EXPERIMENT_NAME = os.getenv("EXP_NAME", "default")


@dataclass
class Assumptions:
    # Channel order inside each cell of data["p"], shape (3, 61000):
    #   0 -> PPG, 1 -> ABP, 2 -> ECG   (as stated in the task description)
    ppg_channel_idx: int = 0
    abp_channel_idx: int = 1
    ecg_channel_idx: int = 2 

    sampling_rate_hz: int = 125

    sbp_valid_range: Tuple[float, float] = (70.0, 220.0)
    dbp_valid_range: Tuple[float, float] = (40.0, 140.0)

    peak_min_distance_sec: float = 0.4


@dataclass
class DataConfig:
    data_dir: str = "/scratch/bhanu/data/cuffless_bp"
    file_glob: str = "*.mat"

    window_length: int = 1024
    stride: int = 512

    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

    num_workers: int = 4
    pin_memory: bool = True

    cache_dir: str = "/scratch/bhanu/alpha/vyas/Mamba_Reg/outputs/cache"
    use_cache: bool = True


@dataclass
class ModelConfig:
    # in_channels: int = 3          # PPG, VPG, APG
    input_mode: str = os.getenv("INPUT_MODE", "ppg_vpg_apg")

    if input_mode == "ppg":
        in_channels: int = 1
    elif input_mode == "ppg_vpg":
        in_channels: int = 2
    else:
        in_channels: int = 3

    seq_len: int = 1024

    # Channel mixing (residual MLP over the 3 input channels)
    channel_mix_expansion: int = 4
    channel_mix_dropout: float = 0.1

    # Multi-scale conv embedding
    conv_branch_kernels: Tuple[int, int, int] = (5, 11, 25)
    conv_branch_strides: Tuple[int, int, int] = (5, 10, 25)
    hidden_dim: int = 128

    # Mamba encoder
    num_mamba_blocks_per_branch: int = int(os.getenv("NUM_MAMBA_BLOCKS", "2"))
    mamba_expand: int = 2
    mamba_d_state: int = 16
    mamba_dt_rank: str = "auto"   # 'auto' -> ceil(hidden_dim / 16)
    mamba_conv_kernel: int = 3
    ffn_expansion: int = 4
    ffn_dropout: float = 0.1
    drop_path: float = 0.1

    # Multi-scale fusion
    fusion_dim: int = 256

    # Attention pooling
    pooled_dim: int = 256

    # Regression head
    head_hidden: int = 128
    head_dropout: float = 0.2

    num_outputs: int = 2  # SBP, DBP


@dataclass
class LossConfig:
    sbp_weight: float = 0.6
    dbp_weight: float = 0.4


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    t_max: int = 100          # set to num_epochs at build time
    eta_min: float = 1e-6


@dataclass
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 100
    early_stopping_patience: int = 15
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    seed: int = 42

    checkpoint_dir: str = (
        f"/scratch/bhanu/alpha/vyas/Mamba_Reg/checkpoints/{EXPERIMENT_NAME}"
    )

    log_dir: str = (
        f"/scratch/bhanu/alpha/vyas/Mamba_Reg/logs/{EXPERIMENT_NAME}"
    )

    output_dir: str = (
        f"/scratch/bhanu/alpha/vyas/Mamba_Reg/outputs/{EXPERIMENT_NAME}"
    )

    best_ckpt_name: str = "best_model.pth"
    last_ckpt_name: str = "last_model.pth"
    metrics_csv_name: str = "training_metrics.csv"

    resume: bool = True  # auto-resume from last_ckpt if present

    # Multi-GPU
    use_ddp: bool = False       # set True to launch with torchrun
    use_dataparallel: bool = False  # simple single-process multi-GPU fallback


@dataclass
class Config:
    assumptions: Assumptions = field(default_factory=Assumptions)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # keep scheduler t_max in sync with num_epochs unless overridden
        self.scheduler.t_max = self.train.num_epochs
        for d in (self.train.checkpoint_dir, self.train.log_dir,
                  self.train.output_dir, self.data.cache_dir):
            os.makedirs(d, exist_ok=True)


cfg = Config()
