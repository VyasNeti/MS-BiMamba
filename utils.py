from __future__ import annotations

import os
import random
import logging
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_derivative(signal: np.ndarray) -> np.ndarray:
    """
    First-order numerical derivative along the last axis, using
    `np.gradient` (central differences, edge-safe). Works on 1-D windows
    or batched (N, L) arrays.
    """
    return np.gradient(signal, axis=-1)


def compute_vpg_apg(ppg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a PPG window/signal, compute:
      VPG = first derivative of PPG
      APG = second derivative of PPG (derivative of VPG)
    """
    vpg = compute_derivative(ppg)
    apg = compute_derivative(vpg)
    return vpg, apg


def zscore_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize along the last axis. Works per-channel when `x`
    has shape (C, L): each row is normalized independently.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)



def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_regression_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """
    pred, target: (N, 2) tensors, columns = [SBP, DBP].
    Returns a flat dict of scalar metrics.
    """
    with torch.no_grad():
        sbp_pred, dbp_pred = pred[:, 0], pred[:, 1]
        sbp_true, dbp_true = target[:, 0], target[:, 1]
        return {
            "sbp_mae": mae(sbp_pred, sbp_true).item(),
            "dbp_mae": mae(dbp_pred, dbp_true).item(),
            "sbp_rmse": rmse(sbp_pred, sbp_true).item(),
            "dbp_rmse": rmse(dbp_pred, dbp_true).item(),
        }


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_mae: float,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_mae": best_val_mae,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict:
    checkpoint = torch.load(path, map_location=map_location)
    _unwrap(model).load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module if wrapped in DataParallel/DDP."""
    return model.module if hasattr(model, "module") else model



def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def config_to_dict(cfg_obj) -> Dict:
    """Recursively convert a nested dataclass Config into a plain dict."""
    return asdict(cfg_obj)
