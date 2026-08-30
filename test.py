from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from config import cfg
from dataset import get_dataloaders
from model import build_model
from utils import compute_regression_metrics, load_checkpoint, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MedMamba-Reg on the test split.")
    p.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(cfg.train.checkpoint_dir, cfg.train.best_ckpt_name),
        help="Path to a .pth checkpoint (default: best_model.pth).",
    )
    p.add_argument("--data-dir", type=str, default=None, help="Override cfg.data.data_dir")
    p.add_argument("--batch-size", type=int, default=None, help="Override cfg.train.batch_size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    logger = get_logger("test", log_file=os.path.join(cfg.train.log_dir, "test.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Building test dataloader...")
    _, _, test_loader = get_dataloaders(seed=cfg.train.seed)

    logger.info(f"Loading model checkpoint: {args.checkpoint}")
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="test"):
            x = x.to(device)
            pred = model(x).cpu()
            all_preds.append(pred)
            all_targets.append(y)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = compute_regression_metrics(preds, targets)
    logger.info(
        f"Test results | SBP MAE={metrics['sbp_mae']:.3f} DBP MAE={metrics['dbp_mae']:.3f} | "
        f"SBP RMSE={metrics['sbp_rmse']:.3f} DBP RMSE={metrics['dbp_rmse']:.3f}"
    )

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.train.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    preds_path = os.path.join(cfg.train.output_dir, "test_predictions.npz")
    np.savez_compressed(
        preds_path,
        pred_sbp=preds[:, 0].numpy(),
        pred_dbp=preds[:, 1].numpy(),
        true_sbp=targets[:, 0].numpy(),
        true_dbp=targets[:, 1].numpy(),
    )
    logger.info(f"Saved raw predictions to {preds_path}")


if __name__ == "__main__":
    main()
