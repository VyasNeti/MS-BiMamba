from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import cfg
from dataset import get_dataloaders
from model import build_model
from losses import WeightedMSELoss
from trainer import Trainer
from utils import set_seed, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MedMamba-Reg for cuffless BP estimation.")
    p.add_argument("--data-dir", type=str, default=None, help="Override cfg.data.data_dir")
    p.add_argument("--epochs", type=int, default=None, help="Override cfg.train.num_epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override cfg.train.batch_size")
    p.add_argument("--lr", type=float, default=None, help="Override cfg.optim.lr")
    p.add_argument("--seed", type=int, default=None, help="Override cfg.train.seed")
    p.add_argument("--no-cache", action="store_true", help="Disable window caching / force rebuild")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument(
        "--multi-gpu", action="store_true",
        help="Wrap model in nn.DataParallel if more than one GPU is visible.",
    )
    return p.parse_args()


def apply_overrides(args: argparse.Namespace) -> None:
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
        cfg.scheduler.t_max = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.optim.lr = args.lr
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.no_cache:
        cfg.data.use_cache = False
    if args.no_amp:
        cfg.train.use_amp = False
    if args.multi_gpu:
        cfg.train.use_dataparallel = True


def main() -> None:
    args = parse_args()
    apply_overrides(args)

    logger = get_logger("train", log_file=os.path.join(cfg.train.log_dir, "train_main.log"))
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info(f"Using device: {device}")

    logger.info("Building dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(seed=cfg.train.seed)

    logger.info("Building model...")
    model = build_model(cfg).to(device)
    # try:
    #     model = torch.compile(model)
    #     logger.info("torch.compile enabled.")
    # except Exception as e:
    #     logger.warning(f"torch.compile unavailable: {e}")

    if cfg.train.use_dataparallel and torch.cuda.device_count() > 1:
        logger.info(f"Using nn.DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    criterion = WeightedMSELoss(cfg.loss)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.scheduler.t_max, eta_min=cfg.scheduler.eta_min
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
