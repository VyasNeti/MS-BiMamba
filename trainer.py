from __future__ import annotations

import csv
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from utils import (
    compute_regression_metrics,
    save_checkpoint,
    load_checkpoint,
    get_logger,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg: Config,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device

        self.logger = get_logger(
            "trainer", log_file=os.path.join(cfg.train.log_dir, "train.log")
        )
        self.writer = SummaryWriter(log_dir=cfg.train.log_dir)

        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.use_amp)

        self.best_val_mae = float("inf")
        self.start_epoch = 0
        self.epochs_since_improvement = 0

        self.metrics_csv_path = os.path.join(cfg.train.output_dir, cfg.train.metrics_csv_name)
        self._init_csv()

        if cfg.train.resume:
            self._maybe_resume()

    def _init_csv(self) -> None:
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "val_loss",
                    "sbp_mae", "dbp_mae", "sbp_rmse", "dbp_rmse",
                    "lr", "epoch_time_sec",
                ])

    def _append_csv(self, row: list) -> None:
        with open(self.metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _maybe_resume(self) -> None:
        last_ckpt = os.path.join(self.cfg.train.checkpoint_dir, self.cfg.train.last_ckpt_name)
        if os.path.exists(last_ckpt):
            self.logger.info(f"Resuming from checkpoint: {last_ckpt}")
            state = load_checkpoint(
                last_ckpt, self.model, self.optimizer, self.scheduler, self.scaler,
                map_location=str(self.device),
            )

            self.start_epoch = state["epoch"] + 1
            self.best_val_mae = state.get("best_val_mae", float("inf"))
            self.logger.info(
                f"Resumed at epoch {self.start_epoch}, best_val_mae={self.best_val_mae:.4f}"
            )

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict:
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        n_batches = 0
        all_preds, all_targets = [], []

        desc = "train" if train else "val"
        pbar = tqdm(loader, desc=desc, leave=False)

        for batch_idx, (x, y) in enumerate(pbar):

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)


            with torch.set_grad_enabled(train):
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.amp.autocast(device_type, enabled=self.cfg.train.use_amp):

                    pred = self.model(x)


                    loss = self.criterion(pred, y)

                if train:

                    self.optimizer.zero_grad(set_to_none=True)

                    if self.cfg.train.use_amp:
                        self.scaler.scale(loss).backward()

                        self.scaler.unscale_(self.optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.train.grad_clip_norm
                        )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.train.grad_clip_norm
                        )

                        self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(pred.detach().float().cpu())
            all_targets.append(y.detach().float().cpu())
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(n_batches, 1)
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_regression_metrics(preds, targets)
        metrics["loss"] = avg_loss
        return metrics

    def fit(self) -> None:
        cfg = self.cfg
        for epoch in range(self.start_epoch, cfg.train.num_epochs):
            t0 = time.time()

            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0

            self.logger.info(
                f"Epoch {epoch + 1}/{cfg.train.num_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
                f"SBP MAE={val_metrics['sbp_mae']:.3f} DBP MAE={val_metrics['dbp_mae']:.3f} | "
                f"SBP RMSE={val_metrics['sbp_rmse']:.3f} DBP RMSE={val_metrics['dbp_rmse']:.3f} | "
                f"lr={lr:.2e} | time={elapsed:.1f}s"
            )

            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("LR", lr, epoch)
            self.writer.add_scalar("MAE/SBP", val_metrics["sbp_mae"], epoch)
            self.writer.add_scalar("MAE/DBP", val_metrics["dbp_mae"], epoch)
            self.writer.add_scalar("RMSE/SBP", val_metrics["sbp_rmse"], epoch)
            self.writer.add_scalar("RMSE/DBP", val_metrics["dbp_rmse"], epoch)

            self._append_csv([
                epoch + 1, train_metrics["loss"], val_metrics["loss"],
                val_metrics["sbp_mae"], val_metrics["dbp_mae"],
                val_metrics["sbp_rmse"], val_metrics["dbp_rmse"],
                lr, round(elapsed, 2),
            ])

            val_mae = (val_metrics["sbp_mae"] + val_metrics["dbp_mae"]) / 2.0
            improved = val_mae < self.best_val_mae

            if improved:
                self.best_val_mae = val_mae
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1

            last_ckpt = os.path.join(cfg.train.checkpoint_dir, cfg.train.last_ckpt_name)
            save_checkpoint(last_ckpt, self.model, self.optimizer, self.scheduler,
                             epoch, self.best_val_mae, self.scaler)

            if improved:
                best_ckpt = os.path.join(cfg.train.checkpoint_dir, cfg.train.best_ckpt_name)
                save_checkpoint(best_ckpt, self.model, self.optimizer, self.scheduler,
                                 epoch, self.best_val_mae, self.scaler)
                self.logger.info(f"New best model saved (mean val MAE={val_mae:.4f}).")
            elif self.epochs_since_improvement >= cfg.train.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {cfg.train.early_stopping_patience} epochs)."
                )
                break

        self.writer.close()
        self.logger.info("Training complete.")
