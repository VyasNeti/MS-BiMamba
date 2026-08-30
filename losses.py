from __future__ import annotations

import torch
import torch.nn as nn

from config import LossConfig


class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error:

        loss = sbp_weight * MSE(pred_sbp, true_sbp)
             + dbp_weight * MSE(pred_dbp, true_dbp)

    Expects `pred` and `target` of shape (B, 2), columns ordered
    [SBP, DBP].
    """

    def __init__(self, loss_cfg: LossConfig):
        super().__init__()
        self.sbp_weight = loss_cfg.sbp_weight
        self.dbp_weight = loss_cfg.dbp_weight
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sbp_loss = self.mse(pred[:, 0], target[:, 0])
        dbp_loss = self.mse(pred[:, 1], target[:, 1])
        return self.sbp_weight * sbp_loss + self.dbp_weight * dbp_loss

    def component_losses(self, pred: torch.Tensor, target: torch.Tensor):
        """Returns (sbp_loss, dbp_loss, total_loss) as detached floats -- useful for logging."""
        with torch.no_grad():
            sbp_loss = self.mse(pred[:, 0], target[:, 0])
            dbp_loss = self.mse(pred[:, 1], target[:, 1])
            total = self.sbp_weight * sbp_loss + self.dbp_weight * dbp_loss
        return sbp_loss.item(), dbp_loss.item(), total.item()
