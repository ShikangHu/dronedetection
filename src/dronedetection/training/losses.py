"""Loss functions for binary drone audio classification."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_bce_loss(pos_weight: float | None = None) -> nn.BCEWithLogitsLoss:
    """
    Binary cross-entropy with logits loss.

    Args:
        pos_weight: Weight for positive (drone) class.
                    Set to n_negative / n_positive for class imbalance.
                    None = equal weighting.

    Returns:
        nn.BCEWithLogitsLoss instance.
    """
    pw = torch.tensor([pos_weight]) if pos_weight is not None else None
    return nn.BCEWithLogitsLoss(pos_weight=pw)


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification.

    Reduces the loss contribution from easy (confident) examples,
    focusing training on hard negatives. Useful if hard negatives
    (helicopter, fan, lawnmower) dominate the non-drone class.

    Args:
        alpha: Weighting factor for positive class (0.25 is a good default).
        gamma: Focusing parameter. 0 = standard BCE, 2 is typical.
        pos_weight: Optional pos_weight passed to BCE before focal scaling.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
