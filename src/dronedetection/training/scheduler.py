"""Learning-rate scheduler construction."""
from __future__ import annotations

import math

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR


def build_scheduler(
    optimizer: Optimizer,
    cfg: DictConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a two-stage scheduler:
      Stage 1: Linear warmup from warmup_start_lr → target LR over warmup_epochs
      Stage 2: CosineAnnealingWarmRestarts(T_0, T_mult) for the remaining epochs

    Args:
        optimizer: AdamW optimizer with pre-set initial LRs as target LRs.
        cfg: Full hydra config (reads cfg.training).

    Returns:
        SequentialLR that switches from warmup to cosine after warmup_epochs.
    """
    t = cfg.training
    warmup_epochs = t.warmup_epochs
    start_factor = t.warmup_start_lr / max(t.lr_head, 1e-10)

    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: start_factor + (1.0 - start_factor) * epoch / max(warmup_epochs, 1)
    )
    cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t.t0,
        T_mult=t.t_mult,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler
