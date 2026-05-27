"""
Training loop for drone audio binary classification.

Features
────────
  • Two-phase training: freeze backbone → unfreeze after freeze_epochs
  • Mixed precision (torch.cuda.amp)
  • Gradient clipping
  • Early stopping on val_loss
  • Weights & Biases logging (optional)
  • Best-model checkpointing

Usage
─────
  from dronedetection.training.trainer import Trainer
  trainer = Trainer(model, train_loader, val_loader, cfg)
  trainer.fit()
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from dronedetection.evaluation.metrics import compute_metrics
from dronedetection.models.efficientnet import EfficientNetB0Classifier
from dronedetection.training.losses import build_bce_loss
from dronedetection.training.scheduler import build_scheduler
from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, value: float, model: nn.Module) -> bool:
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            log.info("Restored best model weights (val_loss=%.4f)", self.best)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        pos_weight: float | None = None,
        checkpoint_name: str = "best_model.pt",
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", self.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = build_bce_loss(pos_weight).to(self.device)

        # Build optimizer with per-group LRs
        t = cfg.training
        if hasattr(model, "get_param_groups"):
            param_groups = model.get_param_groups(t.lr_backbone, t.lr_head)
        else:
            param_groups = [{"params": model.parameters(), "lr": t.lr_head}]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=t.weight_decay)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp)

        self.early_stopping = EarlyStopping(patience=t.patience, mode="min")

        ckpt_dir = Path(cfg.paths.checkpoints)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = ckpt_dir / checkpoint_name

        self._history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        # Optional W&B
        self._wandb = None
        try:
            import wandb
            if cfg.logging.project:
                self._wandb = wandb
                wandb.init(
                    project=cfg.logging.project,
                    entity=cfg.logging.entity or None,
                    config=dict(cfg),
                )
        except ImportError:
            pass

    # ── training ──────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        t = self.cfg.training

        for specs, labels in tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False):
            specs = specs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)  # (B,1)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=t.use_amp):
                logits = self.model(specs)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for specs, labels in tqdm(self.val_loader, desc="Val", leave=False):
            specs = specs.to(self.device, non_blocking=True)
            labels_dev = labels.to(self.device).unsqueeze(1)

            with torch.cuda.amp.autocast(enabled=self.cfg.training.use_amp):
                logits = self.model(specs)
                loss = self.criterion(logits, labels_dev)

            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).cpu().squeeze(1))
            all_labels.append(labels)

        probs = torch.cat(all_probs)
        labels_all = torch.cat(all_labels)
        metrics = compute_metrics(probs, labels_all, threshold=self.cfg.evaluation.threshold)

        return total_loss / len(self.val_loader), metrics

    # ── plot ──────────────────────────────────────────────────────────────────

    def _save_history_plot(self) -> None:
        if not _MPL_AVAILABLE:
            log.warning("matplotlib not available — skipping history plot")
            return
        plot_dir = Path("outputs/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        stem = self.ckpt_path.stem  # e.g. "best_model_no_aug"
        plot_path = plot_dir / f"training_history_{stem}.png"

        epochs = range(1, len(self._history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self._history["train_loss"], label="train_loss")
        ax.plot(epochs, self._history["val_loss"], label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training history — {stem}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info("Training history plot saved → %s", plot_path)

    # ── main fit loop ─────────────────────────────────────────────────────────

    def fit(self) -> None:
        t = self.cfg.training
        freeze_epochs = t.freeze_epochs

        for epoch in range(1, t.epochs + 1):
            # Phase transition: unfreeze backbone after freeze_epochs
            if epoch == freeze_epochs + 1 and isinstance(self.model, EfficientNetB0Classifier):
                self.model.unfreeze_backbone()
                log.info("Epoch %d: unfreezing backbone", epoch)

            train_loss = self._train_epoch(epoch)
            val_loss, metrics = self._val_epoch()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            log.info(
                "E%03d | train_loss=%.4f  val_loss=%.4f  "
                "recall=%.3f  prec=%.3f  f1=%.3f  auc=%.3f  lr=%.2e",
                epoch, train_loss, val_loss,
                metrics["recall"], metrics["precision"], metrics["f1"], metrics["roc_auc"], lr
            )

            if self._wandb:
                self._wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    **{f"val_{k}": v for k, v in metrics.items()},
                })

            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)

            stopped = self.early_stopping.step(val_loss, self.model)
            if stopped:
                log.info("Early stopping triggered at epoch %d", epoch)
                break

        self.early_stopping.restore_best(self.model)
        torch.save(self.model.state_dict(), self.ckpt_path)
        log.info("Best model saved → %s", self.ckpt_path)
        self._save_history_plot()

        if self._wandb:
            self._wandb.save(str(self.ckpt_path))
            self._wandb.finish()
