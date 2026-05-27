"""
Train the drone audio classifier.

Usage
─────
  python scripts/train.py
  python scripts/train.py --config configs/lightweight.yaml
  python scripts/train.py training.lr_backbone=5e-5 model.dropout=0.5
  python scripts/train.py model.name=lightweight_cnn

Hydra-style overrides are supported for any key in the config YAML.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dronedetection.data.augmentation import SpectrogramAugmenter, WaveformAugmenter
from dronedetection.data.dataset import DroneAudioDataset, make_weighted_sampler
from dronedetection.data.features import (
    MelSpectrogramExtractor,
    compute_dataset_stats,
    save_stats,
)
from dronedetection.models.factory import build_model
from dronedetection.training.trainer import Trainer
from dronedetection.utils.logging import get_logger
from dronedetection.utils.seed import seed_everything

log = get_logger(__name__)


def parse_overrides(extra: list[str], cfg):
    """Apply key=value CLI overrides to an OmegaConf config."""
    overrides = OmegaConf.from_dotlist(extra)
    return OmegaConf.merge(cfg, overrides)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint-name", default="best_model.pt",
                        help="Filename for the saved best checkpoint (inside checkpoints/)")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cfg = parse_overrides(overrides, cfg)

    seed_everything(cfg.data.random_seed)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    splits_dir = Path(cfg.paths.splits)
    stats_dir = Path(cfg.paths.checkpoints)

    # ── Build datasets ────────────────────────────────────────────────────────
    waveform_aug = WaveformAugmenter(cfg)
    spec_aug = SpectrogramAugmenter(cfg)

    train_ds = DroneAudioDataset(
        splits_dir / "train.csv", cfg, split="train",
        waveform_augmenter=waveform_aug, spec_augmenter=spec_aug,
    )
    val_ds = DroneAudioDataset(splits_dir / "val.csv", cfg, split="val")

    # ── Compute and save normalisation stats from training split ──────────────
    stats_path = stats_dir / "mean.npy"
    if not stats_path.exists():
        log.info("Computing dataset statistics from training split …")
        extractor = MelSpectrogramExtractor(cfg.data)
        sample_specs = []
        for i in range(min(1000, len(train_ds))):
            spec, _ = train_ds[i]
            sample_specs.append(spec)
        mean, std = compute_dataset_stats(sample_specs)
        save_stats(mean, std, stats_dir)
    else:
        from dronedetection.data.features import load_stats
        mean, std = load_stats(stats_dir)
        log.info("Loaded existing stats: mean=%.4f, std=%.4f", mean, std)

    # Apply stats to all splits
    train_ds.stats = (mean, std)
    val_ds.stats = (mean, std)

    # ── Compute class balance for pos_weight ─────────────────────────────────
    n_pos = sum(1 for seg in train_ds.segments if seg[1] == 1)
    n_neg = len(train_ds.segments) - n_pos
    pos_weight = n_neg / (n_pos + 1e-9)
    log.info("Train: %d pos, %d neg → pos_weight=%.2f", n_pos, n_neg, pos_weight)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    sampler = make_weighted_sampler(train_ds)
    t = cfg.training
    train_loader = DataLoader(
        train_ds, batch_size=t.batch_size, sampler=sampler,
        num_workers=t.num_workers, pin_memory=t.pin_memory, drop_last=True,
        persistent_workers=t.num_workers > 0, prefetch_factor=2 if t.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=t.batch_size, shuffle=False,
        num_workers=t.num_workers, pin_memory=t.pin_memory,
        persistent_workers=t.num_workers > 0, prefetch_factor=2 if t.num_workers > 0 else None,
    )

    # ── Model & Trainer ───────────────────────────────────────────────────────
    model = build_model(cfg)
    log.info(
        "Model: %s  |  params: %s",
        cfg.model.name,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    trainer = Trainer(model, train_loader, val_loader, cfg, pos_weight=pos_weight,
                      checkpoint_name=args.checkpoint_name)
    trainer.fit()


if __name__ == "__main__":
    main()
