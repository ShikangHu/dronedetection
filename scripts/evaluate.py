"""
Evaluate a trained model on the test split.

Performs:
  • Full metrics at default threshold
  • Threshold tuning for >= 97% recall
  • Confusion matrix and ROC/PR curve plots
  • Per-source breakdown
  • Top-K error analysis

Usage
─────
  python scripts/evaluate.py
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
  python scripts/evaluate.py --split val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dronedetection.data.dataset import DroneAudioDataset
from dronedetection.data.features import load_stats
from dronedetection.evaluation.analysis import (
    per_source_metrics,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    top_k_errors,
)
from dronedetection.evaluation.metrics import (
    compute_metrics,
    find_threshold_for_recall,
    print_metrics,
)
from dronedetection.models.factory import build_model
from dronedetection.utils.logging import get_logger
from dronedetection.utils.seed import seed_everything

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to an arbitrary CSV manifest (overrides --split)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    seed_everything(cfg.data.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(cfg.paths.splits)
    stats_dir = Path(cfg.paths.checkpoints)

    mean, std = load_stats(stats_dir)
    csv_path = Path(args.csv) if args.csv else splits_dir / f"{args.split}.csv"
    ds = DroneAudioDataset(csv_path, cfg, split=args.split, stats=(mean, std))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg.training.num_workers)

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_paths: list[str] = []
    all_sources: list[str] = []

    with torch.no_grad():
        for i, (specs, labels) in enumerate(tqdm(loader, desc="Evaluating")):
            specs = specs.to(device)
            with torch.cuda.amp.autocast(enabled=cfg.training.use_amp):
                logits = model(specs)
            probs = torch.sigmoid(logits).cpu().squeeze(1)
            all_probs.append(probs)
            all_labels.append(labels)
            # Recover paths and sources from dataset segments
            batch_start = i * loader.batch_size
            for j in range(len(labels)):
                idx = batch_start + j
                if idx < len(ds.segments):
                    path, lbl, _, _ = ds.segments[idx]
                    all_paths.append(path)
                    rec = next((r for r in ds.records if r["path"] == path), {})
                    all_sources.append(rec.get("source", "unknown"))

    probs_all = torch.cat(all_probs)
    labels_all = torch.cat(all_labels)

    # ── Default threshold metrics ────────────────────────────────────────────
    metrics = compute_metrics(probs_all, labels_all, threshold=cfg.evaluation.threshold)
    print_metrics(metrics, header=f"Metrics on '{args.split}' split  (threshold={cfg.evaluation.threshold})")

    # ── Threshold tuning ─────────────────────────────────────────────────────
    best_thresh, tuned_metrics = find_threshold_for_recall(
        probs_all, labels_all, min_recall=cfg.evaluation.min_recall_target
    )
    print_metrics(tuned_metrics, header=f"After threshold tuning (threshold={best_thresh:.3f})")
    log.info("Recommended threshold for deployment: %.3f", best_thresh)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(probs_all, labels_all, threshold=best_thresh,
                          save_path=plot_dir / "confusion_matrix.png")
    plot_roc_pr_curves(probs_all, labels_all, save_path=plot_dir / "roc_pr_curves.png")

    # ── Per-source breakdown ─────────────────────────────────────────────────
    src_metrics = per_source_metrics(probs_all, labels_all, all_sources, threshold=best_thresh)
    print("\n── Per-source Recall ──")
    for src, m in sorted(src_metrics.items()):
        print(f"  {src:20s}  recall={m['recall']:.3f}  precision={m['precision']:.3f}  n={m['tp']+m['fn']+m['fp']+m['tn']}")

    # ── Top-K errors ─────────────────────────────────────────────────────────
    errors = top_k_errors(probs_all, labels_all, all_paths, k=cfg.logging.log_top_k_errors)
    print("\n── Top False Positives (no-drone predicted as drone) ──")
    for path, conf in errors["false_positives"]:
        print(f"  conf={conf:.3f}  {path}")
    print("\n── Top False Negatives (drone missed) ──")
    for path, conf in errors["false_negatives"]:
        print(f"  conf={conf:.3f}  {path}")


if __name__ == "__main__":
    main()
