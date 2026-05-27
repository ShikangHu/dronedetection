"""
Error analysis and visualisation helpers.

- Confusion matrix plot
- ROC and PR curve plots
- Top-K hardest false positives / false negatives
- Per-source breakdown
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    save_path: Path | None = None,
) -> None:
    p = probs.numpy()
    y = labels.numpy().astype(int)
    preds = (p >= threshold).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Drone", "Drone"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── ROC and PR curves ─────────────────────────────────────────────────────────

def plot_roc_pr_curves(
    probs: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path | None = None,
) -> None:
    p = probs.numpy()
    y = labels.numpy()

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = roc_auc_score(y, p)

    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = average_precision_score(y, p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax1.legend()

    ax2.plot(recall, precision, label=f"AP = {pr_auc:.4f}")
    ax2.axhline(y=0.93, color="orange", linestyle="--", label="Prec target 0.93")
    ax2.axvline(x=0.97, color="red", linestyle="--", label="Recall target 0.97")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Hard example mining ───────────────────────────────────────────────────────

def top_k_errors(
    probs: torch.Tensor,
    labels: torch.Tensor,
    paths: list[str],
    k: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """
    Find the most confident false positives and false negatives.

    Returns:
        {
          "false_positives": [(path, confidence), ...],   # no-drone predicted as drone
          "false_negatives": [(path, confidence), ...],   # drone predicted as no-drone
        }
    """
    p = probs.numpy()
    y = labels.numpy().astype(int)

    fp_mask = (p >= 0.5) & (y == 0)
    fn_mask = (p < 0.5) & (y == 1)

    fp_indices = np.where(fp_mask)[0]
    fp_sorted = fp_indices[np.argsort(-p[fp_indices])][:k]

    fn_indices = np.where(fn_mask)[0]
    fn_sorted = fn_indices[np.argsort(p[fn_indices])][:k]

    return {
        "false_positives": [(paths[i], float(p[i])) for i in fp_sorted],
        "false_negatives": [(paths[i], float(p[i])) for i in fn_sorted],
    }


# ── Per-source breakdown ──────────────────────────────────────────────────────

def per_source_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    sources: list[str],
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Compute metrics broken down by dataset source."""
    from dronedetection.evaluation.metrics import compute_metrics

    p = probs.numpy()
    y = labels.numpy()
    src_arr = np.array(sources)
    unique_sources = sorted(set(sources))
    results = {}

    for src in unique_sources:
        mask = src_arr == src
        if mask.sum() == 0:
            continue
        m = compute_metrics(
            torch.from_numpy(p[mask]),
            torch.from_numpy(y[mask]),
            threshold=threshold,
        )
        results[src] = m

    return results
