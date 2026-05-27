"""
Merge the Kaggle 'dataset-balanced-n-weighted-final' data into the existing
train/val/test splits and invalidate normalisation stats.

  python scripts/add_kaggle_to_splits.py
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

KAGGLE_CSV  = Path("data/splits/kaggle_balanced_test.csv")
SPLITS_DIR  = Path("data/splits")
STATS_DIR   = Path("checkpoints")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42


def _split_stratified(rows, train_ratio, val_ratio, seed):
    by_label = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(r)
    train, val, test = [], [], []
    rng = random.Random(seed)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        t_end = max(1, int(n * train_ratio))
        v_end = t_end + max(1, int(n * val_ratio))
        train.extend(label_rows[:t_end])
        val.extend(label_rows[t_end:v_end])
        test.extend(label_rows[v_end:])
    return train, val, test


def _append_rows(csv_path, rows, split_name):
    if not rows:
        return
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        for r in rows:
            writer.writerow({"path": r["path"], "label": r["label"],
                             "source": r["source"], "split": split_name})
    print(f"  +{len(rows):5d} rows → {csv_path}")


def main():
    with open(KAGGLE_CSV) as f:
        rows = list(csv.DictReader(f))

    n_drone    = sum(1 for r in rows if int(r["label"]) == 1)
    n_no_drone = sum(1 for r in rows if int(r["label"]) == 0)
    print(f"Loaded {len(rows)} records (drone={n_drone}, no-drone={n_no_drone})")

    # Skip files already in splits
    existing: set[str] = set()
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}.csv"
        if p.exists():
            with open(p) as f:
                for r in csv.DictReader(f):
                    existing.add(r["path"])

    new_rows = [r for r in rows if r["path"] not in existing]
    print(f"New files to add: {len(new_rows)}  (skipping {len(rows)-len(new_rows)} already present)")

    if not new_rows:
        print("Nothing to do.")
        return

    train_rows, val_rows, test_rows = _split_stratified(new_rows, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    print(f"Split: {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test\n")

    _append_rows(SPLITS_DIR / "train.csv", train_rows, "train")
    _append_rows(SPLITS_DIR / "val.csv",   val_rows,   "val")
    _append_rows(SPLITS_DIR / "test.csv",  test_rows,  "test")

    for fname in ("mean.npy", "std.npy"):
        p = STATS_DIR / fname
        if p.exists():
            p.unlink()
            print(f"Deleted stale {p}")

    print("\nDone. Run: python3.10 scripts/train.py --checkpoint-name best_model_v3.pt")


if __name__ == "__main__":
    main()
