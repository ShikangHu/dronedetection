"""
Merge the Zenodo "Drone Detection Thesis" data into the existing train/val/test
splits and refresh the normalisation statistics.

What it does
────────────
  1. Reads data/splits/external_test.csv (written by fetch_external_test.py)
  2. Splits the 90 files *stratified by label* at the recording level:
       70% → train, 15% → val, 15% → test
  3. Appends the new rows to data/splits/{train,val,test}.csv
  4. Deletes checkpoints/mean.npy and checkpoints/std.npy so that
     train.py recomputes the normalisation stats on the augmented training set

Run BEFORE re-training:
  python scripts/add_zenodo_to_splits.py

To verify, check the last few lines of the split CSVs:
  tail -n 5 data/splits/train.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


EXTERNAL_CSV  = Path("data/splits/external_test.csv")
SPLITS_DIR    = Path("data/splits")
STATS_DIR     = Path("checkpoints")

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
RANDOM_SEED   = 42


def _split_stratified(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows stratified by label into train / val / test lists."""
    by_label: dict[int, list[dict]] = defaultdict(list)
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


def _append_rows(csv_path: Path, rows: list[dict], split_name: str) -> None:
    """Append rows (with split field overridden) to an existing CSV."""
    if not rows:
        return
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        for r in rows:
            writer.writerow({
                "path":   r["path"],
                "label":  r["label"],
                "source": r["source"],
                "split":  split_name,
            })
    print(f"  +{len(rows):3d} rows → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-csv", default=str(EXTERNAL_CSV),
                        help="Path to the external_test.csv written by fetch_external_test.py")
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--stats-dir", default=str(STATS_DIR))
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    ext_csv   = Path(args.external_csv)
    splits_dir = Path(args.splits_dir)
    stats_dir  = Path(args.stats_dir)

    # ── 1. Load external CSV ──────────────────────────────────────────────────
    if not ext_csv.exists():
        raise FileNotFoundError(
            f"{ext_csv} not found. Run scripts/fetch_external_test.py first."
        )

    with open(ext_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_drone    = sum(1 for r in rows if int(r["label"]) == 1)
    n_no_drone = sum(1 for r in rows if int(r["label"]) == 0)
    print(f"\nLoaded {len(rows)} records from {ext_csv}")
    print(f"  Drone (1):    {n_drone}")
    print(f"  No-drone (0): {n_no_drone}")

    # ── 2. Check for existing overlap ─────────────────────────────────────────
    existing_paths: set[str] = set()
    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    existing_paths.add(r["path"])

    new_rows = [r for r in rows if r["path"] not in existing_paths]
    skipped  = len(rows) - len(new_rows)
    if skipped:
        print(f"  Skipping {skipped} files already present in splits.")
    if not new_rows:
        print("\nAll Zenodo files are already in the splits. Nothing to do.")
        return
    print(f"  Adding {len(new_rows)} new files to splits.")

    # ── 3. Stratified split ───────────────────────────────────────────────────
    train_rows, val_rows, test_rows = _split_stratified(
        new_rows, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"\n  Split: {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test")

    # ── 4. Append to CSVs ─────────────────────────────────────────────────────
    print("\nAppending to split CSVs:")
    _append_rows(splits_dir / "train.csv", train_rows, "train")
    _append_rows(splits_dir / "val.csv",   val_rows,   "val")
    _append_rows(splits_dir / "test.csv",  test_rows,  "test")

    # ── 5. Invalidate normalisation stats ─────────────────────────────────────
    deleted = []
    for fname in ("mean.npy", "std.npy"):
        p = stats_dir / fname
        if p.exists():
            p.unlink()
            deleted.append(str(p))
    if deleted:
        print(f"\nDeleted stale normalisation stats: {', '.join(deleted)}")
        print("  train.py will recompute them on the next run.")
    else:
        print("\nNo existing normalisation stats found (will be computed at training time).")

    print("\nDone. Re-run scripts/train.py to retrain with the augmented dataset.")


if __name__ == "__main__":
    main()
