"""
Download the Kaggle 'dataset-balanced-n-weighted-final' dataset and evaluate
best_model_v2.pt against it.

Usage
─────
  python scripts/fetch_kaggle_balanced_test.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

DATASET = "yehiellevi/dataset-balanced-n-weighted-final"
DEST_DIR = Path("data/raw/kaggle_balanced")
CSV_OUT = Path("data/splits/kaggle_balanced_test.csv")
CHECKPOINT = "checkpoints/best_model_v2.pt"
META_CSV = DEST_DIR / "audio_metadata_shuffled.csv"


def download_dataset() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset from Kaggle …")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            DATASET,
            "-p", str(DEST_DIR),
            "--unzip",
        ],
        check=True,
    )
    print("Download complete.")


def build_manifest() -> None:
    if not META_CSV.exists():
        sys.exit(f"Metadata CSV not found at {META_CSV}")

    rows: list[dict] = []
    with open(META_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["slice_file_name"].strip()
            fold = f"fold{row['fold'].strip()}"
            label = int(row["classID"].strip())
            path = DEST_DIR / fold / fname
            if not path.exists():
                print(f"  [warn] missing file: {path}")
                continue
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "source": f"kaggle_balanced_{fold}",
                    "split": "external_test",
                }
            )

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)

    drone = sum(1 for r in rows if r["label"] == 1)
    non_drone = sum(1 for r in rows if r["label"] == 0)
    print(f"Manifest written: {len(rows)} files  (drone={drone}, no-drone={non_drone})")
    print(f"  → {CSV_OUT}")


def run_evaluate() -> None:
    print(f"\nRunning evaluation with {CHECKPOINT} …\n")
    subprocess.run(
        [
            sys.executable, "scripts/evaluate.py",
            "--checkpoint", CHECKPOINT,
            "--csv", str(CSV_OUT),
        ],
        check=True,
    )


if __name__ == "__main__":
    download_dataset()
    build_manifest()
    run_evaluate()
