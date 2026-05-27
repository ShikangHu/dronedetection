"""
Download the FPV Drone Audio Classification Dataset (GitHub: saarmets/fpv-drone-audio-classification-dataset),
auto-label files based on folder names, write a CSV manifest, and optionally run evaluate.py.

Dataset contents (~856 MB):
  fpv5/       → 9 WAV files   (FPV drone model 5)    → label 1
  fpv7/       → 12 WAV files  (FPV drone model 7)    → label 1
  no_drone/   → 16 WAV files  (ambient, car, wind…)  → label 0

Usage
─────
  python scripts/fetch_fpv_drone_test.py
  python scripts/fetch_fpv_drone_test.py --no-eval
  python scripts/fetch_fpv_drone_test.py --out-csv data/splits/fpv_test.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

GITHUB_API  = "https://api.github.com/repos/saarmets/fpv-drone-audio-classification-dataset/contents"
DEFAULT_DEST = Path("data/raw/fpv_drone_dataset")
DEFAULT_CSV  = Path("data/splits/fpv_test.csv")

# folder name → label
FOLDER_LABELS: dict[str, int] = {
    "fpv5":     1,
    "fpv7":     1,
    "no_drone": 0,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _progress_hook(count: int, block_size: int, total: int) -> None:
    done = count * block_size
    pct  = min(100, done * 100 // total) if total > 0 else 0
    bar  = "#" * (pct // 2)
    print(f"\r  [{bar:<50}] {pct:3d}%  ({done/1e6:.1f} / {total/1e6:.1f} MB)",
          end="", flush=True)


def fetch_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "drone-detection-fetcher/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def download_folder(folder: str, label: int, dest_dir: Path) -> list[dict]:
    """Download all WAV files from one GitHub folder. Returns list of manifest rows."""
    out_dir = dest_dir / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Fetching file list for {folder}/ …")
    items = fetch_json(f"{GITHUB_API}/{folder}")
    wav_items = [it for it in items if it["name"].lower().endswith(".wav")]
    print(f"  Found {len(wav_items)} WAV files (label={label})")

    rows: list[dict] = []
    for it in wav_items:
        dest_file = out_dir / it["name"]
        if dest_file.exists():
            print(f"    {it['name']} — already present, skipping.")
        else:
            print(f"    Downloading {it['name']} ({it.get('size', 0) / 1e6:.1f} MB) …")
            urllib.request.urlretrieve(it["download_url"], dest_file, reporthook=_progress_hook)
            print()
        rows.append({
            "path":   str(dest_file),
            "label":  label,
            "source": f"fpv_drone_{folder}",
            "split":  "external_test",
        })
    return rows


def build_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)
    n_drone    = sum(1 for r in rows if int(r["label"]) == 1)
    n_no_drone = sum(1 for r in rows if int(r["label"]) == 0)
    print(f"\n  CSV written → {csv_path}")
    print(f"  Drone (1):    {n_drone}")
    print(f"  No-drone (0): {n_no_drone}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--out-csv", default=str(DEFAULT_CSV))
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model_v2.pt")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    csv_path = Path(args.out_csv)

    # ── 1. Download ────────────────────────────────────────────────────────────
    print("\n[1/2] Downloading FPV drone dataset from GitHub …")
    all_rows: list[dict] = []
    for folder, label in FOLDER_LABELS.items():
        all_rows.extend(download_folder(folder, label, dest_dir))

    # ── 2. Build CSV ───────────────────────────────────────────────────────────
    print("\n[2/2] Building manifest CSV …")
    build_csv(all_rows, csv_path)

    # ── 3. Evaluate ────────────────────────────────────────────────────────────
    if not args.no_eval:
        print("\n[3/3] Running evaluate.py …\n")
        cmd = [
            sys.executable, "scripts/evaluate.py",
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--csv", str(csv_path),
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"\nDataset ready. To evaluate run:")
        print(f"  python3 scripts/evaluate.py --checkpoint checkpoints/best_model_v2.pt --csv {csv_path}")


if __name__ == "__main__":
    main()
