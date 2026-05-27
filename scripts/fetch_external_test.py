"""
Download the Zenodo "Drone Detection Thesis" dataset (record 5500576),
auto-label audio files based on folder names, write a CSV manifest, and
optionally run evaluate.py on it.

Dataset: https://zenodo.org/records/5500576
Size:    ~311 MB (ZIP)
Labels:  folders containing 'drone'           → 1  (drone)
         folders containing 'background',
         'noise', 'helicopter', 'ambient',
         or 'other'                            → 0  (no-drone)

Usage
─────
  python scripts/fetch_external_test.py
  python scripts/fetch_external_test.py --no-eval        # just download & prep
  python scripts/fetch_external_test.py --out-csv data/splits/external_test.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

ZENODO_URL = (
    "https://zenodo.org/api/records/5500576/files/"
    "DroneDetectionThesis/Drone-detection-dataset-v1.0.0.zip/content"
)
DEFAULT_DEST = Path("data/raw/zenodo_drone_thesis")
DEFAULT_CSV  = Path("data/splits/external_test.csv")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}

# Folder-name keywords → label mapping (checked case-insensitively)
DRONE_KEYWORDS     = {"drone", "uav", "quadrotor", "quadcopter"}
NO_DRONE_KEYWORDS  = {"background", "noise", "helicopter", "ambient",
                      "other", "silence", "bird", "wind", "car", "traffic"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _progress_hook(count: int, block_size: int, total: int) -> None:
    done = count * block_size
    pct  = min(100, done * 100 // total) if total > 0 else 0
    bar  = "#" * (pct // 2)
    print(f"\r  [{bar:<50}] {pct:3d}%  ({done/1e6:.1f} / {total/1e6:.1f} MB)",
          end="", flush=True)


def download_zip(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "dataset.zip"
    if zip_path.exists():
        print(f"  ZIP already present at {zip_path}, skipping download.")
        return zip_path
    print(f"  Downloading from Zenodo → {zip_path}")
    urllib.request.urlretrieve(url, zip_path, reporthook=_progress_hook)
    print()
    return zip_path


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    marker = dest_dir / ".extracted"
    if marker.exists():
        print(f"  Already extracted at {dest_dir}, skipping.")
        return
    print(f"  Extracting {zip_path.name} → {dest_dir} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    marker.touch()
    print("  Extraction complete.")


def _infer_label(file_path: Path) -> int | None:
    """
    Infer binary label from filename stem first (highest priority), then
    directory parts.  Filename takes precedence so that a parent folder like
    'DroneDetectionThesis-...' doesn't override a file named BACKGROUND_001.wav.
    Returns 1 (drone), 0 (no-drone), or None (unknown).
    """
    # 1. Check filename stem first — most reliable signal
    stem = file_path.stem.lower()
    for kw in DRONE_KEYWORDS:
        if kw in stem:
            return 1
    for kw in NO_DRONE_KEYWORDS:
        if kw in stem:
            return 0

    # 2. Fall back to directory parts (skip the filename itself)
    for part in [p.lower() for p in file_path.parent.parts]:
        for kw in DRONE_KEYWORDS:
            if kw in part:
                return 1
        for kw in NO_DRONE_KEYWORDS:
            if kw in part:
                return 0
    return None


def build_csv(audio_root: Path, csv_path: Path, source: str = "zenodo_thesis") -> dict:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped_unknown = 0

    for fpath in sorted(audio_root.rglob("*")):
        if fpath.suffix.lower() not in AUDIO_EXTS:
            continue
        label = _infer_label(fpath)
        if label is None:
            skipped_unknown += 1
            continue
        rows.append({
            "path":   str(fpath),
            "label":  label,
            "source": source,
            "split":  "external_test",
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)

    n_drone    = sum(r["label"] == 1 for r in rows)
    n_no_drone = sum(r["label"] == 0 for r in rows)
    print(f"\n  CSV written → {csv_path}")
    print(f"  Drone (1):    {n_drone:>6}")
    print(f"  No-drone (0): {n_no_drone:>6}")
    if skipped_unknown:
        print(f"  Skipped (unknown label): {skipped_unknown}")
    return {"n_drone": n_drone, "n_no_drone": n_no_drone, "csv_path": csv_path}


def print_tree(root: Path, depth: int = 3, indent: str = "") -> None:
    """Print directory tree up to `depth` levels."""
    if depth == 0:
        return
    try:
        entries = sorted(root.iterdir())
    except PermissionError:
        return
    dirs   = [e for e in entries if e.is_dir()]
    files  = [e for e in entries if e.is_file()]
    for d in dirs[:6]:
        print(f"{indent}  {d.name}/")
        print_tree(d, depth - 1, indent + "  ")
    if files:
        shown = files[:4]
        for f in shown:
            print(f"{indent}  {f.name}")
        if len(files) > 4:
            print(f"{indent}  … ({len(files) - 4} more files)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_DEST),
                        help="Directory to download and extract the dataset into")
    parser.add_argument("--out-csv", default=str(DEFAULT_CSV),
                        help="Path to write the manifest CSV")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip running evaluate.py after preparation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    csv_path = Path(args.out_csv)

    # ── 1. Download ────────────────────────────────────────────────────────────
    print("\n[1/3] Downloading dataset …")
    zip_path = download_zip(ZENODO_URL, dest_dir)

    # ── 2. Extract ─────────────────────────────────────────────────────────────
    print("\n[2/3] Extracting …")
    extract_zip(zip_path, dest_dir)

    print("\n  Discovered directory structure:")
    print_tree(dest_dir)

    # ── 3. Build CSV ──────────────────────────────────────────────────────────
    print("\n[3/3] Building manifest CSV …")
    info = build_csv(dest_dir, csv_path)

    if info["n_drone"] == 0 and info["n_no_drone"] == 0:
        print("\n  No audio files labelled — check the directory tree above.")
        print("  You may need to add more keywords to DRONE_KEYWORDS / NO_DRONE_KEYWORDS.")
        sys.exit(1)

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    if not args.no_eval:
        print("\n[4/4] Running evaluate.py …\n")
        cmd = [
            sys.executable, "scripts/evaluate.py",
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--csv", str(csv_path),
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"\nDataset ready. To evaluate run:")
        print(f"  python scripts/evaluate.py --csv {csv_path}")


if __name__ == "__main__":
    main()
