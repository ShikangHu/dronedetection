"""
Dataset download helpers.

Drone (positive) sources
────────────────────────
  • DroneAudioset  — HuggingFace ahlab-drone-project/DroneAudioSet  (~23.5 h, MIT)
  • DADS           — HuggingFace geronimobasso/drone-audio-detection-samples (~5 h)
  • Al-Emadi       — GitHub saraalemadi/DroneAudioDataset (~2 h)

Non-drone (negative) sources
─────────────────────────────
  • UrbanSound8K   — https://urbansounddataset.weebly.com/urbansound8k.html  (manual)
  • ESC-50         — GitHub karolpiczak/ESC-50 (wget / git)
  • FSD50K         — Zenodo 4060432 (manual download, large)

Usage
─────
  python scripts/download_data.py --datasets droneaudioset dads esc50
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _unzip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    log.info("Extracted %s -> %s", zip_path.name, dest)


# ── drone datasets ────────────────────────────────────────────────────────────

def download_droneaudioset(raw_dir: Path) -> Path:
    """
    Download DroneAudioset from HuggingFace using the `datasets` library.
    Saves audio files to raw_dir/droneaudioset/.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError("Install `datasets`: pip install datasets")

    dest = raw_dir / "droneaudioset"
    if dest.exists():
        log.info("DroneAudioset already present at %s, skipping.", dest)
        return dest

    log.info("Downloading DroneAudioset from HuggingFace ...")
    # Dataset has 28 sub-splits: train_001 ... train_028
    ds_info = load_dataset("ahlab-drone-project/DroneAudioSet", "drone-only", split=None)
    split_names = [s for s in ds_info.keys() if s.startswith("train")]
    log.info("Found %d sub-splits: %s ... %s", len(split_names), split_names[0], split_names[-1])

    dest.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    counter = 0
    for split_name in split_names:
        ds_split = load_dataset("ahlab-drone-project/DroneAudioSet", "drone-only",
                                split=split_name)
        for sample in ds_split:
            audio = sample["audio"]
            out = dest / f"drone_{counter:06d}.wav"
            if not out.exists():
                sf.write(str(out), audio["array"], audio["sampling_rate"])
            counter += 1

    log.info("DroneAudioset saved: %d files in %s", counter, dest)
    return dest


def download_dads(raw_dir: Path) -> Path:
    """Download DADS (drone-audio-detection-samples) from HuggingFace."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError("Install `datasets`: pip install datasets")

    dest = raw_dir / "dads"
    if dest.exists():
        log.info("DADS already present at %s, skipping.", dest)
        return dest

    log.info("Downloading DADS from HuggingFace ...")
    ds = load_dataset("geronimobasso/drone-audio-detection-samples", split="train")
    dest.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(ds):
        audio = sample["audio"]
        out = dest / f"drone_{i:06d}.wav"
        if not out.exists():
            import soundfile as sf
            sf.write(str(out), audio["array"], audio["sampling_rate"])

    log.info("DADS saved: %d files in %s", i + 1, dest)
    return dest


def download_alemadi(raw_dir: Path) -> Path:
    """Clone Sara Al-Emadi DroneAudioDataset from GitHub."""
    dest = raw_dir / "alemadi"
    if dest.exists():
        log.info("Al-Emadi dataset already present, skipping.")
        return dest
    log.info("Cloning Al-Emadi DroneAudioDataset …")
    _run(["git", "clone", "--depth=1",
          "https://github.com/saraalemadi/DroneAudioDataset.git", str(dest)])
    return dest


# ── non-drone datasets ────────────────────────────────────────────────────────

def download_esc50(raw_dir: Path) -> Path:
    """Clone ESC-50 from GitHub (includes audio in .ogg format)."""
    dest = raw_dir / "esc50"
    if dest.exists():
        log.info("ESC-50 already present, skipping.")
        return dest
    log.info("Cloning ESC-50 …")
    _run(["git", "clone", "--depth=1",
          "https://github.com/karoldvl/ESC-50.git", str(dest)])
    return dest


def download_urbansound8k(raw_dir: Path) -> None:
    """
    UrbanSound8K requires manual download (registration required).
    Prints instructions if the dataset is not found.
    """
    dest = raw_dir / "urbansound8k"
    if dest.exists():
        log.info("UrbanSound8K found at %s.", dest)
        return
    log.warning(
        "UrbanSound8K not found at %s.\n"
        "Manual download required:\n"
        "  1. Register at https://urbansounddataset.weebly.com/urbansound8k.html\n"
        "  2. Download UrbanSound8K.tar.gz\n"
        "  3. Extract to %s",
        dest, dest
    )


def download_fsd50k_instructions() -> None:
    log.warning(
        "FSD50K requires manual download from Zenodo:\n"
        "  https://zenodo.org/record/4060432\n"
        "Download FSD50K.zip and extract to data/raw/fsd50k/\n"
        "Then run scripts/filter_fsd50k.py to select relevant categories."
    )


# ── entry point ───────────────────────────────────────────────────────────────

DATASET_MAP = {
    "droneaudioset": download_droneaudioset,
    "dads": download_dads,
    "alemadi": download_alemadi,
    "esc50": download_esc50,
}


def download_all(raw_dir: Path, datasets: list[str] | None = None) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = datasets or list(DATASET_MAP.keys())
    for name in targets:
        if name not in DATASET_MAP:
            log.warning("Unknown dataset '%s', skipping.", name)
            continue
        DATASET_MAP[name](raw_dir)
    download_urbansound8k(raw_dir)
    download_fsd50k_instructions()
