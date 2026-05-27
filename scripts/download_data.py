"""
Download drone and non-drone audio datasets.

Usage
─────
  python scripts/download_data.py                          # all datasets
  python scripts/download_data.py --datasets droneaudioset dads esc50
  python scripts/download_data.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf

from dronedetection.data.download import download_all
from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="*",
                        help="Subset of: droneaudioset dads alemadi esc50")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    raw_dir = Path(cfg.paths.data_raw)

    print(f"Downloading to: {raw_dir.resolve()}")
    download_all(raw_dir, datasets=args.datasets)
    log.info("Done.")


if __name__ == "__main__":
    main()
