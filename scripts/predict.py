"""
Run inference on an audio file or live microphone stream.

Usage
─────
  # Classify a file
  python scripts/predict.py --file path/to/audio.wav

  # Live stream from microphone
  python scripts/predict.py --stream
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from dronedetection.inference.predictor import FilePredictor, StreamPredictor
from dronedetection.models.factory import build_model
from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--stats-dir", default="checkpoints")
    parser.add_argument("--file", type=str, help="Path to an audio file")
    parser.add_argument("--stream", action="store_true", help="Real-time mic stream")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)

    if args.file:
        predictor = FilePredictor(model, cfg, stats_dir=args.stats_dir)
        result = predictor.predict(args.file)
        status = "DRONE DETECTED" if result["drone_detected"] else "No drone"
        print(f"\n{status}  (confidence={result['confidence']:.3f})")
        print(f"Segment probabilities: {[f'{p:.3f}' for p in result['segment_probs']]}")

    elif args.stream:
        predictor = StreamPredictor(model, cfg, stats_dir=args.stats_dir)
        predictor.start()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
