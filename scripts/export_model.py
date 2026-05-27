"""
Export a trained model to TorchScript or ONNX.

Usage
─────
  python scripts/export_model.py --format onnx
  python scripts/export_model.py --format torchscript
  python scripts/export_model.py --format onnx --quantize
  python scripts/export_model.py --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from dronedetection.inference.export import (
    export_onnx,
    export_onnx_quantized,
    export_torchscript,
)
from dronedetection.models.factory import build_model
from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 quantization (ONNX only)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    exports_dir = Path(cfg.paths.exports)

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    # Dummy input: batch=1, 1 channel, n_mels x T
    n_mels = cfg.data.n_mels
    sr = cfg.data.sample_rate
    seg_dur = cfg.data.segment_duration
    hop = cfg.data.hop_length
    T = int(sr * seg_dur) // hop + 1
    dummy = torch.zeros(1, 1, n_mels, T)
    log.info("Dummy input shape: %s", list(dummy.shape))

    if args.format == "torchscript":
        out = exports_dir / f"{cfg.model.name}.torchscript.pt"
        export_torchscript(model, dummy, out)

    elif args.format == "onnx":
        out = exports_dir / f"{cfg.model.name}.onnx"
        export_onnx(model, dummy, out, opset=cfg.inference.onnx_opset)
        if args.quantize:
            quant_out = exports_dir / f"{cfg.model.name}_int8.onnx"
            export_onnx_quantized(out, quant_out)


if __name__ == "__main__":
    main()
