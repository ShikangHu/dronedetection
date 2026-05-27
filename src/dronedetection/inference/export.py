"""
Model export utilities.

Supported formats
─────────────────
  • TorchScript (torch.jit.trace)  — Python-native deployment
  • ONNX (opset 17)                — cross-platform, 3-5x faster on CPU via ONNX Runtime
  • Quantized ONNX (INT8)          — ~4x smaller, ~2x faster on CPU

Usage
─────
  python scripts/export_model.py --format onnx --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from dronedetection.utils.logging import get_logger

log = get_logger(__name__)


def export_torchscript(
    model: nn.Module,
    example_input: torch.Tensor,
    out_path: Path,
) -> None:
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    log.info("TorchScript model saved → %s", out_path)


def export_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    out_path: Path,
    opset: int = 17,
) -> None:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(out_path),
            opset_version=opset,
            input_names=["spectrogram"],
            output_names=["logit"],
            dynamic_axes={
                "spectrogram": {0: "batch_size"},
                "logit": {0: "batch_size"},
            },
        )
    log.info("ONNX model (opset %d) saved → %s", opset, out_path)
    _validate_onnx(out_path, example_input)


def _validate_onnx(onnx_path: Path, example_input: torch.Tensor) -> None:
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(str(onnx_path))
        out = sess.run(None, {"spectrogram": example_input.numpy()})
        log.info("ONNX validation passed. Output shape: %s", out[0].shape)
    except ImportError:
        log.warning("onnx/onnxruntime not installed; skipping ONNX validation.")
    except Exception as e:
        log.error("ONNX validation failed: %s", e)
        raise


def export_onnx_quantized(onnx_path: Path, out_path: Path) -> None:
    """
    Apply post-training INT8 dynamic quantization to an ONNX model.
    Reduces model size by ~4x with minimal accuracy loss.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        raise RuntimeError("Install onnxruntime: pip install onnxruntime")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )
    log.info("Quantized (INT8) ONNX model saved → %s", out_path)
