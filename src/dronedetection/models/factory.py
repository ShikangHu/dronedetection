"""Model factory — instantiate model from config."""
from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from dronedetection.models.efficientnet import EfficientNetB0Classifier
from dronedetection.models.lightweight import LightweightCNN

_REGISTRY = {
    "efficientnet_b0": EfficientNetB0Classifier,
    "lightweight_cnn": LightweightCNN,
}


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Instantiate the model specified in cfg.model.name.

    Args:
        cfg: Full hydra config.

    Returns:
        Instantiated nn.Module.
    """
    name = cfg.model.name
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_REGISTRY)}")

    kwargs: dict = {"dropout": cfg.model.dropout}
    if name == "efficientnet_b0":
        kwargs["pretrained"] = cfg.model.pretrained
        kwargs["freeze_backbone"] = cfg.training.freeze_epochs > 0

    return _REGISTRY[name](**kwargs)
