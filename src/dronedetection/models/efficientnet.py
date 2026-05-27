"""
EfficientNet-B0 classifier for drone audio binary classification.

Input:  (batch, 1, n_mels, T) log-mel spectrogram
Output: (batch, 1) raw logit — pass through sigmoid for probability

The single-channel spectrogram is broadcast to 3 channels before entering
the ImageNet-pretrained EfficientNet-B0 backbone, which expects (B, 3, H, W).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with a custom binary classification head.

    Args:
        pretrained: Load ImageNet weights for the backbone.
        dropout: Dropout probability before the final linear layer.
        freeze_backbone: If True, all backbone parameters are frozen
                         (used during Phase 1 of two-phase training).
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Remove the original classifier (1000-class head)
        self.features = backbone.features       # Conv + MBConv blocks
        self.avgpool = backbone.avgpool          # AdaptiveAvgPool2d(1,1)
        in_features = backbone.classifier[1].in_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T) — single-channel log-mel spectrogram

        Returns:
            (B, 1) raw logit
        """
        # Broadcast mono channel to 3 channels for pretrained weights
        x = x.repeat(1, 3, 1, 1)           # (B, 3, n_mels, T)
        x = self.features(x)               # (B, 1280, H', W')
        x = self.avgpool(x)                # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)            # (B, 1280)
        x = self.classifier(x)             # (B, 1)
        return x

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        """Return param groups with separate LRs for backbone vs head."""
        return [
            {"params": self.features.parameters(), "lr": lr_backbone},
            {"params": list(self.avgpool.parameters()) + list(self.classifier.parameters()),
             "lr": lr_head},
        ]
