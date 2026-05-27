"""
Lightweight custom CNN for edge/CPU deployment.

~400K parameters, ~1.6MB (FP32).
Input:  (batch, 1, n_mels, T)
Output: (batch, 1) raw logit
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightCNN(nn.Module):
    """
    Four-block CNN: progressively halves spatial dimensions while doubling channels.
    Final global average pooling collapses spatial dims to a 256-dim vector.

    Args:
        dropout: Dropout probability before the final linear layer.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),    # (B,  32, n_mels/2,   T/2)
            ConvBlock(32, 64),   # (B,  64, n_mels/4,   T/4)
            ConvBlock(64, 128),  # (B, 128, n_mels/8,   T/8)
            ConvBlock(128, 256), # (B, 256, n_mels/16, T/16)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)    # (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)     # (B, 256, H', W')
        x = self.pool(x)        # (B, 256, 1, 1)
        x = torch.flatten(x, 1) # (B, 256)
        x = self.classifier(x)  # (B, 1)
        return x

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        """For API compatibility with EfficientNet. Uses single LR (lr_head)."""
        return [{"params": self.parameters(), "lr": lr_head}]
