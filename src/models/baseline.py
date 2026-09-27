"""Baseline model for monocular 3D object detection.

This module combines the YOLOv10 backbone with placeholder heads
for the baseline architecture. The heads are not yet implemented
(see EXP-003C) but the interface is defined here for integration testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.models.backbone.yolov10 import Yolov10Backbone, Yolov10Config


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the baseline model."""
    backbone: Yolov10Config = Yolov10Config()
    num_classes: int = 3


class Baseline3DDetector(nn.Module):
    """Baseline 3D object detector with YOLOv10 backbone.

    This is the EXP-003B model with only the backbone implemented.
    The 3D prediction heads will be added in EXP-003C.

    Architecture:
        Input [B, 3, H, W]
            │
            ▼
        YOLOv10 Backbone
            │
            ▼
        Feature Pyramid (P3, P4, P5)
            │
            ├── Future 2D Detection Head
            ├── Future 3D Dimension Head
            ├── Future 3D Location Head
            └── Future Orientation Head
    """

    def __init__(self, config: Optional[BaselineConfig] = None) -> None:
        super().__init__()
        self.config = config or BaselineConfig()
        self.backbone = Yolov10Backbone(
            variant=self.config.backbone.variant,
            pretrained=self.config.backbone.pretrained,
            input_channels=self.config.backbone.input_channels,
        )

        # Placeholder for future heads - will be implemented in EXP-003C
        self._heads_initialized = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning backbone feature pyramid.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Dict with:
                'p3': [B, C3, H/8, W/8]
                'p4': [B, C4, H/16, W/16]
                'p5': [B, C5, H/32, W/32]
                'backbone_channels': (C3, C4, C5)
                'strides': (8, 16, 32)
        """
        p3, p4, p5 = self.backbone(x)

        return {
            "p3": p3,
            "p4": p4,
            "p5": p5,
            "backbone_channels": self.backbone.feature_channels,
            "strides": self.backbone.strides,
        }

    @property
    def feature_channels(self) -> tuple[int, int, int]:
        """Return output channels for P3, P4, P5."""
        return self.backbone.feature_channels

    @property
    def num_parameters(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_baseline_model(config: Optional[BaselineConfig] = None) -> Baseline3DDetector:
    """Factory function to create baseline model."""
    return Baseline3DDetector(config or BaselineConfig())