"""Baseline model for monocular 3D object detection.

This module combines the YOLOv10 backbone with 3D prediction heads.

Architecture:
    Input [B, 3, H, W]
        │
        ▼
    YOLOv10 Backbone
        │
        ▼
    Feature Pyramid (P3, P4, P5)
        │
        ├── 2D Detection Head
        ├── 3D Dimension Head
        ├── 3D Location Head
        └── Orientation Head
        │
        ▼
    Structured Baseline Output
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.models.backbone.yolov10 import Yolov10Backbone, Yolov10Config
from src.models.heads.detection_2d import Detection2DHead, Detection2DOutput
from src.models.heads.dimension_3d import Dimension3DHead, Dimension3DOutput
from src.models.heads.location_3d import Location3DHead, Location3DOutput
from src.models.heads.orientation import OrientationHead, OrientationOutput


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the baseline model."""
    backbone: Yolov10Config = Yolov10Config()
    num_classes: int = 3
    # Head configurations
    det_hidden_dim: int = 256
    det_num_anchors: int = 3
    dim_hidden_dim: int = 256
    loc_hidden_dim: int = 256
    orient_hidden_dim: int = 256


@dataclass(frozen=True)
class BaselineOutput:
    """Structured output from the baseline model.

    Attributes:
        detection_2d: 2D detection predictions (class logits, bboxes, objectness)
        dimensions_3d: 3D dimension predictions (height, width, length in metres)
        locations_3d: 3D location predictions (X, Y, Z in metres, camera coords)
        orientation: Orientation predictions (rotation_y in radians)
        backbone_features: dict with feature pyramid info
    """
    detection_2d: 'Detection2DOutput'
    dimensions_3d: 'Dimension3DOutput'
    locations_3d: 'Location3DOutput'
    orientation: 'OrientationOutput'
    backbone_features: dict


class Baseline3DDetector(nn.Module):
    """Baseline 3D object detector with YOLOv10 backbone and 3D prediction heads.

    Architecture:
        Input [B, 3, H, W]
            │
            ▼
        YOLOv10 Backbone
            │
            ▼
        Feature Pyramid (P3, P4, P5)
            │
            ├── 2D Detection Head
            ├── 3D Dimension Head
            ├── 3D Location Head
            └── Orientation Head
            │
            ▼
        Structured Baseline Output
    """

    def __init__(self, config: Optional[BaselineConfig] = None) -> None:
        super().__init__()
        self.config = config or BaselineConfig()

        # Backbone
        self.backbone = Yolov10Backbone(
            variant=self.config.backbone.variant,
            pretrained=self.config.backbone.pretrained,
            input_channels=self.config.backbone.input_channels,
        )

        # Heads
        self.det_head = Detection2DHead(
            in_channels=self.backbone.feature_channels,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.det_hidden_dim,
            num_anchors=self.config.det_num_anchors,
        )

        self.dim_head = Dimension3DHead(
            in_channels=self.backbone.feature_channels,
            hidden_dim=self.config.dim_hidden_dim,
        )

        self.loc_head = Location3DHead(
            in_channels=self.backbone.feature_channels,
            hidden_dim=self.config.loc_hidden_dim,
        )

        self.orient_head = OrientationHead(
            in_channels=self.backbone.feature_channels,
            hidden_dim=self.config.orient_hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        object_indices: Optional[torch.Tensor] = None,
    ) -> BaselineOutput:
        """Forward pass returning structured predictions.

        Args:
            x: Input tensor [B, 3, H, W]
            object_indices: Optional [B, N] indices of objects to predict for.
                If None, heads predict a single object per image (placeholder).

        Returns:
            BaselineOutput with all predictions structured.
        """
        # Backbone feature extraction
        p3, p4, p5 = self.backbone(x)
        features = (p3, p4, p5)

        # 2D Detection head
        det_out = self.det_head(features)

        # 3D prediction heads
        dim_out = self.dim_head(features, object_indices)
        loc_out = self.loc_head(features, object_indices)
        orient_out = self.orient_head(features, object_indices)

        # Backbone feature info
        backbone_info = {
            "p3": p3,
            "p4": p4,
            "p5": p5,
            "backbone_channels": self.backbone.feature_channels,
            "strides": self.backbone.strides,
        }

        return BaselineOutput(
            detection_2d=det_out,
            dimensions_3d=dim_out,
            locations_3d=loc_out,
            orientation=orient_out,
            backbone_features=backbone_info,
        )

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