"""2D detection prediction head for monocular 3D object detection baseline.

This is a CUSTOM 2D detection head for the baseline architecture.
It is NOT the native YOLOv10 detection head (v10Detect).

The native YOLOv10 detection head (v10Detect) performs:
- Object classification
- Bounding box regression
- Objectness scoring

This custom head provides a clean interface for the 3D baseline:
- Predicts class logits and 2D bounding boxes per object
- Uses the feature pyramid from the backbone
- Outputs are structured for 3D baseline consumption

Do NOT confuse this with the native YOLOv10 detection head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Detection2DOutput:
    """Output from the 2D detection head.

    Attributes:
        class_logits: [B, num_objects, num_classes] class logits per object
        bboxes_2d: [B, num_objects, 4] normalised (x1, y1, x2, y2) in [0, 1]
        objectness: [B, num_objects] object confidence scores (optional)
    """

    class_logits: torch.Tensor
    bboxes_2d: torch.Tensor
    objectness: Optional[torch.Tensor] = None


class Detection2DHead(nn.Module):
    """Custom 2D detection head for the baseline architecture.

    This head takes the feature pyramid (P3, P4, P5) from the YOLOv10 backbone
    and predicts:
    - Class logits per object
    - 2D bounding box coordinates (normalised to [0, 1])
    - Objectness scores

    The head uses a simple anchor-free design on top of the feature pyramid.
    It applies a small subnetwork to each feature level and concatenates predictions.

    This is a CUSTOM head, NOT the native YOLOv10 detection head (v10Detect).
    The native v10Detect is intentionally not used to maintain a clean
    separation between the backbone feature extraction and the custom 3D
    prediction pipeline.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        num_classes: int,
        hidden_dim: int = 256,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.hidden_dim = hidden_dim

        # Shared prediction subnetwork for each feature level
        # Each level gets its own small prediction head
        self.level_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 3, padding=1),
                nn.GroupNorm(min(32, hidden_dim), hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(min(32, hidden_dim), hidden_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])

        # Class prediction head
        self.class_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors * num_classes, 1),
        )

        # Bbox regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors * 4, 1),
        )

        # Objectness head
        self.obj_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_anchors, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for prediction heads."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Detection2DOutput:
        """Forward pass.

        Args:
            features: Tuple of (P3, P4, P5) feature maps from backbone
                P3: [B, C3, H/8, W/8]
                P4: [B, C4, H/16, W/16]
                P5: [B, C5, H/32, W/32]

        Returns:
            Detection2DOutput with class_logits, bboxes_2d, objectness
            All predictions concatenated across feature levels.
            Shapes:
                class_logits: [B, N, num_classes]
                bboxes_2d: [B, N, 4] - normalised (x1, y1, x2, y2) in [0, 1]
                objectness: [B, N] - sigmoid applied
        """
        batch_size = features[0].shape[0]
        device = features[0].device

        all_class_logits = []
        all_bboxes = []
        all_objectness = []

        for feat, head in zip(features, self.level_heads):
            # feat: [B, C, H, W]
            feat = head(feat)

            # Class logits: [B, num_anchors * num_classes, H, W]
            class_logits = self.class_head(feat)
            B, _, H, W = class_logits.shape
            class_logits = class_logits.view(B, self.num_anchors, self.num_classes, H, W)
            class_logits = class_logits.permute(0, 3, 4, 1, 2).contiguous()
            class_logits = class_logits.view(B, H * W * self.num_anchors, self.num_classes)
            all_class_logits.append(class_logits)

            # Bbox: [B, num_anchors * 4, H, W]
            bbox = self.bbox_head(feat)
            bbox = bbox.view(B, self.num_anchors, 4, H, W)
            bbox = bbox.permute(0, 3, 4, 1, 2).contiguous()
            bbox = bbox.view(B, H * W * self.num_anchors, 4)
            # Sigmoid to keep in [0, 1] - normalised coordinates
            bbox = torch.sigmoid(bbox)
            all_bboxes.append(bbox)

            # Objectness: [B, num_anchors, H, W]
            obj = self.obj_head(feat)
            obj = obj.view(B, self.num_anchors, H, W)
            obj = obj.permute(0, 2, 3, 1).contiguous()
            obj = obj.view(B, H * W * self.num_anchors)
            obj = torch.sigmoid(obj)
            all_objectness.append(obj)

        # Concatenate predictions from all levels
        class_logits = torch.cat(all_class_logits, dim=1)  # [B, N, num_classes]
        bboxes = torch.cat(all_bboxes, dim=1)  # [B, N, 4]
        objectness = torch.cat(all_objectness, dim=1)  # [B, N]

        return Detection2DOutput(
            class_logits=class_logits,
            bboxes_2d=bboxes,
            objectness=objectness,
        )