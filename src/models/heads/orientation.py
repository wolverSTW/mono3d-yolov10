"""Orientation prediction head for monocular 3D object detection baseline.

Predicts object orientation as rotation_y (camera-frame yaw) in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OrientationOutput:
    """Output from the orientation head.

    Attributes:
        rotation_y: [B, num_objects] predicted rotation_y in radians
        logits: [B, num_objects, 2] or [B, num_objects] raw predictions
    """

    rotation_y: torch.Tensor
    logits: torch.Tensor


class OrientationHead(nn.Module):
    """Orientation prediction head.

    Predicts object orientation as rotation_y (camera-frame yaw) in radians.

    For the baseline, we use direct regression of rotation_y.
    Range: [-pi, pi] (radians).

    An alternative encoding (not used in baseline) would be:
    - sin/cos encoding: predict sin(rotation_y) and cos(rotation_y)
    - then decode with atan2(sin, cos)
    This is NOT implemented in the baseline but documented for reference.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Fuse feature pyramid channels to hidden_dim
        total_channels = sum(in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, hidden_dim, 1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Orientation prediction MLP (direct regression)
        self.orient_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),  # rotation_y in radians
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        object_indices: Optional[torch.Tensor] = None,
    ) -> OrientationOutput:
        """Forward pass.

        Args:
            features: Tuple of (P3, P4, P5) feature maps
                P3: [B, C3, H/8, W/8]
                P4: [B, C4, H/16, W/16]
                P5: [B, C5, H/32, W/32]
            object_indices: Optional [B, N] indices of objects to predict for.
                If None, predicts for a fixed grid (baseline placeholder).

        Returns:
            OrientationOutput with:
                rotation_y: [B, N] predicted rotation_y in radians
                logits: [B, N, 1] raw predictions (same as rotation_y for direct regression)
        """
        batch_size = features[0].shape[0]
        device = features[0].device

        # Fuse feature pyramid: upsample all to P3 resolution and concatenate
        p3, p4, p5 = features
        _, _, h3, w3 = p3.shape

        p4_up = F.interpolate(p4, size=(h3, w3), mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=(h3, w3), mode='bilinear', align_corners=False)

        fused = torch.cat([p3, p4_up, p5_up], dim=1)  # [B, C3+C4+C5, H/8, W/8]
        fused = self.fusion(fused)  # [B, hidden_dim, H/8, W/8]

        # Adaptive pooling to get fixed-size features
        pooled = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)  # [B, hidden_dim]

        if object_indices is None:
            # Predict for a single "object" per image (placeholder)
            logits = self.orient_head(pooled)  # [B, 1]
            logits = logits.unsqueeze(1)  # [B, 1, 1]
        else:
            num_objects = object_indices.shape[1]
            # Broadcast pooled features for each object
            logits = self.orient_head(pooled).unsqueeze(1).expand(-1, num_objects, -1)  # [B, N, 1]

        # Direct regression - logits are the predicted rotation_y
        rotation_y = logits.squeeze(-1)  # [B, N]

        return OrientationOutput(
            rotation_y=rotation_y,
            logits=logits,
        )

    @staticmethod
    def decode_sincos(sin_cos: torch.Tensor) -> torch.Tensor:
        """Decode sin/cos encoding to rotation_y.

        Args:
            sin_cos: [B, N, 2] where [:, :, 0] = sin(theta), [:, :, 1] = cos(theta)

        Returns:
            rotation_y: [B, N] in radians, range [-pi, pi]
        """
        sin_theta = sin_cos[..., 0]
        cos_theta = sin_cos[..., 1]
        return torch.atan2(sin_theta, cos_theta)

    @staticmethod
    def encode_sincos(rotation_y: torch.Tensor) -> torch.Tensor:
        """Encode rotation_y to sin/cos representation.

        Args:
            rotation_y: [B, N] in radians

        Returns:
            sin_cos: [B, N, 2] where [:, :, 0] = sin(theta), [:, :, 1] = cos(theta)
        """
        return torch.stack([torch.sin(rotation_y), torch.cos(rotation_y)], dim=-1)