"""3D location prediction head for monocular 3D object detection baseline.

Predicts 3D object location (X, Y, Z) in rectified camera coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Location3DOutput:
    """Output from the 3D location head.

    Attributes:
        locations: [B, num_objects, 3] predicted (X, Y, Z) in metres
        logits: [B, num_objects, 3] raw logits (same as locations for direct regression)
    """

    locations: torch.Tensor
    logits: torch.Tensor


class Location3DHead(nn.Module):
    """3D location prediction head.

    Predicts 3D object location (X, Y, Z) in rectified camera coordinates.
    X = horizontal, Y = vertical, Z = forward depth.

    Uses direct regression (no special encoding) for the baseline.
    The raw logits are the predicted locations.

    This head takes fused features from the feature pyramid and predicts
    the 3D location per object.
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

        # Location prediction MLP
        self.loc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # (X, Y, Z) in metres
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        object_indices: Optional[torch.Tensor] = None,
    ) -> Location3DOutput:
        """Forward pass.

        Args:
            features: Tuple of (P3, P4, P5) feature maps
                P3: [B, C3, H/8, W/8]
                P4: [B, C4, H/16, W/16]
                P5: [B, C5, H/32, W/32]
            object_indices: Optional [B, N] indices of objects to predict for.
                If None, predicts for a fixed grid (baseline placeholder).

        Returns:
            Location3DOutput with:
                locations: [B, N, 3] (X, Y, Z) in metres, rectified camera coords
                logits: [B, N, 3] raw predictions (same as locations)
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
            logits = self.loc_head(pooled)  # [B, 3]
            logits = logits.unsqueeze(1)  # [B, 1, 3]
        else:
            num_objects = object_indices.shape[1]
            # Broadcast pooled features for each object
            logits = self.loc_head(pooled).unsqueeze(1).expand(-1, num_objects, -1)  # [B, N, 3]

        # Direct regression - logits are the predicted locations
        locations = logits

        return Location3DOutput(
            locations=locations,
            logits=logits,
        )