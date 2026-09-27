"""3D dimension prediction head for monocular 3D object detection baseline.

Predicts 3D object dimensions (height, width, length) in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Dimension3DOutput:
    """Output from the 3D dimension head.

    Attributes:
        dimensions: [B, num_objects, 3] predicted (height, width, length) in metres
        logits: [B, num_objects, 3] raw logits before positive transform (for loss)
    """

    dimensions: torch.Tensor
    logits: torch.Tensor


class Dimension3DHead(nn.Module):
    """3D dimension prediction head.

    Predicts 3D object dimensions (height, width, length) in metres.

    Uses exponential transform (exp) to ensure positive dimensions.
    The raw logits are also returned for loss computation.

    The head uses a small MLP applied to pooled features per object.
    In practice, this would be applied to ROI-aligned features per detected object.
    For the baseline, we use a simple global pooling approach on the feature pyramid.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        hidden_dim: int = 256,
        num_objects_max: int = 100,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_objects_max = num_objects_max

        # Fuse feature pyramid channels to hidden_dim
        total_channels = sum(in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, hidden_dim, 1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Dimension prediction MLP
        self.dim_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # raw logits for (h, w, l)
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        object_indices: Optional[torch.Tensor] = None,
    ) -> Dimension3DOutput:
        """Forward pass.

        Args:
            features: Tuple of (P3, P4, P5) feature maps
                P3: [B, C3, H/8, W/8]
                P4: [B, C4, H/16, W/16]
                P5: [B, C5, H/32, W/32]
            object_indices: Optional [B, N] indices of objects to predict for.
                If None, predicts for a fixed grid (baseline placeholder).

        Returns:
            Dimension3DOutput with:
                dimensions: [B, N, 3] positive (height, width, length) in metres
                logits: [B, N, 3] raw logits before exp
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

        # For baseline: use adaptive pooling to get fixed-size features per object
        # This is a placeholder - in practice would use ROI Align on detected boxes
        pooled = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)  # [B, hidden_dim]

        # For baseline, we predict a fixed number of objects (or use object_indices)
        # This is a simplified implementation
        if object_indices is None:
            # Predict for a single "object" per image (placeholder)
            logits = self.dim_head(pooled)  # [B, 3]
            logits = logits.unsqueeze(1)  # [B, 1, 3]
        else:
            num_objects = object_indices.shape[1]
            # In practice, would gather features for each object
            # For now, broadcast pooled features
            logits = self.dim_head(pooled).unsqueeze(1).expand(-1, num_objects, -1)  # [B, N, 3]

        # Exponential transform for positive dimensions
        dimensions = torch.exp(logits)  # [B, N, 3]

        return Dimension3DOutput(
            dimensions=dimensions,
            logits=logits,
        )

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode raw logits to positive dimensions.

        Args:
            logits: [B, N, 3] raw logits

        Returns:
            dimensions: [B, N, 3] positive (height, width, length) in metres
        """
        return torch.exp(logits)