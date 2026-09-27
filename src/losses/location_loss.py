"""3D location regression loss for baseline.

The model predicts 3D location (X, Y, Z) in rectified camera coordinates
using direct regression. This loss operates on the predicted locations
in metres.

Important: The loss is on X, Y, Z camera coordinates (metres).
Z is camera-axis depth, NOT Euclidean distance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LocationLoss(nn.Module):
    """3D location regression loss.

    The model predicts 3D location (X, Y, Z) in rectified camera coordinates
    using direct regression. This loss operates on the predicted locations
    in metres.

    Uses Smooth L1 loss on (X, Y, Z) in metres.
    Z is camera-axis depth, NOT Euclidean distance.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        beta: float = 1.0 / 9.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.weight = weight

    def forward(
        self,
        pred_locations: torch.Tensor,
        pred_logits: torch.Tensor,
        target_locations: torch.Tensor,
        assignment_matched_mask: torch.Tensor,
        assignment_pred_indices: torch.Tensor,
        assignment_target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute location regression loss.

        Args:
            pred_locations: [B, N, 3] predicted (X, Y, Z) in metres.
            pred_logits: [B, N, 3] raw logits (same as locations for direct regression).
            target_locations: [B, M, 3] target (X, Y, Z) in metres.
            assignment_matched_mask: [B, N] boolean mask of matched predictions.
            assignment_pred_indices: [B, M] indices of matched predictions.
            assignment_target_indices: [B, M] indices of matched targets.

        Returns:
            (location_loss, total_loss) as scalar tensors.
        """
        device = pred_locations.device

        if assignment_pred_indices.numel() == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Gather matched predictions
        matched_pred_locs = torch.gather(
            pred_locations,
            1,
            assignment_pred_indices.unsqueeze(-1).expand(-1, -1, 3),
        )  # [B, M, 3]

        # Gather matched targets
        matched_target_locs = torch.gather(
            target_locations,
            1,
            assignment_target_indices.unsqueeze(-1).expand(-1, -1, 3),
        )  # [B, M, 3]

        # Smooth L1 loss
        diff = torch.abs(matched_pred_locs - matched_target_locs)
        cond = diff < self.beta
        loss = torch.where(cond, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        location_loss = loss.mean()

        total = location_loss * self.weight
        return location_loss, total