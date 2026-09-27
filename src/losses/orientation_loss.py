"""Orientation regression loss for baseline.

The model predicts rotation_y (camera-frame yaw) in radians using direct
regression. This loss uses angular difference to handle periodicity.

The angular difference is computed as:
    delta = atan2(sin(pred - target), cos(pred - target))

This gives the shortest signed angular difference in [-pi, pi].
Then Smooth L1 is applied to delta.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OrientationLoss(nn.Module):
    """Orientation regression loss with angular periodicity handling.

    The model predicts rotation_y (camera-frame yaw) in radians.
    This loss computes the shortest angular difference and applies
    Smooth L1 loss on the angular difference.

    The angular difference is computed as:
        delta = atan2(sin(pred - target), cos(pred - target))

    This gives the signed shortest angular difference in [-pi, pi],
    correctly handling the wrap-around at +/- pi.
    """

    def __init__(
        self,
        loss_type: str = "angular_smooth_l1",
        beta: float = 1.0 / 9.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta
        self.weight = weight

    def forward(
        self,
        pred_rotation_y: torch.Tensor,
        pred_logits: torch.Tensor,
        target_rotation_y: torch.Tensor,
        assignment_pred_indices: torch.Tensor,
        assignment_target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute orientation regression loss with angular periodicity.

        Args:
            pred_rotation_y: [B, N] predicted rotation_y in radians.
            pred_logits: [B, N, 1] raw logits (same as rotation_y for direct regression).
            target_rotation_y: [B, M] target rotation_y in radians.
            assignment_pred_indices: [B, M] indices of matched predictions.
            assignment_target_indices: [B, M] indices of matched targets.

        Returns:
            (orientation_loss, total_loss) as scalar tensors.
        """
        device = pred_rotation_y.device

        if assignment_pred_indices.numel() == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Gather matched predictions
        matched_pred_rot = torch.gather(
            pred_rotation_y,
            1,
            assignment_pred_indices,
        )  # [B, M]

        # Gather matched targets
        matched_target_rot = torch.gather(
            target_rotation_y,
            1,
            assignment_target_indices,
        )  # [B, M]

        # Compute angular difference with periodicity handling
        # delta = atan2(sin(pred - target), cos(pred - target))
        # This gives the shortest signed angular difference in [-pi, pi]
        delta = pred_rotation_y - target_rotation_y
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))  # [-pi, pi]

        # Smooth L1 on angular difference
        diff = torch.abs(delta)
        cond = diff < 1.0 / 9.0
        loss = torch.where(cond, 0.5 * diff ** 2 * 9.0, diff - 0.5 / 9.0)
        orientation_loss = loss.mean()

        total = orientation_loss * self.weight
        return orientation_loss, total