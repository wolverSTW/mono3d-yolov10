"""3D dimension regression loss for baseline.

The model predicts dimensions as exp(logits) to ensure positivity.
This loss operates on the decoded physical dimensions (h, w, l) in metres.

Uses Smooth L1 loss on the physical dimensions in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DimensionLossOutput:
    """Output from dimension loss computation.

    Attributes:
        dimension_loss: Scalar loss for 3D dimension regression.
        total: Same as dimension_loss (for consistent interface).
    """

    dimension_loss: torch.Tensor
    total: torch.Tensor


class DimensionLoss(nn.Module):
    """3D dimension regression loss.

    The model predicts dimensions as exp(logits) to ensure positive values.
    This loss operates on the decoded physical dimensions (h, w, l) in metres.

    Uses Smooth L1 loss on physical dimensions (h, w, l) in metres.
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
        pred_dimensions: torch.Tensor,
        pred_logits: torch.Tensor,
        target_dimensions: torch.Tensor,
        assignment_matched_mask: torch.Tensor,
        assignment_pred_indices: torch.Tensor,
        assignment_target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dimension regression loss.

        Args:
            pred_dimensions: [B, N, 3] predicted (h, w, l) in metres (already exp'd).
            pred_logits: [B, N, 3] raw logits before exp (for gradient flow).
            target_dimensions: [B, M, 3] target (h, w, l) in metres.
            assignment_matched_mask: [B, N] boolean mask of matched predictions.
            assignment_pred_indices: [B, M] indices of matched predictions.
            assignment_target_indices: [B, M] indices of matched targets.

        Returns:
            (dimension_loss, total_loss) as scalar tensors.
        """
        device = pred_dimensions.device

        if assignment_pred_indices.numel() == 0:
            # No matched pairs -> zero loss
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Gather matched predictions
        matched_pred_dims = torch.gather(
            pred_dimensions,
            1,
            assignment_pred_indices.unsqueeze(-1).expand(-1, -1, 3),
        )  # [B, M, 3]

        # Gather matched targets
        matched_target_dims = torch.gather(
            target_dimensions,
            1,
            assignment_target_indices.unsqueeze(-1).expand(-1, -1, 3),
        )  # [B, M, 3]

        # Smooth L1 loss
        diff = torch.abs(matched_pred_dims - matched_target_dims)
        cond = diff < self.beta
        loss = torch.where(cond, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        dimension_loss = loss.mean()

        total = dimension_loss * self.weight
        return dimension_loss, total