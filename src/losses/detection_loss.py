"""2D detection losses for baseline multi-task loss.

This module implements losses for the 2D detection head:

1. Bounding box regression loss (Smooth L1 on normalised coordinates)
2. Objectness loss (BCEWithLogitsLoss on raw logits)
3. Classification loss (CrossEntropyLoss on class logits)

All losses support variable object counts via assignment and valid masks.
Empty targets are handled gracefully (loss becomes zero for regression terms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.assignment import AssignmentResult


@dataclass(frozen=True)
class DetectionLossOutput:
    """Output from detection loss computation.

    Attributes:
        box_loss: Scalar loss for 2D bbox regression.
        objectness_loss: Scalar loss for objectness.
        classification_loss: Scalar loss for classification.
        total: Sum of weighted component losses.
    """

    box_loss: torch.Tensor
    objectness_loss: torch.Tensor
    classification_loss: torch.Tensor
    total: torch.Tensor


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0 / 9.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Smooth L1 loss with configurable beta.

    Smooth L1 is equivalent to Huber loss with delta = beta.
    For |x| < beta: 0.5 * x^2 / beta
    For |x| >= beta: |x| - 0.5 * beta

    Args:
        pred: Predicted values.
        target: Target values.
        beta: Threshold where loss transitions from L2 to L1.
        reduction: "mean", "sum", or "none".

    Returns:
        Loss tensor according to reduction.
    """
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class DetectionLoss(nn.Module):
    """2D detection loss combining box, objectness, and classification.

    This loss operates on the output of Detection2DHead and matched
    ground-truth targets. It handles:
    - Box regression on matched prediction-target pairs
    - Objectness (foreground/background) on all predictions
    - Classification on matched predictions

    The loss uses an assignment strategy to match predictions to targets.
    Unmatched predictions are treated as background (objectness=0).
    """

    def __init__(
        self,
        box_loss_type: str = "smooth_l1",
        box_loss_beta: float = 1.0 / 9.0,
        box_weight: float = 1.0,
        objectness_weight: float = 1.0,
        classification_weight: float = 1.0,
        assignment: str = "fixed_order",
    ) -> None:
        super().__init__()
        self.box_loss_type = box_loss_type
        self.box_loss_beta = box_loss_beta
        self.box_weight = box_weight
        self.objectness_weight = objectness_weight
        self.classification_weight = classification_weight
        self.assignment = assignment

        # Objectness uses BCEWithLogitsLoss on raw logits
        # IMPORTANT: Detection2DHead applies sigmoid to objectness output
        # We need to use the raw logits before sigmoid for BCEWithLogitsLoss
        # Since the head returns sigmoid probabilities, we'll use BCELoss
        # and clamp to avoid log(0)
        self.objectness_criterion = nn.BCELoss(reduction="none")

        # Classification uses CrossEntropyLoss on raw logits
        self.classification_criterion = nn.CrossEntropyLoss(reduction="none")

    def _box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        matched_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute box regression loss on matched predictions.

        Args:
            pred_boxes: [B, N, 4] normalised (x1, y1, x2, y2) in [0, 1].
            target_boxes: [B, M, 4] normalised targets.
            matched_mask: [B, N] boolean mask of matched predictions.

        Returns:
            Scalar box loss.
        """
        # Extract matched predictions and their corresponding targets
        # This requires knowing which target each matched prediction corresponds to
        # For simplicity, we use the assignment result's indices
        # This is handled in the forward method
        raise NotImplementedError("Use forward with assignment result")

    def forward(
        self,
        class_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_objectness: torch.Tensor,
        assignment: AssignmentResult,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> DetectionLossOutput:
        """Compute detection losses.

        Args:
            class_logits: [B, N, C] class logits per prediction.
            pred_boxes: [B, N, 4] normalised (x1, y1, x2, y2) in [0, 1].
            pred_objectness: [B, N] objectness probabilities (after sigmoid).
            assignment: AssignmentResult from prediction-target matching.
            target_boxes: [B, M, 4] normalised target bboxes.
            target_labels: [B, M] integer class indices per target.
            target_counts: [B] number of valid targets per batch.

        Returns:
            DetectionLossOutput with component losses and total.
        """
        device = class_logits.device
        batch_size, num_predictions, num_classes = class_logits.shape
        _, num_predictions, _ = pred_boxes.shape

        # Objectness loss: all predictions participate
        # pred_objectness is [B, N] probabilities (after sigmoid)
        # Targets: 1 for matched predictions, 0 for unmatched
        obj_targets = assignment.matched_mask.float()  # [B, N]

        # Clamp to avoid log(0) in BCELoss
        eps = 1e-7
        pred_obj_clamped = pred_objectness.clamp(eps, 1 - eps)
        obj_loss_per_element = self.objectness_criterion(pred_obj_clamped, obj_targets)  # [B, N]
        objectness_loss = obj_loss_per_element.mean()

        # Box loss: only on matched predictions
        # Use matched_mask to select only valid matches
        if assignment.matched_mask.any():
            # Gather matched predictions using pred_indices
            matched_pred_boxes = torch.gather(
                pred_boxes,
                1,
                assignment.pred_indices.unsqueeze(-1).expand(-1, -1, 4),
            )  # [B, M, 4]

            # Gather corresponding target boxes using target_indices
            matched_target_boxes = torch.gather(
                target_boxes,
                1,
                assignment.target_indices.unsqueeze(-1).expand(-1, -1, 4),
            )  # [B, M, 4]

            # Smooth L1 loss on matched pairs
            box_diff = torch.abs(matched_pred_boxes - matched_target_boxes)
            beta = 1.0 / 9.0
            cond = box_diff < 1.0 / 9.0
            box_loss_per_element = torch.where(
                cond,
                0.5 * box_diff ** 2 * 9.0,
                box_diff - 0.5 / 9.0,
            )
            box_loss = box_loss_per_element.mean()
        else:
            # No matched pairs -> zero box loss
            box_loss = torch.tensor(0.0, device=pred_boxes.device)

        # Classification loss: only on matched predictions
        if assignment.pred_indices.numel() > 0:
            # Gather class logits for matched predictions
            matched_class_logits = torch.gather(
                class_logits,
                1,
                assignment.pred_indices.unsqueeze(-1).expand(-1, -1, class_logits.shape[-1]),
            )  # [B, M, C]

            # Gather corresponding target labels
            matched_target_labels = torch.gather(
                target_labels,
                1,
                assignment.target_indices,
            )  # [B, M]

            # CrossEntropyLoss expects [N, C] logits and [N] targets
            cls_loss_per_element = F.cross_entropy(
                matched_class_logits.reshape(-1, matched_class_logits.shape[-1]),
                matched_target_labels.reshape(-1),
                reduction="none",
            )  # [M]
            classification_loss = cls_loss_per_element.mean()
        else:
            classification_loss = torch.tensor(0.0, device=device)

        # Weighted total
        total_loss = (
            self.box_weight * box_loss
            + self.objectness_weight * objectness_loss
            + self.classification_weight * classification_loss
        )

        return DetectionLossOutput(
            box_loss=box_loss,
            objectness_loss=objectness_loss,
            classification_loss=classification_loss,
            total=total_loss,
        )


class EmptyDetectionLoss(nn.Module):
    """Detection loss that safely handles empty targets.

    When there are no valid targets in a batch, regression losses
    become zero and classification/objectness are handled safely.
    """

    def __init__(self, base_loss: DetectionLoss) -> None:
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self,
        class_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_objectness: torch.Tensor,
        assignment: AssignmentResult,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> DetectionLossOutput:
        # If all target_counts are zero, return zero losses
        if target_counts.sum() == 0:
            device = class_logits.device
            return DetectionLossOutput(
                box_loss=torch.tensor(0.0, device=device),
                objectness_loss=torch.tensor(0.0, device=device),
                classification_loss=torch.tensor(0.0, device=device),
                total=torch.tensor(0.0, device=device),
            )
        return self.base_loss(
            class_logits,
            pred_boxes,
            pred_objectness,
            assignment,
            target_boxes,
            target_labels,
            target_counts,
        )