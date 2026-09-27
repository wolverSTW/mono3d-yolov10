"""Multi-task total loss aggregator for baseline.

Combines individual losses with configurable weights into a single
loss value for optimization. Returns both total and component losses
for monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.losses.detection_loss import DetectionLoss, DetectionLossOutput
from src.losses.dimension_loss import DimensionLoss
from src.losses.location_loss import LocationLoss
from src.losses.orientation_loss import OrientationLoss
from src.losses.assignment import assign_fixed_order, AssignmentResult


@dataclass(frozen=True)
class TotalLossOutput:
    """Output from total loss computation.

    Attributes:
        total: Weighted sum of all component losses.
        detection: DetectionLossOutput with box, objectness, classification.
        dimension_3d: Scalar dimension loss.
        location_3d: Scalar location loss.
        orientation: Scalar orientation loss.
        weights: Dict of loss weights used.
    """

    total: torch.Tensor
    detection: DetectionLossOutput
    dimension_3d: torch.Tensor
    location_3d: torch.Tensor
    orientation: torch.Tensor
    weights: dict[str, float]


class TotalLoss(nn.Module):
    """Multi-task baseline loss aggregator.

    Combines:
    - 2D detection: box regression + objectness + classification
    - 3D dimension regression
    - 3D location regression
    - Orientation regression

    Uses fixed-order assignment for matching predictions to targets.
    All losses are weighted and summed.
    """

    def __init__(
        self,
        num_classes: int,
        box_weight: float = 1.0,
        objectness_weight: float = 1.0,
        classification_weight: float = 1.0,
        dimension_3d_weight: float = 1.0,
        location_3d_weight: float = 1.0,
        orientation_weight: float = 1.0,
        box_loss_beta: float = 1.0 / 9.0,
        assignment: str = "fixed_order",
    ) -> None:
        super().__init__()
        # Store config for inspection
        self.config = type('Config', (), {
            'num_classes': num_classes,
        })()

        self.num_classes = num_classes

        # Store weights
        self.weights = {
            "box": box_weight,
            "objectness": objectness_weight,
            "classification": classification_weight,
            "dimension_3d": dimension_3d_weight,
            "location_3d": location_3d_weight,
            "orientation": orientation_weight,
        }

        # Individual loss modules
        self.detection_loss = DetectionLoss(
            box_loss_type="smooth_l1",
            box_loss_beta=box_loss_beta,
            box_weight=box_weight,
            objectness_weight=objectness_weight,
            classification_weight=classification_weight,
            assignment=assignment,
        )

        self.dimension_loss = DimensionLoss(
            loss_type="smooth_l1",
            weight=dimension_3d_weight,
        )

        self.location_loss = LocationLoss(
            loss_type="smooth_l1",
            weight=location_3d_weight,
        )

        self.orientation_loss = OrientationLoss(
            loss_type="angular_smooth_l1",
            weight=orientation_weight,
        )

    def forward(
        self,
        # Model outputs
        class_logits: torch.Tensor,           # [B, N, C]
        pred_boxes: torch.Tensor,             # [B, N, 4] normalised
        pred_objectness: torch.Tensor,        # [B, N] probabilities
        pred_dimensions: torch.Tensor,        # [B, N, 3] metres
        pred_dim_logits: torch.Tensor,        # [B, N, 3] raw logits
        pred_locations: torch.Tensor,         # [B, N, 3] metres
        pred_loc_logits: torch.Tensor,        # [B, N, 3] raw logits
        pred_rotation_y: torch.Tensor,        # [B, N] radians
        pred_orient_logits: torch.Tensor,     # [B, N, 1] raw logits

        # Targets
        target_boxes: torch.Tensor,           # [B, M, 4] normalised
        target_labels: torch.Tensor,          # [B, M] class indices
        target_dimensions: torch.Tensor,      # [B, M, 3] metres
        target_locations: torch.Tensor,       # [B, M, 3] metres
        target_rotation_y: torch.Tensor,      # [B, M] radians
        target_counts: torch.Tensor,          # [B] valid targets per batch
    ) -> TotalLossOutput:
        """Compute total multi-task loss.

        Args:
            Model outputs (from Baseline3DDetector.forward):
                class_logits: [B, N, C]
                pred_boxes: [B, N, 4] normalised [0,1]
                pred_objectness: [B, N] in [0,1]
                pred_dimensions: [B, N, 3] metres (exp'd)
                pred_dim_logits: [B, N, 3] raw logits
                pred_locations: [B, N, 3] metres
                pred_loc_logits: [B, N, 3] raw logits
                pred_rotation_y: [B, N] radians
                pred_orient_logits: [B, N, 1] raw logits

            Ground truth targets:
                target_boxes: [B, M, 4] normalised
                target_labels: [B, M] class indices
                target_dimensions: [B, M, 3] metres
                target_locations: [B, M, 3] metres
                target_rotation_y: [B, M] radians
                target_counts: [B] valid targets per batch

        Returns:
            TotalLossOutput with total loss and all components.
        """
        device = class_logits.device
        batch_size, num_predictions, _ = class_logits.shape

        # Assignment: match predictions to targets
        assignment = assign_fixed_order(
            num_predictions=num_predictions,
            target_counts=target_counts,
        )

        # Pad targets to max targets for batching
        max_targets = target_labels.shape[1]
        target_boxes_padded = target_boxes
        target_labels_padded = target_labels
        target_dimensions_padded = target_dimensions
        target_locations_padded = target_locations
        target_rotation_y_padded = target_rotation_y

        # Detection loss (includes box, objectness, classification)
        detection_output = self.detection_loss(
            class_logits=class_logits,
            pred_boxes=pred_boxes,
            pred_objectness=pred_objectness,
            assignment=assignment,
            target_boxes=target_boxes_padded,
            target_labels=target_labels_padded,
            target_counts=target_counts,
        )

        # Dimension loss
        dimension_loss, _ = self.dimension_loss(
            pred_dimensions=pred_dimensions,
            pred_logits=None,  # not used in current impl
            target_dimensions=target_dimensions_padded,
            assignment_matched_mask=assignment.matched_mask,
            assignment_pred_indices=assignment.pred_indices,
            assignment_target_indices=assignment.target_indices,
        )

        # Location loss
        location_loss, _ = self.location_loss(
            pred_locations=pred_locations,
            pred_logits=pred_locations,  # direct regression
            target_locations=target_locations_padded,
            assignment_matched_mask=assignment.matched_mask,
            assignment_pred_indices=assignment.pred_indices,
            assignment_target_indices=assignment.target_indices,
        )

        # Orientation loss
        orientation_loss, _ = self.orientation_loss(
            pred_rotation_y=pred_rotation_y,
            pred_logits=None,
            target_rotation_y=target_rotation_y_padded,
            assignment_pred_indices=assignment.pred_indices,
            assignment_target_indices=assignment.target_indices,
        )

        # Weighted total
        total = (
            detection_output.total
            + self.weights["dimension_3d"] * dimension_loss
            + self.weights["location_3d"] * location_loss
            + self.weights["orientation"] * orientation_loss
        )

        return TotalLossOutput(
            total=total,
            detection=detection_output,
            dimension_3d=dimension_loss,
            location_3d=location_loss,
            orientation=orientation_loss,
            weights=self.weights.copy(),
        )


def create_total_loss_from_config(config: dict) -> TotalLoss:
    """Factory function to create TotalLoss from config dict.

    Expected config structure:
    {
        "model": {"num_classes": 3},
        "loss": {
            "box_weight": 1.0,
            "objectness_weight": 1.0,
            "classification_weight": 1.0,
            "dimension_3d_weight": 1.0,
            "location_3d_weight": 1.0,
            "orientation_weight": 1.0,
            "box_loss_beta": 1.0/9.0,
        }
    }
    """
    model_config = config.get("model", {})
    loss_config = config.get("loss", {})

    return TotalLoss(
        num_classes=model_config.get("num_classes", 3),
        box_weight=loss_config.get("box_weight", 1.0),
        objectness_weight=loss_config.get("objectness_weight", 1.0),
        classification_weight=loss_config.get("classification_weight", 1.0),
        dimension_3d_weight=loss_config.get("dimension_3d_weight", 1.0),
        location_3d_weight=loss_config.get("location_3d_weight", 1.0),
        orientation_weight=loss_config.get("orientation_weight", 1.0),
        box_loss_beta=loss_config.get("box_loss_beta", 1.0 / 9.0),
    )