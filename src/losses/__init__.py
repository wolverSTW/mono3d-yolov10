"""Loss modules for baseline multi-task loss.

This package contains:
- detection_loss.py: 2D bbox, objectness, classification losses
- dimension_loss.py: 3D dimension regression loss
- location_loss.py: 3D location regression loss
- orientation_loss.py: Orientation regression loss (angular)
- assignment.py: Prediction-target assignment utilities
- total_loss.py: Multi-task loss aggregator
"""

from src.losses.assignment import AssignmentResult, assign_fixed_order, assign_center_distance, compute_centers
from src.losses.detection_loss import DetectionLoss, DetectionLossOutput, EmptyDetectionLoss
from src.losses.dimension_loss import DimensionLoss, DimensionLossOutput
from src.losses.location_loss import LocationLoss
from src.losses.orientation_loss import OrientationLoss
from src.losses.total_loss import TotalLoss, TotalLossOutput, create_total_loss_from_config

__all__ = [
    "AssignmentResult",
    "assign_fixed_order",
    "assign_center_distance",
    "compute_centers",
    "DetectionLoss",
    "DetectionLossOutput",
    "EmptyDetectionLoss",
    "DimensionLoss",
    "DimensionLossOutput",
    "LocationLoss",
    "OrientationLoss",
    "TotalLoss",
    "TotalLossOutput",
    "create_total_loss_from_config",
]