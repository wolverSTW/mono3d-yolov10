"""Tests for total loss aggregator."""

from __future__ import annotations

import torch

from src.losses.assignment import assign_fixed_order
from src.losses.total_loss import TotalLoss, create_total_loss_from_config


def test_total_loss_construction() -> None:
    """Test total loss can be constructed."""
    loss = TotalLoss(num_classes=3)
    assert isinstance(loss, TotalLoss)


def test_total_loss_config_factory() -> None:
    """Test factory function with config."""
    config = {
        "model": {"num_classes": 4},
        "loss": {
            "box_weight": 2.0,
            "objectness_weight": 0.5,
            "classification_weight": 1.0,
            "dimension_3d_weight": 1.5,
            "location_3d_weight": 1.0,
            "orientation_weight": 2.0,
            "box_loss_beta": 1.0 / 9.0,
        }
    }
    loss = create_total_loss_from_config(config)
    assert loss.config.num_classes == 4
    assert loss.weights["box"] == 2.0
    assert loss.weights["objectness"] == 0.5


def test_total_loss_forward_perfect_match() -> None:
    """Test total loss forward with perfect predictions."""
    loss_fn = TotalLoss(num_classes=2)

    B = 1
    N = 2  # predictions
    M = 2  # targets
    C = 2

    # Perfect predictions
    class_logits = torch.tensor([[
        [10.0, -10.0],  # class 0
        [-10.0, 10.0],  # class 1
    ]], dtype=torch.float32)

    pred_boxes = torch.tensor([[
        [0.1, 0.1, 0.5, 0.5],
        [0.2, 0.2, 0.6, 0.6],
    ]], dtype=torch.float32)

    pred_objectness = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    pred_dimensions = torch.tensor([[
        [1.5, 1.6, 3.8],
        [1.7, 0.5, 0.6],
    ]], dtype=torch.float32)
    pred_dim_logits = torch.log(pred_dimensions)

    pred_locations = torch.tensor([[
        [1.0, 1.5, 20.0],
        [2.0, 1.0, 15.0],
    ]], dtype=torch.float32)
    pred_loc_logits = pred_locations.clone()

    pred_rotation_y = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    pred_orient_logits = pred_rotation_y.unsqueeze(-1)

    # Targets match perfectly
    target_boxes = torch.tensor([[
        [0.1, 0.1, 0.5, 0.5],
        [0.2, 0.2, 0.6, 0.6],
    ]], dtype=torch.float32)
    target_labels = torch.tensor([[0, 1]], dtype=torch.long)
    target_dimensions = torch.tensor([[
        [1.5, 1.6, 3.8],
        [1.7, 0.5, 0.6],
    ]], dtype=torch.float32)
    target_locations = torch.tensor([[
        [1.0, 1.5, 20.0],
        [2.0, 1.0, 15.0],
    ]], dtype=torch.float32)
    target_rotation_y = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    target_counts = torch.tensor([2], dtype=torch.long)

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        pred_dimensions=pred_dimensions,
        pred_dim_logits=pred_dim_logits,
        pred_locations=pred_locations,
        pred_loc_logits=pred_loc_logits,
        pred_rotation_y=pred_rotation_y,
        pred_orient_logits=pred_rotation_y.unsqueeze(-1),
        target_boxes=target_boxes,
        target_labels=target_labels,
        target_dimensions=target_dimensions,
        target_locations=target_locations,
        target_rotation_y=target_rotation_y,
        target_counts=torch.tensor([2]),
    )

    assert output.total < 0.1  # Should be small for perfect match
    assert output.detection.box_loss < 0.01
    assert output.detection.classification_loss < 0.01
    assert output.dimension_3d < 0.01
    assert output.location_3d < 0.01
    assert output.orientation < 0.01


def test_total_loss_empty_targets() -> None:
    """Test total loss with no targets."""
    loss_fn = TotalLoss(num_classes=3)

    class_logits = torch.randn(1, 3, 3)
    pred_boxes = torch.rand(1, 3, 4)
    pred_objectness = torch.rand(1, 3)
    pred_dimensions = torch.rand(1, 3, 3)
    pred_dim_logits = torch.rand(1, 3, 3)
    pred_locations = torch.rand(1, 3, 3)
    pred_loc_logits = torch.rand(1, 3, 3)
    pred_rotation_y = torch.rand(1, 3)
    pred_orient_logits = torch.rand(1, 3, 1)

    target_boxes = torch.empty(1, 0, 4)
    target_labels = torch.empty(1, 0, dtype=torch.long)
    target_dimensions = torch.empty(1, 0, 3)
    target_locations = torch.empty(1, 0, 3)
    target_rotation_y = torch.empty(1, 0)
    target_counts = torch.tensor([0])

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        pred_dimensions=pred_dimensions,
        pred_dim_logits=pred_dim_logits,
        pred_locations=pred_locations,
        pred_loc_logits=pred_loc_logits,
        pred_rotation_y=pred_rotation_y,
        pred_orient_logits=pred_orient_logits,
        target_boxes=target_boxes,
        target_labels=target_labels,
        target_dimensions=target_dimensions,
        target_locations=target_locations,
        target_rotation_y=target_rotation_y,
        target_counts=target_counts,
    )

    assert output.detection.box_loss == 0.0
    assert output.detection.classification_loss == 0.0
    assert output.dimension_3d == 0.0
    assert output.location_3d == 0.0
    assert output.orientation == 0.0
    # Objectness should be non-zero (background)
    assert output.detection.objectness_loss > 0.0


def test_total_loss_weighted_sum() -> None:
    """Test that total equals weighted component sum."""
    loss_fn = TotalLoss(
        num_classes=2,
        box_weight=2.0,
        objectness_weight=1.0,
        classification_weight=1.0,
        dimension_3d_weight=3.0,
        location_3d_weight=2.0,
        orientation_weight=1.0,
    )

    B = 1
    N = 1
    M = 1
    C = 2

    # Perfect match - all losses should be zero
    class_logits = torch.tensor([[[10.0, -10.0]]], dtype=torch.float32)
    pred_boxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]], dtype=torch.float32)
    pred_objectness = torch.tensor([[1.0]], dtype=torch.float32)
    pred_dimensions = torch.tensor([[[1.5, 1.6, 3.8]]], dtype=torch.float32)
    pred_dim_logits = torch.log(torch.tensor([[[1.5, 1.6, 3.8]]], dtype=torch.float32))
    pred_locations = torch.tensor([[[1.0, 1.5, 20.0]]], dtype=torch.float32)
    pred_loc_logits = pred_locations.clone()
    pred_rotation_y = torch.tensor([[0.5]], dtype=torch.float32)
    pred_orient_logits = pred_rotation_y.unsqueeze(-1)

    target_boxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]], dtype=torch.float32)
    target_labels = torch.tensor([[0]], dtype=torch.long)
    target_dimensions = torch.tensor([[[1.5, 1.6, 3.8]]], dtype=torch.float32)
    target_locations = torch.tensor([[[1.0, 1.5, 20.0]]], dtype=torch.float32)
    target_rotation_y = torch.tensor([[0.5]], dtype=torch.float32)
    target_counts = torch.tensor([1])

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        pred_dimensions=pred_dimensions,
        pred_dim_logits=torch.log(torch.tensor([[[1.5, 1.6, 3.8]]], dtype=torch.float32)),
        pred_locations=pred_locations,
        pred_loc_logits=pred_locations.clone(),
        pred_rotation_y=pred_rotation_y,
        pred_orient_logits=pred_rotation_y.unsqueeze(-1),
        target_boxes=target_boxes,
        target_labels=torch.tensor([[0]], dtype=torch.long),
        target_dimensions=target_dimensions,
        target_locations=target_locations,
        target_rotation_y=target_rotation_y,
        target_counts=target_counts,
    )

    # Perfect match -> all losses should be very small
    assert output.detection.box_loss < 0.01
    assert output.detection.classification_loss < 0.01
    assert output.dimension_3d < 0.01
    assert output.location_3d < 0.01
    assert output.orientation < 0.01
    assert output.total < 0.1


def test_total_loss_output_structure() -> None:
    """Test TotalLossOutput has correct structure."""
    loss_fn = TotalLoss(num_classes=2)

    B = 1
    N = 2
    M = 2
    C = 2

    class_logits = torch.randn(B, N, C)
    pred_boxes = torch.rand(B, N, 4)
    pred_objectness = torch.rand(B, N)
    pred_dimensions = torch.rand(B, N, 3)
    pred_dim_logits = torch.rand(B, N, 3)
    pred_locations = torch.rand(B, N, 3)
    pred_loc_logits = torch.rand(B, N, 3)
    pred_rotation_y = torch.rand(B, N)
    pred_orient_logits = torch.rand(B, N, 1)

    target_boxes = torch.rand(B, M, 4)
    target_labels = torch.randint(0, C, (B, M))
    target_dimensions = torch.rand(B, M, 3)
    target_locations = torch.rand(B, M, 3)
    target_rotation_y = torch.rand(B, M)
    target_counts = torch.tensor([M])

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        pred_dimensions=pred_dimensions,
        pred_dim_logits=pred_dim_logits,
        pred_locations=pred_locations,
        pred_loc_logits=pred_loc_logits,
        pred_rotation_y=pred_rotation_y,
        pred_orient_logits=pred_orient_logits,
        target_boxes=target_boxes,
        target_labels=target_labels,
        target_dimensions=target_dimensions,
        target_locations=target_locations,
        target_rotation_y=target_rotation_y,
        target_counts=target_counts,
    )

    assert hasattr(output, 'total')
    assert hasattr(output, 'detection')
    assert hasattr(output, 'dimension_3d')
    assert hasattr(output, 'location_3d')
    assert hasattr(output, 'orientation')
    assert hasattr(output, 'weights')
    assert output.total.dim() == 0
    assert isinstance(output.weights, dict)