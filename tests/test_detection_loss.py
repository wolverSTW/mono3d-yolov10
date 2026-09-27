"""Tests for detection loss."""

from __future__ import annotations

import torch

from src.losses.assignment import assign_fixed_order
from src.losses.detection_loss import DetectionLoss, DetectionLossOutput


def test_detection_loss_construction() -> None:
    """Test detection loss can be constructed with defaults."""
    loss = DetectionLoss()
    assert isinstance(loss, DetectionLoss)


def test_detection_loss_custom_weights() -> None:
    """Test detection loss with custom weights."""
    loss = DetectionLoss(
        box_weight=2.0,
        objectness_weight=0.5,
        classification_weight=1.5,
    )
    assert loss.box_weight == 2.0
    assert loss.objectness_weight == 0.5
    assert loss.classification_weight == 1.5


def test_detection_loss_perfect_match() -> None:
    """Test detection loss with perfect predictions."""
    loss_fn = DetectionLoss(
        box_weight=1.0,
        objectness_weight=1.0,
        classification_weight=1.0,
    )

    # Single batch, 2 predictions, 1 class
    class_logits = torch.tensor([[
        [10.0, -10.0],  # class 0
        [-10.0, 10.0],  # class 1
    ]], dtype=torch.float32)

    pred_boxes = torch.tensor([[
        [0.1, 0.1, 0.5, 0.5],
        [0.2, 0.2, 0.6, 0.6],
    ]], dtype=torch.float32)

    pred_objectness = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    target_boxes = torch.tensor([[
        [0.1, 0.1, 0.5, 0.5],
        [0.2, 0.2, 0.6, 0.6],
    ]], dtype=torch.float32)

    target_labels = torch.tensor([[0, 1]], dtype=torch.long)
    target_counts = torch.tensor([2], dtype=torch.long)

    assignment = assign_fixed_order(num_predictions=2, target_counts=target_counts)

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        assignment=assignment,
        target_boxes=target_boxes,
        target_labels=target_labels,
        target_counts=target_counts,
    )

    # All losses should be small (near zero) for perfect predictions
    assert output.box_loss < 0.01
    assert output.objectness_loss < 0.01
    assert output.classification_loss < 0.01
    assert output.total < 0.03


def test_detection_loss_zero_targets() -> None:
    """Test detection loss with no targets."""
    loss_fn = DetectionLoss()

    class_logits = torch.tensor([[
        [1.0, -1.0],
        [-1.0, 1.0],
    ]], dtype=torch.float32)

    pred_boxes = torch.tensor([[
        [0.1, 0.1, 0.5, 0.5],
        [0.2, 0.2, 0.6, 0.6],
    ]], dtype=torch.float32)

    pred_objectness = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    target_boxes = torch.empty(1, 0, 4, dtype=torch.float32)
    target_labels = torch.empty(1, 0, dtype=torch.long)
    target_counts = torch.tensor([0], dtype=torch.long)

    assignment = assign_fixed_order(num_predictions=2, target_counts=torch.tensor([0]))

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        assignment=assignment,
        target_boxes=target_boxes,
        target_labels=torch.empty(1, 0, dtype=torch.long),
        target_counts=target_counts,
    )

    # Box and classification losses should be zero
    assert output.box_loss == 0.0
    assert output.classification_loss == 0.0
    # Objectness loss should be non-zero (background predictions)
    assert output.objectness_loss > 0.0


def test_detection_loss_variable_objects() -> None:
    """Test detection loss with variable number of objects."""
    loss_fn = DetectionLoss()

    batch_size = 2
    num_preds = 3

    class_logits = torch.randn(batch_size, num_preds, 2)
    pred_boxes = torch.rand(batch_size, num_preds, 4)
    pred_objectness = torch.rand(batch_size, num_preds)

    # Image 0: 1 object, Image 1: 2 objects
    target_boxes = torch.tensor([
        [[0.1, 0.1, 0.5, 0.5], [0, 0, 0, 0]],
        [[0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]],
    ])
    target_labels = torch.tensor([[0, -1], [1, 0]])  # -1 = padded
    target_counts = torch.tensor([1, 2])

    assignment = assign_fixed_order(num_predictions=num_preds, target_counts=target_counts)

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        assignment=assignment,
        target_boxes=target_boxes,
        target_labels=target_labels,
        target_counts=target_counts,
    )

    assert isinstance(output.total, torch.Tensor)
    assert output.total.dim() == 0


def test_detection_loss_output_structure() -> None:
    """Test DetectionLossOutput has correct structure."""
    loss_fn = DetectionLoss()

    class_logits = torch.tensor([[
        [10.0, -10.0],
        [-10.0, 10.0],
    ]], dtype=torch.float32)
    pred_boxes = torch.tensor([[
        [0.1, 0.1, 0.5, 0.5],
        [0.2, 0.2, 0.6, 0.6],
    ]], dtype=torch.float32)
    pred_objectness = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    target_boxes = torch.tensor([[
        [0.1, 0.1, 0.5, 0.5],
        [0.2, 0.2, 0.6, 0.6],
    ]], dtype=torch.float32)
    target_labels = torch.tensor([[0, 1]], dtype=torch.long)
    target_counts = torch.tensor([2], dtype=torch.long)

    assignment = assign_fixed_order(num_predictions=2, target_counts=target_counts)

    output = loss_fn(
        class_logits=class_logits,
        pred_boxes=pred_boxes,
        pred_objectness=pred_objectness,
        assignment=assignment,
        target_boxes=target_boxes,
        target_labels=target_labels,
        target_counts=target_counts,
    )

    assert hasattr(output, 'box_loss')
    assert hasattr(output, 'objectness_loss')
    assert hasattr(output, 'classification_loss')
    assert hasattr(output, 'total')
    assert output.box_loss.dim() == 0
    assert output.objectness_loss.dim() == 0
    assert output.classification_loss.dim() == 0
    assert output.total.dim() == 0