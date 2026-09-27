"""Tests for orientation loss."""

from __future__ import annotations

import math
import torch

from src.losses.assignment import AssignmentResult
from src.losses.orientation_loss import OrientationLoss


def test_orientation_loss_construction() -> None:
    """Test orientation loss can be constructed."""
    loss = OrientationLoss()
    assert isinstance(loss, OrientationLoss)


def test_orientation_loss_perfect_match() -> None:
    """Test orientation loss with perfect predictions."""
    loss_fn = OrientationLoss(weight=1.0)

    pred_rotation_y = torch.tensor([[0.5]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.5]]], dtype=torch.float32)
    target_rotation_y = torch.tensor([[0.5]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    orient_loss, total = loss_fn(
        pred_rotation_y=pred_rotation_y,
        pred_logits=pred_logits,
        target_rotation_y=target_rotation_y,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    assert orient_loss < 0.01
    assert total < 0.01


def test_orientation_loss_pi_wrap() -> None:
    """Test orientation loss handles pi wrap-around correctly."""
    loss_fn = OrientationLoss(weight=1.0)

    # Prediction near -pi, target near +pi (should be small difference)
    pred_rotation_y = torch.tensor([[-3.1]], dtype=torch.float32)
    pred_logits = torch.tensor([[[ -3.1 ]]], dtype=torch.float32)
    target_rotation_y = torch.tensor([[3.1]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    orient_loss, total = loss_fn(
        pred_rotation_y=pred_rotation_y,
        pred_logits=pred_logits,
        target_rotation_y=target_rotation_y,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    # Angular difference should be ~0.083 (2*pi - 6.2), not 6.2
    angular_diff = abs(math.atan2(math.sin(-3.1 - 3.1), math.cos(-3.1 - 3.1)))
    assert orient_loss < angular_diff + 0.01


def test_orientation_loss_opposite() -> None:
    """Test orientation loss with opposite directions."""
    loss_fn = OrientationLoss(weight=1.0)

    pred_rotation_y = torch.tensor([[0.0]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.0]]], dtype=torch.float32)
    target_rotation_y = torch.tensor([[math.pi]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    orient_loss, total = loss_fn(
        pred_rotation_y=pred_rotation_y,
        pred_logits=pred_logits,
        target_rotation_y=target_rotation_y,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    # Angular difference = pi
    # Smooth L1: diff = pi > beta, loss = pi - 0.5/9
    expected = math.pi - 0.5 / 9.0
    assert abs(orient_loss.item() - expected) < 0.01
    assert abs(total.item() - expected) < 0.01


def test_orientation_loss_empty_assignment() -> None:
    """Test orientation loss with empty assignment."""
    loss_fn = OrientationLoss()

    pred_rotation_y = torch.randn(1, 3)
    pred_logits = torch.randn(1, 3, 1)
    target_rotation_y = torch.randn(1, 2)

    assignment = AssignmentResult(
        pred_indices=torch.empty(1, 0, dtype=torch.long),
        target_indices=torch.empty(1, 0, dtype=torch.long),
        matched_mask=torch.zeros(1, 3, dtype=torch.bool),
        target_matched_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    orient_loss, total = loss_fn(
        pred_rotation_y=pred_rotation_y,
        pred_logits=pred_logits,
        target_rotation_y=target_rotation_y,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    assert orient_loss == 0.0
    assert total == 0.0


def test_orientation_loss_weight() -> None:
    """Test orientation loss weight scaling."""
    loss_fn = OrientationLoss(weight=4.0)

    pred_rotation_y = torch.tensor([[0.0]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.0]]], dtype=torch.float32)
    target_rotation_y = torch.tensor([[math.pi]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    orient_loss, total = loss_fn(
        pred_rotation_y=pred_rotation_y,
        pred_logits=pred_logits,
        target_rotation_y=target_rotation_y,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    # Angular difference = pi > beta (1/9)
    # Smooth L1: diff = pi > beta, loss = pi - 0.5 * beta
    # beta = 1/9, so loss = pi - 0.5/9
    expected_orient_loss = math.pi - 0.5 / 9.0
    expected_total = expected_orient_loss * 4.0
    assert abs(total.item() - expected_total) < 0.01


def test_orientation_encode_decode_sincos() -> None:
    """Test sin/cos encoding/decoding helpers."""
    from src.models.heads.orientation import OrientationHead
    # Avoid pi boundary which wraps to -pi
    angles = torch.tensor([0.0, math.pi/4, math.pi/2, -math.pi/4, 3.0])
    sin_cos = OrientationHead.encode_sincos(angles)
    decoded = OrientationHead.decode_sincos(sin_cos)
    assert torch.allclose(angles, decoded, atol=1e-6)