"""Tests for 3D location loss."""

from __future__ import annotations

import torch

from src.losses.assignment import AssignmentResult
from src.losses.location_loss import LocationLoss


def test_location_loss_construction() -> None:
    """Test location loss can be constructed."""
    loss = LocationLoss()
    assert isinstance(loss, LocationLoss)


def test_location_loss_perfect_match() -> None:
    """Test location loss with perfect predictions."""
    loss_fn = LocationLoss(weight=1.0)

    pred_locations = torch.tensor([[[1.0, 1.5, 20.0]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[1.0, 1.5, 20.0]]], dtype=torch.float32)
    target_locations = torch.tensor([[[1.0, 1.5, 20.0]]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    loc_loss, total = loss_fn(
        pred_locations=pred_locations,
        pred_logits=pred_logits,
        target_locations=target_locations,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    assert loc_loss < 0.01
    assert total < 0.01


def test_location_loss_nonzero_error() -> None:
    """Test location loss with non-zero error."""
    loss_fn = LocationLoss(weight=1.0)

    pred_locations = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    target_locations = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    loc_loss, total = loss_fn(
        pred_locations=pred_locations,
        pred_logits=pred_logits,
        target_locations=target_locations,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    expected_loss = (1.0 - 0.5 / 9.0)  # per coordinate
    assert abs(loc_loss.item() - expected_loss) < 0.01
    assert abs(total.item() - expected_loss) < 0.01


def test_location_loss_empty_assignment() -> None:
    """Test location loss with empty assignment."""
    loss_fn = LocationLoss()

    pred_locations = torch.randn(1, 3, 3)
    pred_logits = torch.randn(1, 3, 3)
    target_locations = torch.randn(1, 2, 3)

    assignment = AssignmentResult(
        pred_indices=torch.empty(1, 0, dtype=torch.long),
        target_indices=torch.empty(1, 0, dtype=torch.long),
        matched_mask=torch.zeros(1, 3, dtype=torch.bool),
        target_matched_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    loc_loss, total = loss_fn(
        pred_locations=pred_locations,
        pred_logits=pred_logits,
        target_locations=target_locations,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    assert loc_loss == 0.0
    assert total == 0.0


def test_location_loss_weight() -> None:
    """Test location loss weight scaling."""
    loss_fn = LocationLoss(weight=3.0)

    pred_locations = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    target_locations = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    loc_loss, total = loss_fn(
        pred_locations=pred_locations,
        pred_logits=pred_logits,
        target_locations=target_locations,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    expected = (1.0 - 0.5 / 9.0) * 3.0
    assert abs(total.item() - expected) < 0.01


def test_location_loss_xyz_ordering() -> None:
    """Test that location loss respects X, Y, Z ordering."""
    loss_fn = LocationLoss(weight=1.0)

    # X error only
    pred_locations = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    target_locations = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    loc_loss, _ = loss_fn(
        pred_locations=pred_locations,
        pred_logits=pred_logits,
        target_locations=target_locations,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    # X error = 1.0, Y/Z error = 0
    # Smooth L1: X loss = (1.0 - 0.5/9), Y/Z loss = 0
    # Mean over 3 coordinates
    expected = (1.0 - 0.5 / 9.0) / 3.0
    assert abs(loc_loss.item() - expected) < 0.01