"""Tests for 3D dimension loss."""

from __future__ import annotations

import torch

from src.losses.assignment import AssignmentResult
from src.losses.dimension_loss import DimensionLoss


def test_dimension_loss_construction() -> None:
    """Test dimension loss can be constructed."""
    loss = DimensionLoss()
    assert isinstance(loss, DimensionLoss)


def test_dimension_loss_custom_weight() -> None:
    """Test dimension loss with custom weight."""
    loss = DimensionLoss(weight=2.5)
    assert loss.weight == 2.5


def test_dimension_loss_perfect_match() -> None:
    """Test dimension loss with perfect predictions."""
    loss_fn = DimensionLoss(weight=1.0)

    pred_dimensions = torch.tensor([[[1.5, 1.6, 3.8]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.405, 0.470, 1.335]]], dtype=torch.float32)  # log of dims
    target_dimensions = torch.tensor([[[1.5, 1.6, 3.8]]], dtype=torch.float32)

    # Create assignment with 1 matched pair
    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    dim_loss, total = loss_fn(
        pred_dimensions=pred_dimensions,
        pred_logits=pred_logits,
        target_dimensions=target_dimensions,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    assert dim_loss < 0.01
    assert total < 0.01


def test_dimension_loss_nonzero_error() -> None:
    """Test dimension loss with non-zero error."""
    loss_fn = DimensionLoss(weight=1.0)

    pred_dimensions = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    target_dimensions = torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    dim_loss, total = loss_fn(
        pred_dimensions=pred_dimensions,
        pred_logits=pred_logits,
        target_dimensions=target_dimensions,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    # Expected Smooth L1 loss: each dim diff = 1.0
    # For beta=1/9, diff=1.0 > beta, so loss = diff - 0.5*beta = 1 - 0.5/9 = 0.944...
    expected_loss = (1.0 - 0.5 / 9.0)  # per dimension
    assert abs(dim_loss.item() - expected_loss) < 0.01
    assert abs(total.item() - expected_loss) < 0.01


def test_dimension_loss_empty_assignment() -> None:
    """Test dimension loss with empty assignment."""
    loss_fn = DimensionLoss()

    pred_dimensions = torch.randn(1, 3, 3)
    pred_logits = torch.randn(1, 3, 3)
    target_dimensions = torch.randn(1, 2, 3)

    # Empty assignment
    assignment = AssignmentResult(
        pred_indices=torch.empty(1, 0, dtype=torch.long),
        target_indices=torch.empty(1, 0, dtype=torch.long),
        matched_mask=torch.zeros(1, 3, dtype=torch.bool),
        target_matched_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    dim_loss, total = loss_fn(
        pred_dimensions=pred_dimensions,
        pred_logits=pred_logits,
        target_dimensions=target_dimensions,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    assert dim_loss == 0.0
    assert total == 0.0


def test_dimension_loss_weight() -> None:
    """Test dimension loss weight scaling."""
    loss_fn = DimensionLoss(weight=2.0)

    pred_dimensions = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    target_dimensions = torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float32)

    assignment = AssignmentResult(
        pred_indices=torch.tensor([[0]], dtype=torch.long),
        target_indices=torch.tensor([[0]], dtype=torch.long),
        matched_mask=torch.tensor([[True]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    dim_loss, total = loss_fn(
        pred_dimensions=pred_dimensions,
        pred_logits=pred_logits,
        target_dimensions=target_dimensions,
        assignment_matched_mask=assignment.matched_mask,
        assignment_pred_indices=assignment.pred_indices,
        assignment_target_indices=assignment.target_indices,
    )

    expected = (1.0 - 0.5 / 9.0) * 2.0
    assert abs(total.item() - expected) < 0.01