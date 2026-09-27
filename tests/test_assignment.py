"""Tests for assignment utilities."""

from __future__ import annotations

import torch

from src.losses.assignment import (
    AssignmentResult,
    assign_fixed_order,
    assign_center_distance,
    compute_centers,
)


def test_assign_fixed_order_basic() -> None:
    """Test fixed-order assignment with matching counts."""
    target_counts = torch.tensor([2, 3, 1])
    num_predictions = 5

    result = assign_fixed_order(num_predictions, target_counts)

    assert result.matched_mask.shape == (3, 5)
    assert result.matched_mask[0].sum() == 2
    assert result.matched_mask[1].sum() == 3
    assert result.matched_mask[2].sum() == 1

    # Check indices (padded with 0 to max_matches=3)
    assert result.pred_indices[0].tolist() == [0, 1, 0]
    assert result.target_indices[0].tolist() == [0, 1, 0]
    assert result.pred_indices[1].tolist() == [0, 1, 2]
    assert result.target_indices[1].tolist() == [0, 1, 2]
    assert result.pred_indices[2].tolist() == [0, 0, 0]
    assert result.target_indices[2].tolist() == [0, 0, 0]


def test_assign_fixed_order_excess_predictions() -> None:
    """Test assignment when predictions exceed targets."""
    target_counts = torch.tensor([1])
    num_predictions = 5

    result = assign_fixed_order(num_predictions, target_counts)

    assert result.matched_mask[0].sum() == 1
    assert result.matched_mask[0, 0] == True
    assert result.matched_mask[0, 1:].all() == False


def test_assign_fixed_order_zero_targets() -> None:
    """Test assignment with zero targets."""
    target_counts = torch.tensor([0, 0])
    num_predictions = 3

    result = assign_fixed_order(num_predictions, target_counts)

    assert result.matched_mask.sum() == 0
    assert result.pred_indices.shape == (2, 0)
    assert result.target_indices.shape == (2, 0)


def test_assign_fixed_order_max_predictions() -> None:
    """Test assignment with max_predictions cap."""
    target_counts = torch.tensor([5])
    num_predictions = 10
    max_predictions = 3

    result = assign_fixed_order(num_predictions, target_counts, max_predictions=max_predictions)

    assert result.matched_mask[0].sum() == 3


def test_compute_centers() -> None:
    """Test center point computation."""
    bboxes = torch.tensor([[
        [0.0, 0.0, 1.0, 1.0],  # center (0.5, 0.5)
        [0.2, 0.4, 0.8, 0.6],  # center (0.5, 0.5)
    ]])

    centers = compute_centers(bboxes)

    expected = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    assert torch.allclose(centers, expected)


def test_assign_center_distance_basic() -> None:
    """Test center-distance assignment with simple case."""
    pred_bboxes = torch.tensor([[
        [0.0, 0.0, 0.5, 0.5],  # center (0.25, 0.25)
        [0.5, 0.5, 1.0, 1.0],  # center (0.75, 0.75)
    ]])

    target_bboxes = [
        torch.tensor([[0.1, 0.1, 0.6, 0.6]], dtype=torch.float32),  # center (0.35, 0.35)
    ]
    target_counts = torch.tensor([1])

    result = assign_center_distance(pred_bboxes, target_bboxes, target_counts)

    # Closest prediction should be first one (center 0.25 vs 0.35)
    assert result.matched_mask[0, 0] == True
    assert result.matched_mask[0, 1] == False


def test_assignment_result_structure() -> None:
    """Test AssignmentResult has correct structure."""
    result = AssignmentResult(
        pred_indices=torch.tensor([[0, 1], [0, -1]], dtype=torch.long),
        target_indices=torch.tensor([[0, 1], [0, -1]], dtype=torch.long),
        matched_mask=torch.tensor([[True, True], [True, False]], dtype=torch.bool),
        target_matched_mask=torch.tensor([[True, True], [True, False]], dtype=torch.bool),
    )

    assert hasattr(result, 'pred_indices')
    assert hasattr(result, 'target_indices')
    assert hasattr(result, 'matched_mask')
    assert hasattr(result, 'target_matched_mask')