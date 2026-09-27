"""Prediction-target assignment utilities for baseline losses.

This module provides deterministic, testable assignment strategies for
matching model predictions to ground-truth targets.

The baseline model produces a fixed number of predictions per image
(from the feature pyramid), while ground-truth has a variable number
of objects. This module provides a simple, deterministic assignment
strategy suitable for baseline experiments.

Key design principles:
- Deterministic: same inputs always produce same assignments
- Testable: each strategy can be unit tested independently
- Isolated: assignment logic is separate from individual loss modules
- Baseline-oriented: simple strategies appropriate for initial experiments

The default strategy is "fixed_order": predictions are matched to targets
in order, up to min(num_predictions, num_targets). Excess predictions
become background (no-object). Excess targets are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AssignmentResult:
    """Result of prediction-target assignment.

    Attributes:
        pred_indices: [B, M] indices of selected predictions for each batch.
            M = min(num_predictions, num_targets) for matched pairs.
        target_indices: [B, M] indices of assigned targets for each batch.
        matched_mask: [B, num_predictions] boolean mask indicating which
            predictions are matched to a target.
        target_matched_mask: [B, num_targets] boolean mask indicating which
            targets are matched to a prediction.
    """

    pred_indices: torch.Tensor
    target_indices: torch.Tensor
    matched_mask: torch.Tensor
    target_matched_mask: torch.Tensor


def assign_fixed_order(
    num_predictions: int,
    target_counts: torch.Tensor,
    max_predictions: Optional[int] = None,
) -> AssignmentResult:
    """Assign predictions to targets in fixed order.

    For each batch element, the first min(num_predictions, num_targets)
    predictions are matched to targets in order. Excess predictions become
    background. Excess targets are unmatched.

    This is the simplest deterministic assignment suitable for baseline
    experiments with synthetic data where prediction and target orders
    are aligned.

    Args:
        num_predictions: Number of predictions per image (fixed N).
        target_counts: [B] tensor with number of valid targets per batch.
        max_predictions: Optional cap on number of matched pairs per image.
            If None, uses min(num_predictions, target_counts[b]) per batch.

    Returns:
        AssignmentResult with matched indices and masks.
    """
    device = target_counts.device
    batch_size = target_counts.shape[0]

    # Clamp target counts to num_predictions
    effective_counts = torch.minimum(target_counts, torch.full_like(target_counts, num_predictions))

    if max_predictions is not None:
        effective_counts = torch.minimum(effective_counts, torch.full_like(target_counts, max_predictions))

    # Build matched masks
    matched_mask = torch.zeros(batch_size, num_predictions, dtype=torch.bool, device=device)
    target_matched_mask_list = []

    pred_indices_list = []
    target_indices_list = []

    for b in range(batch_size):
        n_match = effective_counts[b].item()
        if n_match > 0:
            # First n_match predictions matched to first n_match targets
            matched_mask[b, :n_match] = True
            pred_indices_list.append(torch.arange(n_match, device=device, dtype=torch.long))
            target_indices_list.append(torch.arange(n_match, device=device, dtype=torch.long))
            target_matched = torch.zeros(target_counts[b].item(), dtype=torch.bool, device=device)
            target_matched[:n_match] = True
            target_matched_mask_list.append(target_matched)
        else:
            pred_indices_list.append(torch.empty(0, dtype=torch.long, device=device))
            target_indices_list.append(torch.empty(0, dtype=torch.long, device=device))
            target_matched_mask_list.append(torch.zeros(target_counts[b].item(), dtype=torch.bool, device=device))

    # Pad sequences to same length for tensor stacking
    max_matches = max((len(p) for p in pred_indices_list), default=0)
    if max_matches > 0:
        pred_indices = torch.stack([
            F.pad(p, (0, max_matches - len(p)), value=0)
            for p in pred_indices_list
        ])
        target_indices = torch.stack([
            F.pad(t, (0, max_matches - len(t)), value=0)
            for t in target_indices_list
        ])
    else:
        pred_indices = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        target_indices = torch.empty(batch_size, 0, dtype=torch.long, device=device)

    # Pad target_matched_mask to max target count
    max_targets = target_counts.max().item() if batch_size > 0 else 0
    if max_targets > 0:
        target_matched_mask = torch.stack([
            F.pad(t, (0, max_targets - len(t)), value=False)
            for t in target_matched_mask_list
        ])
    else:
        target_matched_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)

    return AssignmentResult(
        pred_indices=pred_indices,
        target_indices=target_indices,
        matched_mask=matched_mask,
        target_matched_mask=target_matched_mask,
    )


def compute_centers(bboxes: torch.Tensor) -> torch.Tensor:
    """Compute center points of bounding boxes.

    Args:
        bboxes: [B, N, 4] in (x1, y1, x2, y2) format, normalised [0, 1].

    Returns:
        centers: [B, N, 2] in (cx, cy) format, normalised [0, 1].
    """
    x1, y1, x2, y2 = bboxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return torch.stack([cx, cy], dim=-1)


def assign_center_distance(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_counts: torch.Tensor,
) -> AssignmentResult:
    """Assign predictions to targets based on center point distance.

    For each batch, computes pairwise center distances between predictions
    and targets, then greedily matches closest pairs (one-to-one).

    This is a more geometrically aware assignment, still deterministic.

    Args:
        pred_bboxes: [B, N, 4] normalised (x1, y1, x2, y2).
        target_bboxes: List of [Ni, 4] tensors or padded [B, max_targets, 4].
        target_counts: [B] number of valid targets per batch.

    Returns:
        AssignmentResult with distance-based matching.
    """
    batch_size, num_predictions, _ = pred_bboxes.shape
    device = pred_bboxes.device

    pred_centers = compute_centers(pred_bboxes)  # [B, N, 2]

    # Handle target bboxes - could be list or padded tensor
    if isinstance(target_bboxes, list):
        max_targets = max((len(t) for t in target_bboxes), default=0)
        target_bboxes_padded = torch.zeros(len(target_bboxes), max_targets, 4, device=pred_bboxes.device)
        for b, t in enumerate(target_bboxes):
            if len(t) > 0:
                target_bboxes_padded[b, :len(t)] = t
        target_bboxes = target_bboxes_padded
    else:
        max_targets = target_bboxes.shape[1]

    target_centers = compute_centers(target_bboxes)  # [B, max_targets, 2]

    matched_mask = torch.zeros(pred_bboxes.shape[:2], dtype=torch.bool, device=device)
    target_matched_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=device)
    pred_indices_list = []
    target_indices_list = []

    for b in range(batch_size):
        n_targets = target_counts[b].item()
        if n_targets == 0:
            pred_indices_list.append(torch.empty(0, dtype=torch.long, device=device))
            target_indices_list.append(torch.empty(0, dtype=torch.long, device=device))
            continue

        # Compute pairwise distances
        pred_c = pred_centers[b]  # [N, 2]
        target_c = target_centers[b, :n_targets]  # [n_targets, 2]

        # [N, n_targets]
        dist = torch.cdist(pred_c, target_c, p=2)

        # Greedy matching: for each target, find closest unmatched prediction
        matched_preds = torch.zeros(num_predictions, dtype=torch.bool, device=device)
        matched_targets = torch.zeros(n_targets, dtype=torch.bool, device=device)

        # Sort all pairs by distance
        flat_dist = dist.flatten()
        sorted_indices = flat_dist.argsort()

        # Map flat indices to (pred_idx, target_idx)
        pred_idx_flat = sorted_indices // n_targets
        target_idx_flat = sorted_indices % n_targets

        matched_pred_list = []
        matched_target_list = []

        for pi, ti in zip(pred_idx_flat.tolist(), target_idx_flat.tolist()):
            if not matched_preds[pi] and not matched_targets[ti]:
                matched_preds[pi] = True
                matched_targets[ti] = True
                matched_pred_list.append(pi)
                matched_target_list.append(ti)

        matched_mask[b] = matched_preds
        target_matched_mask[b, :n_targets] = matched_targets

        pred_indices_list.append(torch.tensor(matched_pred_list, device=device, dtype=torch.long))
        target_indices_list.append(torch.tensor(matched_target_list, device=device, dtype=torch.long))

    # Pad to same length
    max_matches = max((len(p) for p in pred_indices_list), default=0)
    if max_matches > 0:
        pred_indices = torch.stack([
            F.pad(p, (0, max_matches - len(p)), value=-1)
            for p in pred_indices_list
        ])
        target_indices = torch.stack([
            F.pad(t, (0, max_matches - len(t)), value=-1)
            for t in target_indices_list
        ])
    else:
        pred_indices = torch.empty(len(pred_bboxes), 0, dtype=torch.long, device=device)
        target_indices = torch.empty(len(pred_bboxes), 0, dtype=torch.long, device=device)

    return AssignmentResult(
        pred_indices=pred_indices,
        target_indices=target_indices,
        matched_mask=matched_mask,
        target_matched_mask=target_matched_mask,
    )