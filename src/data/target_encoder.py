"""Target encoding for KITTI 3D object detection baseline.

This module defines how raw KITTI annotations are encoded into model-ready
targets. The encoding is kept simple and explicit for the baseline.

Coordinate conventions (from EXP-002):
- 2D bbox: (x1, y1, x2, y2) in image pixel coordinates
- 3D dimensions: (height, width, length) in metres
- 3D location: (X, Y, Z) in rectified camera coordinates (metres)
- Orientation: rotation_y in radians (camera-frame yaw)

Target encoding choices (baseline):
- 2D bbox: normalised to [0, 1] by output image width/height
- 3D dimensions: absolute metres (no normalisation)
- 3D location: absolute metres (no normalisation)
- Orientation: rotation_y in radians (no encoding)
- Class: integer index from configured class mapping

These choices are baseline defaults. They are not claimed to be optimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.data.kitti_parser import BoundingBox2D, Dimensions3D, CameraLocation3D, KittiObject


@dataclass(frozen=True)
class EncodedTarget:
    """Model-ready target for a single frame.

    Attributes:
        class_ids: Tensor of shape [N] with class indices for N objects.
        bboxes_2d: Tensor of shape [N, 4] with normalised (x1, y1, x2, y2).
        dimensions_3d: Tensor of shape [N, 3] with (height, width, length) in metres.
        locations_3d: Tensor of shape [N, 3] with (X, Y, Z) in metres.
        rotation_y: Tensor of shape [N] with camera-frame yaw in radians.
        image_size: (width, height) of the preprocessed output image.
        transformed_p2: 3×4 projection matrix for the preprocessed image.
    """

    class_ids: torch.Tensor
    bboxes_2d: torch.Tensor
    dimensions_3d: torch.Tensor
    locations_3d: torch.Tensor
    rotation_y: torch.Tensor
    image_size: tuple[int, int]
    transformed_p2: torch.Tensor


def build_class_mapping(
    configured_classes: Sequence[str] | None,
    observed_classes: Sequence[str],
) -> tuple[dict[str, int], list[str]]:
    """Build class name -> index mapping.

    If configured_classes is provided, it is used as the authoritative mapping.
    Observed classes not in configured_classes are ignored (warning logged).

    If configured_classes is None, the sorted observed detection classes
    (excluding DontCare) are used as the mapping.

    Returns:
        (class_to_idx, idx_to_class) where idx_to_class[i] gives the class name.
    """
    if configured_classes is not None:
        class_list = list(configured_classes)
    else:
        # Fallback: use sorted observed non-DontCare classes
        detection_classes = [c for c in observed_classes if c != "DontCare"]
        class_list = sorted(set(detection_classes))

    class_to_idx = {name: idx for idx, name in enumerate(class_list)}
    return class_to_idx, class_list


def encode_annotations(
    annotations: Sequence[KittiObject],
    class_mapping: dict[str, int],
    image_size: tuple[int, int],
    transformed_p2: np.ndarray,
) -> EncodedTarget:
    """Encode a list of KITTI annotations into a model-ready target.

    DontCare annotations are excluded from the encoded targets.

    Args:
        annotations: Parsed KITTI annotations for one frame.
        class_mapping: Dictionary mapping class name to class index.
        image_size: (width, height) of the preprocessed output image.
        transformed_p2: 3×4 projection matrix for the preprocessed image.

    Returns:
        EncodedTarget with tensors ready for model consumption.

    Raises:
        ValueError: If an annotation's class is not in the class mapping.
    """
    image_width, image_height = image_size

    # Filter out DontCare
    valid_annotations = [ann for ann in annotations if not ann.is_dontcare]

    if not valid_annotations:
        # Return empty tensors with correct shapes
        return EncodedTarget(
            class_ids=torch.empty(0, dtype=torch.long),
            bboxes_2d=torch.empty(0, 4, dtype=torch.float32),
            dimensions_3d=torch.empty(0, 3, dtype=torch.float32),
            locations_3d=torch.empty(0, 3, dtype=torch.float32),
            rotation_y=torch.empty(0, dtype=torch.float32),
            image_size=image_size,
            transformed_p2=torch.from_numpy(transformed_p2.astype(np.float32)),
        )

    class_ids = []
    bboxes_2d = []
    dimensions_3d = []
    locations_3d = []
    rotation_y = []

    for ann in valid_annotations:
        if ann.class_name not in class_mapping:
            raise ValueError(
                f"Class '{ann.class_name}' not in class mapping. "
                f"Configured classes: {list(class_mapping.keys())}"
            )
        class_ids.append(class_mapping[ann.class_name])

        # Normalise 2D bbox to [0, 1] by output image size
        bboxes_2d.append([
            ann.bbox_2d.x1 / image_width,
            ann.bbox_2d.y1 / image_height,
            ann.bbox_2d.x2 / image_width,
            ann.bbox_2d.y2 / image_height,
        ])

        # 3D dimensions in metres (H, W, L)
        dimensions_3d.append([
            ann.dimensions_3d.height,
            ann.dimensions_3d.width,
            ann.dimensions_3d.length,
        ])

        # 3D location in metres (X, Y, Z) rectified camera coordinates
        locations_3d.append([
            ann.location_3d.x,
            ann.location_3d.y,
            ann.location_3d.z,
        ])

        # Orientation: rotation_y in radians
        rotation_y.append(ann.rotation_y)

    return EncodedTarget(
        class_ids=torch.tensor(class_ids, dtype=torch.long),
        bboxes_2d=torch.tensor(bboxes_2d, dtype=torch.float32),
        dimensions_3d=torch.tensor(dimensions_3d, dtype=torch.float32),
        locations_3d=torch.tensor(locations_3d, dtype=torch.float32),
        rotation_y=torch.tensor(rotation_y, dtype=torch.float32),
        image_size=image_size,
        transformed_p2=torch.from_numpy(transformed_p2.astype(np.float32)),
    )


def decode_bbox_2d(
    bbox_norm: torch.Tensor | np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Decode normalised bbox back to pixel coordinates.

    Args:
        bbox_norm: Normalised (x1, y1, x2, y2) in [0, 1].
        image_size: (width, height) of the image.

    Returns:
        Pixel coordinates as numpy array.
    """
    bbox_norm = np.asarray(bbox_norm)
    width, height = image_size
    return np.array([
        bbox_norm[..., 0] * width,
        bbox_norm[..., 1] * height,
        bbox_norm[..., 2] * width,
        bbox_norm[..., 3] * height,
    ]).T if bbox_norm.ndim == 2 else np.array([
        bbox_norm[0] * width,
        bbox_norm[1] * height,
        bbox_norm[2] * width,
        bbox_norm[3] * height,
    ])