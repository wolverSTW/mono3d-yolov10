"""Tests for target encoder."""

from __future__ import annotations

import numpy as np
import torch

from src.data.kitti_parser import (
    BoundingBox2D,
    CameraLocation3D,
    Dimensions3D,
    KittiObject,
)
from src.data.target_encoder import (
    build_class_mapping,
    decode_bbox_2d,
    encode_annotations,
)


def make_annotation(
    class_name: str = "Car",
    *,
    bbox: BoundingBox2D | None = None,
    dimensions: Dimensions3D | None = None,
    location: CameraLocation3D | None = None,
    rotation_y: float = 0.0,
) -> KittiObject:
    """Create a valid KittiObject for testing."""
    return KittiObject(
        class_name=class_name,
        truncated=0.0,
        occluded=0,
        alpha=0.0,
        bbox_2d=bbox or BoundingBox2D(10.0, 10.0, 50.0, 40.0),
        dimensions_3d=dimensions or Dimensions3D(1.5, 1.6, 3.8),
        location_3d=location or CameraLocation3D(1.0, 1.5, 20.0),
        rotation_y=rotation_y,
    )


def test_build_class_mapping_with_configured() -> None:
    """Test class mapping with explicit configuration."""
    configured = ["Car", "Pedestrian", "Cyclist"]
    observed = ["Car", "Van", "Pedestrian"]

    class_to_idx, idx_to_class = build_class_mapping(configured, observed)

    assert class_to_idx == {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    assert idx_to_class == ["Car", "Pedestrian", "Cyclist"]


def test_build_class_mapping_without_configured() -> None:
    """Test class mapping falls back to observed classes."""
    observed = ["Van", "Car", "Pedestrian", "DontCare", "Cyclist"]

    class_to_idx, idx_to_class = build_class_mapping(None, observed)

    # DontCare should be excluded, rest sorted
    assert class_to_idx == {"Car": 0, "Cyclist": 1, "Pedestrian": 2, "Van": 3}
    assert idx_to_class == ["Car", "Cyclist", "Pedestrian", "Van"]


def test_encode_annotations_basic() -> None:
    """Test encoding a single annotation."""
    ann = make_annotation("Car")
    class_mapping = {"Car": 0}
    image_size = (640, 640)
    transformed_p2 = np.eye(3, 4, dtype=np.float64)

    target = encode_annotations([ann], class_mapping, image_size, transformed_p2)

    assert target.class_ids.shape == (1,)
    assert target.class_ids[0].item() == 0
    assert target.bboxes_2d.shape == (1, 4)
    assert target.dimensions_3d.shape == (1, 3)
    assert target.locations_3d.shape == (1, 3)
    assert target.rotation_y.shape == (1,)
    assert target.image_size == (640, 640)
    assert target.transformed_p2.shape == (3, 4)

    # Check bbox normalisation
    bbox = target.bboxes_2d[0]
    assert torch.allclose(bbox, torch.tensor([
        10.0/640, 10.0/640, 50.0/640, 40.0/640
    ]))

    # Check dimensions (H, W, L)
    dims = target.dimensions_3d[0]
    assert torch.allclose(dims, torch.tensor([1.5, 1.6, 3.8]))

    # Check location (X, Y, Z)
    loc = target.locations_3d[0]
    assert torch.allclose(loc, torch.tensor([1.0, 1.5, 20.0]))

    # Check rotation
    assert target.rotation_y[0].item() == 0.0


def test_encode_annotations_multiple_objects() -> None:
    """Test encoding multiple annotations."""
    ann1 = make_annotation("Car", bbox=BoundingBox2D(10, 10, 50, 40))
    ann2 = make_annotation("Pedestrian", bbox=BoundingBox2D(100, 20, 130, 80))
    class_mapping = {"Car": 0, "Pedestrian": 1}
    image_size = (640, 640)
    transformed_p2 = np.eye(3, 4, dtype=np.float64)

    target = encode_annotations([ann1, ann2], class_mapping, image_size, transformed_p2)

    assert target.class_ids.shape == (2,)
    assert target.class_ids.tolist() == [0, 1]
    assert target.bboxes_2d.shape == (2, 4)


def test_encode_annotations_excludes_dontcare() -> None:
    """Test that DontCare annotations are excluded."""
    ann1 = make_annotation("Car")
    ann2 = make_annotation("DontCare")
    class_mapping = {"Car": 0}
    image_size = (640, 640)
    transformed_p2 = np.eye(3, 4, dtype=np.float64)

    target = encode_annotations([ann1, ann2], class_mapping, image_size, transformed_p2)

    assert target.class_ids.shape == (1,)
    assert target.class_ids[0].item() == 0


def test_encode_annotations_empty() -> None:
    """Test encoding empty annotation list."""
    class_mapping = {"Car": 0}
    image_size = (640, 640)
    transformed_p2 = np.eye(3, 4, dtype=np.float64)

    target = encode_annotations([], class_mapping, image_size, transformed_p2)

    assert target.class_ids.shape == (0,)
    assert target.bboxes_2d.shape == (0, 4)
    assert target.dimensions_3d.shape == (0, 3)
    assert target.locations_3d.shape == (0, 3)
    assert target.rotation_y.shape == (0,)


def test_encode_annotations_unknown_class_raises() -> None:
    """Test that unknown class raises ValueError."""
    ann = make_annotation("Truck")
    class_mapping = {"Car": 0}
    image_size = (640, 640)
    transformed_p2 = np.eye(3, 4, dtype=np.float64)

    try:
        encode_annotations([ann], class_mapping, image_size, transformed_p2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Truck" in str(e)
        assert "not in class mapping" in str(e)


def test_decode_bbox_2d() -> None:
    """Test decoding normalised bbox back to pixels."""
    bbox_norm = np.array([0.1, 0.2, 0.5, 0.6])
    image_size = (640, 480)

    decoded = decode_bbox_2d(bbox_norm, image_size)

    expected = np.array([64.0, 96.0, 320.0, 288.0])
    assert np.allclose(decoded, expected)


def test_decode_bbox_2d_batch() -> None:
    """Test decoding batch of normalised bboxes."""
    bboxes_norm = np.array([
        [0.1, 0.2, 0.5, 0.6],
        [0.0, 0.0, 1.0, 1.0],
    ])
    image_size = (640, 480)

    decoded = decode_bbox_2d(bboxes_norm, image_size)

    assert decoded.shape == (2, 4)
    assert np.allclose(decoded[0], [64.0, 96.0, 320.0, 288.0])
    assert np.allclose(decoded[1], [0.0, 0.0, 640.0, 480.0])


def test_encode_annotations_preserves_dtype() -> None:
    """Test that encoded tensors have correct dtypes."""
    ann = make_annotation("Car")
    class_mapping = {"Car": 0}
    image_size = (640, 640)
    transformed_p2 = np.eye(3, 4, dtype=np.float64)

    target = encode_annotations([ann], class_mapping, image_size, transformed_p2)

    assert target.class_ids.dtype == torch.long
    assert target.bboxes_2d.dtype == torch.float32
    assert target.dimensions_3d.dtype == torch.float32
    assert target.locations_3d.dtype == torch.float32
    assert target.rotation_y.dtype == torch.float32
    assert target.transformed_p2.dtype == torch.float32


def test_encode_annotations_transformed_p2_passed_through() -> None:
    """Test that transformed_p2 is passed through correctly."""
    ann = make_annotation("Car")
    class_mapping = {"Car": 0}
    image_size = (640, 640)
    # Create a specific P2 matrix
    transformed_p2 = np.array([
        [500.0, 0.0, 320.0, 0.0],
        [0.0, 500.0, 240.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)

    target = encode_annotations([ann], class_mapping, image_size, transformed_p2)

    assert torch.allclose(target.transformed_p2, torch.from_numpy(transformed_p2.astype(np.float32)))