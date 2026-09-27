"""Numerical tests for EXP-002 image and projection transformations."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.data.kitti_parser import BoundingBox2D
from src.data.transforms import (
    inverse_transform_bbox,
    make_letterbox_transform,
    make_resize_transform,
    preprocess_image,
    transform_bbox,
    transform_projection_matrix,
)


def test_resize_transforms_bbox_with_independent_axis_scales() -> None:
    transform = make_resize_transform(100, 200, 200, 100)

    transformed = transform_bbox(BoundingBox2D(10.0, 20.0, 30.0, 40.0), transform)

    assert transformed == BoundingBox2D(20.0, 10.0, 60.0, 20.0)
    assert inverse_transform_bbox(transformed, transform) == BoundingBox2D(10.0, 20.0, 30.0, 40.0)


def test_letterbox_transforms_bbox_and_records_reversible_padding() -> None:
    transform = make_letterbox_transform(200, 100, 300, 300)

    transformed = transform_bbox(BoundingBox2D(20.0, 10.0, 100.0, 50.0), transform)

    assert transform.scale_x == transform.scale_y == 1.5
    assert (transform.pad_x, transform.pad_y, transform.pad_right, transform.pad_bottom) == (0, 75, 0, 75)
    assert transformed == BoundingBox2D(30.0, 90.0, 150.0, 150.0)
    assert inverse_transform_bbox(transformed, transform) == BoundingBox2D(20.0, 10.0, 100.0, 50.0)


def test_resize_scales_camera_intrinsics_and_full_projection_matrix() -> None:
    projection = np.asarray([[100.0, 0.0, 50.0, 4.0], [0.0, 120.0, 60.0, 8.0], [0.0, 0.0, 1.0, 0.0]])
    transform = make_resize_transform(100, 200, 200, 100)

    transformed = transform_projection_matrix(projection, transform)

    assert transformed[0, 0] == 200.0
    assert transformed[1, 1] == 60.0
    assert transformed[0, 2] == 100.0
    assert transformed[1, 2] == 30.0
    assert np.allclose(transformed, transform.matrix @ projection)


def test_letterbox_translates_principal_point_and_uses_a_times_p() -> None:
    projection = np.asarray([[100.0, 0.0, 50.0, 0.0], [0.0, 120.0, 40.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    transform = make_letterbox_transform(200, 100, 300, 300)

    transformed = transform_projection_matrix(projection, transform)

    assert transformed[0, 0] == 150.0
    assert transformed[1, 1] == 180.0
    assert transformed[0, 2] == 75.0
    assert transformed[1, 2] == 135.0
    assert np.allclose(transformed, transform.matrix @ projection)


def test_preprocess_image_applies_letterbox_canvas_and_padding_value() -> None:
    image = Image.new("RGB", (200, 100), color=(10, 20, 30))
    transform = make_letterbox_transform(200, 100, 300, 300)

    processed = preprocess_image(image, transform, padding_value=114)

    assert processed.size == (300, 300)
    assert processed.getpixel((5, 5)) == (114, 114, 114)
    assert processed.getpixel((5, 100)) == (10, 20, 30)
    processed.close()
    image.close()
