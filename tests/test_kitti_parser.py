"""Unit tests for structured KITTI annotation parsing and validation."""

from __future__ import annotations

import math

import pytest

from src.data.kitti_parser import (
    CameraLocation3D,
    Dimensions3D,
    KittiAnnotationError,
    parse_annotation_line,
    parse_label_file,
    validate_dimensions,
    validate_location,
)


def test_parse_valid_annotation_preserves_dimension_and_orientation_conventions(synthetic_kitti) -> None:
    line = synthetic_kitti.label_line(alpha=-0.4, rotation_y=1.2)

    annotation = parse_annotation_line(line, frame_id="000001", line_number=3)

    assert annotation.class_name == "Car"
    assert annotation.dimensions_3d == Dimensions3D(height=1.5, width=1.6, length=3.8)
    assert annotation.location_3d == CameraLocation3D(x=1.0, y=1.5, z=20.0)
    assert annotation.alpha == -0.4
    assert annotation.rotation_y == 1.2
    assert annotation.alpha != annotation.rotation_y


def test_parser_reports_malformed_field_count_with_frame_and_line_context(synthetic_kitti) -> None:
    synthetic_kitti.write_label("000002", "Car 0.0 0\n")

    with pytest.raises(KittiAnnotationError, match=r"frame 000002 line 1: expected 15 fields"):
        parse_label_file(synthetic_kitti.root / "label_2" / "000002.txt")


def test_parser_reports_non_numeric_values(synthetic_kitti) -> None:
    line = synthetic_kitti.label_line().replace("20.0 0.2", "not-a-depth 0.2")

    with pytest.raises(KittiAnnotationError, match="location_z is not numeric"):
        parse_annotation_line(line, frame_id="000003", line_number=1)


def test_dontcare_is_preserved_without_physical_dimension_constraints(synthetic_kitti) -> None:
    dontcare = "DontCare -1 -1 -10 0 0 50 40 -1 -1 -1 -1000 -1000 -1000 -10"

    annotation = parse_annotation_line(dontcare)

    assert annotation.is_dontcare is True
    assert annotation.dimensions_3d.height == -1.0


def test_parser_rejects_invalid_bounding_box(synthetic_kitti) -> None:
    line = synthetic_kitti.label_line(bbox=(50.0, 8.0, 10.0, 40.0))

    with pytest.raises(KittiAnnotationError, match="x1 < x2"):
        parse_annotation_line(line)


def test_parser_rejects_nonpositive_dimensions_and_location_validation_is_finite(synthetic_kitti) -> None:
    line = synthetic_kitti.label_line(dimensions=(0.0, 1.6, 3.8))

    with pytest.raises(KittiAnnotationError, match="height must be positive"):
        parse_annotation_line(line)
    assert validate_dimensions(Dimensions3D(1.0, 2.0, 3.0)).valid is True
    assert validate_location(CameraLocation3D(0.0, 1.0, math.nan)).valid is False
