"""Unit tests for KITTI calibration parsing and camera-2 intrinsics."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.calibration import KittiCalibrationError, parse_calibration_file


def test_parse_standard_calibration_and_preserve_p2(synthetic_kitti) -> None:
    calibration_path = synthetic_kitti.write_calibration("000010")

    calibration = parse_calibration_file(calibration_path)

    assert calibration.p2.shape == (3, 4)
    assert calibration.matrix("P0").shape == (3, 4)
    assert calibration.matrix("R0_rect").shape == (3, 3)
    assert calibration.matrix("Tr_velo_to_cam").shape == (3, 4)
    assert calibration.camera_2_intrinsics == (100.0, 120.0, 50.0, 40.0)
    assert np.allclose(calibration.p2[2], [0.0, 0.0, 1.0, 0.0])


def test_calibration_parser_rejects_malformed_entry(synthetic_kitti) -> None:
    calibration_path = synthetic_kitti.write_calibration("000011", "P2 1 2 3\n")

    with pytest.raises(KittiCalibrationError, match="expected a 'key: values' entry"):
        parse_calibration_file(calibration_path)


def test_calibration_parser_rejects_invalid_matrix_length(synthetic_kitti) -> None:
    calibration_path = synthetic_kitti.write_calibration("000012", "P2: 1 2 3\n")

    with pytest.raises(KittiCalibrationError, match="P2 requires 12 values"):
        parse_calibration_file(calibration_path)


def test_calibration_parser_requires_p2_for_camera_2_geometry(synthetic_kitti) -> None:
    calibration_path = synthetic_kitti.write_calibration("000013", "P0: " + " ".join("1" for _ in range(12)))

    with pytest.raises(KittiCalibrationError, match="required P2 matrix is missing"):
        parse_calibration_file(calibration_path)
