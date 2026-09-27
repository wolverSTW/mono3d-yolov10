"""Parsing KITTI camera calibration files used by the monocular pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


CALIBRATION_MATRIX_SHAPES: dict[str, tuple[int, int]] = {
    "P0": (3, 4),
    "P1": (3, 4),
    "P2": (3, 4),
    "P3": (3, 4),
    "R0_rect": (3, 3),
    "R_rect": (3, 3),
    "Tr_velo_to_cam": (3, 4),
    "Tr_imu_to_velo": (3, 4),
}


class KittiCalibrationError(ValueError):
    """A malformed calibration entry with file and line context."""

    def __init__(self, message: str, *, calibration_path: Path, line_number: int | None = None) -> None:
        prefix = str(calibration_path)
        if line_number is not None:
            prefix += f":{line_number}"
        super().__init__(f"{prefix}: {message}")


@dataclass(frozen=True)
class KittiCalibration:
    """Validated KITTI calibration entries, preserving each complete matrix."""

    matrices: dict[str, np.ndarray]

    def matrix(self, key: str) -> np.ndarray | None:
        """Return a matrix by its KITTI key, or ``None`` when it is absent."""
        return self.matrices.get(key)

    @property
    def p2(self) -> np.ndarray:
        """Return the required 3 × 4 camera-2 projection matrix."""
        return self.matrices["P2"]

    @property
    def camera_2_intrinsics(self) -> tuple[float, float, float, float]:
        """Derive ``(fx, fy, cx, cy)`` directly from the preserved P2 matrix."""
        return (
            float(self.p2[0, 0]),
            float(self.p2[1, 1]),
            float(self.p2[0, 2]),
            float(self.p2[1, 2]),
        )


def _parse_values(
    raw_values: str,
    *,
    key: str,
    calibration_path: Path,
    line_number: int,
) -> list[float]:
    """Parse finite calibration numbers with useful source context."""
    values: list[float] = []
    for raw_value in raw_values.split():
        try:
            value = float(raw_value)
        except ValueError as error:
            raise KittiCalibrationError(
                f"{key} contains a non-numeric value: {raw_value!r}",
                calibration_path=calibration_path,
                line_number=line_number,
            ) from error
        if not math.isfinite(value):
            raise KittiCalibrationError(
                f"{key} contains a non-finite value: {raw_value!r}",
                calibration_path=calibration_path,
                line_number=line_number,
            )
        values.append(value)
    return values


def parse_calibration_file(
    calibration_path: str | Path,
    *,
    require_p2: bool = True,
) -> KittiCalibration:
    """Parse standard KITTI calibration entries and validate known dimensions.

    ``P0``–``P3`` and the common rectification/extrinsic keys are accepted when
    present.  P2 is required by default because camera-2 RGB geometry relies
    on its complete 3 × 4 projection matrix.
    """
    calibration_path = Path(calibration_path)
    try:
        lines = calibration_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise KittiCalibrationError(
            f"could not read calibration file: {error}", calibration_path=calibration_path
        ) from error

    matrices: dict[str, np.ndarray] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise KittiCalibrationError(
                "expected a 'key: values' entry", calibration_path=calibration_path, line_number=line_number
            )
        key, raw_values = line.split(":", maxsplit=1)
        key = key.strip()
        if not key:
            raise KittiCalibrationError(
                "calibration key is empty", calibration_path=calibration_path, line_number=line_number
            )
        if key in matrices:
            raise KittiCalibrationError(
                f"duplicate calibration key: {key}", calibration_path=calibration_path, line_number=line_number
            )

        values = _parse_values(
            raw_values,
            key=key,
            calibration_path=calibration_path,
            line_number=line_number,
        )
        shape = CALIBRATION_MATRIX_SHAPES.get(key)
        if shape is not None:
            expected_count = shape[0] * shape[1]
            if len(values) != expected_count:
                raise KittiCalibrationError(
                    f"{key} requires {expected_count} values for shape {shape}, found {len(values)}",
                    calibration_path=calibration_path,
                    line_number=line_number,
                )
            matrices[key] = np.asarray(values, dtype=np.float64).reshape(shape)
        else:
            matrices[key] = np.asarray(values, dtype=np.float64)

    if require_p2 and "P2" not in matrices:
        raise KittiCalibrationError("required P2 matrix is missing", calibration_path=calibration_path)
    return KittiCalibration(matrices=matrices)
