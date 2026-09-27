"""Temporary synthetic KITTI fixtures shared by EXP-002 tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import pytest


def _projection_values(fx: float = 100.0, fy: float = 120.0, cx: float = 50.0, cy: float = 40.0) -> str:
    return f"{fx} 0 {cx} 10 0 {fy} {cy} 20 0 0 1 0"


@dataclass
class SyntheticKitti:
    """Create only temporary KITTI-style files for unit and CLI tests."""

    root: Path

    def label_line(
        self,
        class_name: str = "Car",
        *,
        bbox: tuple[float, float, float, float] = (10.0, 8.0, 50.0, 40.0),
        dimensions: tuple[float, float, float] = (1.5, 1.6, 3.8),
        location: tuple[float, float, float] = (1.0, 1.5, 20.0),
        alpha: float = 0.1,
        rotation_y: float = 0.2,
    ) -> str:
        """Return a standard 15-field KITTI annotation line."""
        return (
            f"{class_name} 0.0 0 {alpha} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} "
            f"{dimensions[0]} {dimensions[1]} {dimensions[2]} "
            f"{location[0]} {location[1]} {location[2]} {rotation_y}"
        )

    def write_image(self, frame_id: str, *, size: tuple[int, int] = (100, 80)) -> Path:
        """Write a small readable RGB image."""
        image_path = self.root / "image_2" / f"{frame_id}.png"
        Image.new("RGB", size, color=(10, 20, 30)).save(image_path)
        return image_path

    def write_label(self, frame_id: str, contents: str | None = None) -> Path:
        """Write a label file, defaulting to one valid Car annotation."""
        label_path = self.root / "label_2" / f"{frame_id}.txt"
        label_path.write_text(
            contents if contents is not None else self.label_line() + "\n",
            encoding="utf-8",
        )
        return label_path

    def write_calibration(self, frame_id: str, contents: str | None = None) -> Path:
        """Write a KITTI calibration fixture containing common standard entries."""
        calibration_path = self.root / "calib" / f"{frame_id}.txt"
        if contents is None:
            projection = _projection_values()
            contents = "\n".join(
                (
                    f"P0: {projection}",
                    f"P1: {projection}",
                    f"P2: {projection}",
                    f"P3: {projection}",
                    "R0_rect: 1 0 0 0 1 0 0 0 1",
                    "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0",
                    "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0",
                    "",
                )
            )
        calibration_path.write_text(contents, encoding="utf-8")
        return calibration_path

    def write_frame(
        self,
        frame_id: str,
        *,
        label_contents: str | None = None,
        image_size: tuple[int, int] = (100, 80),
        calibration_contents: str | None = None,
    ) -> None:
        """Write a matched synthetic image, label, and calibration frame."""
        self.write_image(frame_id, size=image_size)
        self.write_label(frame_id, label_contents)
        self.write_calibration(frame_id, calibration_contents)


@pytest.fixture
def synthetic_kitti(tmp_path: Path) -> SyntheticKitti:
    """Provide an empty temporary KITTI directory tree."""
    root = tmp_path / "kitti"
    for directory_name in ("image_2", "label_2", "calib"):
        (root / directory_name).mkdir(parents=True)
    return SyntheticKitti(root=root)
