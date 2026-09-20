"""Validation utilities for the KITTI 3D Object Detection dataset."""

from pathlib import Path


def validate_kitti_dataset(dataset_root: str | Path) -> dict:
    """Validate the expected KITTI dataset directory structure."""
    dataset_root = Path(dataset_root)

    required_directories = {
        "image_dir": dataset_root / "image_2",
        "label_dir": dataset_root / "label_2",
        "calibration_dir": dataset_root / "calib",
    }

    validation = {
        name: path.is_dir()
        for name, path in required_directories.items()
    }

    validation["dataset_root"] = dataset_root.is_dir()
    validation["valid"] = all(validation.values())

    return validation


if __name__ == "__main__":
    result = validate_kitti_dataset("data/raw/kitti")

    for name, status in result.items():
        print(f"{name}: {status}")
