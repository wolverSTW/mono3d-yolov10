"""Tests for the non-destructive EXP-001 KITTI dataset audit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from src.data.kitti_audit import audit_kitti_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "kitti"
    for directory_name in ("image_2", "label_2", "calib"):
        (root / directory_name).mkdir(parents=True)
    return root


def _write_image(root: Path, frame_id: str, suffix: str = ".png", mode: str = "RGB") -> Path:
    image_path = root / "image_2" / f"{frame_id}{suffix}"
    Image.new(mode, (32, 16), color=0).save(image_path)
    return image_path


def _label_line(object_type: str = "Car") -> str:
    return f"{object_type} 0.0 0 0.1 1.0 2.0 20.0 30.0 1.5 1.6 3.8 1.0 1.5 20.0 0.2"


def _write_label(root: Path, frame_id: str, contents: str | None = None) -> Path:
    label_path = root / "label_2" / f"{frame_id}.txt"
    label_path.write_text(contents if contents is not None else _label_line() + "\n", encoding="utf-8")
    return label_path


def _write_calibration(root: Path, frame_id: str, contents: str | None = None) -> Path:
    calibration_path = root / "calib" / f"{frame_id}.txt"
    if contents is None:
        contents = "P2: " + " ".join(str(value) for value in range(12)) + "\n"
    calibration_path.write_text(contents, encoding="utf-8")
    return calibration_path


def _write_valid_frame(root: Path, frame_id: str = "000000", label_contents: str | None = None) -> None:
    _write_image(root, frame_id)
    _write_label(root, frame_id, label_contents)
    _write_calibration(root, frame_id)


def test_valid_frame_matching_and_successful_audit(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root)

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is True
    assert report["overall_status"] == "passed"
    assert report["frame_matching"]["matched_frame_ids"] == ["000000"]
    assert report["image_details"] == [
        {
            "frame_id": "000000",
            "file": "image_2/000000.png",
            "width": 32,
            "height": 16,
            "mode": "RGB",
            "channels": 3,
        }
    ]


def test_missing_label_fails_audit(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_image(root, "000001")
    _write_calibration(root, "000001")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert report["frame_matching"]["image_ids_missing_labels"] == ["000001"]
    assert report["missing"]["labels"] == ["000001"]


def test_missing_calibration_fails_audit(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_image(root, "000002")
    _write_label(root, "000002")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert report["frame_matching"]["image_ids_missing_calibration"] == ["000002"]
    assert report["missing"]["calibration"] == ["000002"]


def test_missing_image_fails_audit(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_label(root, "000003")
    _write_calibration(root, "000003")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert report["frame_matching"]["label_ids_missing_images"] == ["000003"]
    assert report["missing"]["images"] == ["000003"]


def test_malformed_label_field_count_is_reported(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root, label_contents="Car 0.0 0\n")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert report["malformed_labels"][0]["frame_id"] == "000000"
    assert "expected 15 fields" in report["malformed_labels"][0]["reason"]


def test_invalid_numeric_label_field_is_reported(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root, label_contents=_label_line().replace("20.0 0.2", "invalid 0.2") + "\n")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert "location_z is not numeric" in report["malformed_labels"][0]["reason"]


def test_malformed_calibration_is_reported(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root)
    _write_calibration(root, "000000", "P2: 0 1 2\n")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert "P2 must contain 12 values" in report["malformed_calibrations"][0]["reason"]


def test_unreadable_image_is_reported(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    (root / "image_2" / "000000.png").write_bytes(b"not an image")
    _write_label(root, "000000")
    _write_calibration(root, "000000")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert report["unreadable_images"][0]["frame_id"] == "000000"


def test_duplicate_identifier_is_preserved_and_reported(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root)
    _write_image(root, "000000", suffix=".jpg")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is False
    assert report["file_counts"]["images"] == 2
    assert report["duplicate_ids"]["images"]["000000"] == [
        "image_2/000000.jpg",
        "image_2/000000.png",
    ]


def test_class_distribution_and_configured_class_comparison(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root, label_contents=_label_line("Car") + "\n" + _label_line("Van") + "\n")

    report = audit_kitti_dataset(root, configured_classes=["Car"])

    assert report["class_distribution"]["class_counts"] == {"Car": 1, "Van": 1}
    assert report["class_distribution"]["observed_detection_classes_not_configured"] == ["Van"]
    assert report["class_distribution"]["class_mapping_configured"] is True


def test_dontcare_is_explicitly_counted_not_a_detection_class(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root, label_contents=_label_line("DontCare") + "\n")

    report = audit_kitti_dataset(root)

    assert report["is_valid"] is True
    assert report["class_distribution"]["dontcare_count"] == 1
    assert report["class_distribution"]["observed_classes"] == ["DontCare"]


def test_empty_dataset_fails_audit(tmp_path: Path) -> None:
    report = audit_kitti_dataset(_dataset_root(tmp_path))

    assert report["is_valid"] is False
    assert "one or more required modalities contain no recognised files" in report["failure_reasons"]


def test_cli_writes_json_report_for_synthetic_dataset(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    _write_valid_frame(root)
    output_path = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_kitti.py",
            "--data-root",
            str(root),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "KITTI audit: PASSED" in completed.stdout
    assert json.loads(output_path.read_text(encoding="utf-8"))["is_valid"] is True


def test_cli_returns_nonzero_and_writes_report_for_failed_audit(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path)
    output_path = tmp_path / "failed-audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_kitti.py",
            "--data-root",
            str(root),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "KITTI audit: FAILED" in completed.stdout
    assert json.loads(output_path.read_text(encoding="utf-8"))["is_valid"] is False
