"""Tests for deterministic split generation and preparation manifests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.data.converter import generate_train_validation_split, prepare_kitti_dataset, write_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_three_valid_frames(synthetic_kitti) -> None:
    for frame_id in ("000000", "000001", "000002"):
        synthetic_kitti.write_frame(frame_id)


def test_split_is_deterministic_reproducible_and_has_no_overlap() -> None:
    frame_ids = [f"{index:06d}" for index in range(10)]

    first_train, first_validation = generate_train_validation_split(
        frame_ids, validation_ratio=0.2, seed=42
    )
    second_train, second_validation = generate_train_validation_split(
        frame_ids, validation_ratio=0.2, seed=42
    )

    assert (first_train, first_validation) == (second_train, second_validation)
    assert set(first_train).isdisjoint(first_validation)
    assert set(first_train) | set(first_validation) == set(frame_ids)
    assert len(first_validation) == 2


def test_preparation_generates_portable_manifest_with_transformed_p2(synthetic_kitti, tmp_path: Path) -> None:
    _write_three_valid_frames(synthetic_kitti)

    result = prepare_kitti_dataset(
        synthetic_kitti.root,
        output_width=200,
        output_height=200,
        preprocessing_mode="letterbox",
        validation_ratio=0.34,
        seed=7,
    )
    manifest_path = write_manifest(result.manifest, tmp_path / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.is_valid is True
    assert len(manifest["frames"]) == 3
    assert {frame["split"] for frame in manifest["frames"]} == {"train", "validation"}
    first_frame = manifest["frames"][0]
    assert first_frame["image_path"].startswith("image_2/")
    assert first_frame["label_path"].startswith("label_2/")
    assert first_frame["calibration_path"].startswith("calib/")
    assert first_frame["preprocessing"]["mode"] == "letterbox"
    assert first_frame["transformed_p2"][0][0] == 200.0
    assert first_frame["transformed_p2"][1][1] == 240.0


def test_preparation_reports_invalid_frame_without_silently_discarding_it(synthetic_kitti) -> None:
    _write_three_valid_frames(synthetic_kitti)
    synthetic_kitti.write_frame(
        "000003",
        label_contents=synthetic_kitti.label_line(dimensions=(0.0, 1.0, 2.0)) + "\n",
    )

    result = prepare_kitti_dataset(
        synthetic_kitti.root,
        output_width=200,
        output_height=200,
        preprocessing_mode="resize",
        validation_ratio=0.25,
        seed=11,
    )

    assert result.is_valid is False
    assert len(result.manifest["frames"]) == 3
    assert result.manifest["invalid_samples"][0]["frame_id"] == "000003"
    assert "height must be positive" in result.manifest["invalid_samples"][0]["reason"]


def test_prepare_cli_succeeds_for_valid_synthetic_dataset(synthetic_kitti, tmp_path: Path) -> None:
    _write_three_valid_frames(synthetic_kitti)
    manifest_path = tmp_path / "prepared.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_kitti.py",
            "--data-root",
            str(synthetic_kitti.root),
            "--output",
            str(manifest_path),
            "--validation-ratio",
            "0.34",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "KITTI preparation: PASSED" in completed.stdout
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["frames"]


def test_prepare_cli_fails_but_writes_manifest_for_invalid_fixture(synthetic_kitti, tmp_path: Path) -> None:
    _write_three_valid_frames(synthetic_kitti)
    synthetic_kitti.write_frame("000003", label_contents="Car 0 0\n")
    manifest_path = tmp_path / "invalid.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_kitti.py",
            "--data-root",
            str(synthetic_kitti.root),
            "--output",
            str(manifest_path),
            "--validation-ratio",
            "0.25",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "KITTI preparation: FAILED" in completed.stdout
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["invalid_samples"]
