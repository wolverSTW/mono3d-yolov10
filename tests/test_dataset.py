"""Tests for KITTI PyTorch dataset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.data.dataset import KittiManifestDataset, kitti_collate_fn
from src.data.kitti_parser import parse_label_file
from src.data.transforms import (
    ImageTransform,
    PreprocessingMode,
    make_letterbox_transform,
    transform_projection_matrix,
)


def _create_synthetic_kitti_structure(root: Path, frame_ids: list[str]) -> list:
    """Create synthetic KITTI files and return manifest data for them."""
    for d in ("image_2", "label_2", "calib"):
        (root / d).mkdir(parents=True, exist_ok=True)

    manifest_frames = []
    for i, frame_id in enumerate(frame_ids):
        # Create image
        img = Image.new("RGB", (100, 80), color=(10 + i * 10, 20, 30))
        img.save(root / "image_2" / f"{frame_id}.png")

        # Create label - one Car per frame
        label_content = f"Car 0.0 0 0.1 10.0 8.0 50.0 40.0 1.5 1.6 3.8 1.0 1.5 20.0 0.2\n"
        (root / "label_2" / f"{frame_id}.txt").write_text(label_content, encoding="utf-8")

        # Create calibration
        proj = "100 0 50 10 0 120 40 20 0 0 1 0"
        calib_content = (
            f"P0: {proj}\n"
            f"P1: {proj}\n"
            f"P2: {proj}\n"
            f"P3: {proj}\n"
            "R0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        )
        (root / "calib" / f"{frame_id}.txt").write_text(calib_content, encoding="utf-8")

        # Compute expected transform for letterbox 100x80 -> 200x200
        transform = make_letterbox_transform(100, 80, 200, 200)
        p2_orig = np.array([
            [100.0, 0.0, 50.0, 10.0],
            [0.0, 120.0, 40.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        transformed_p2 = transform_projection_matrix(p2_orig, transform)

        manifest_frames.append({
            "frame_id": frame_id,
            "image_path": f"image_2/{frame_id}.png",
            "label_path": f"label_2/{frame_id}.txt",
            "calibration_path": f"calib/{frame_id}.txt",
            "split": "train" if i < 2 else "validation",
            "original_image_size": {"width": 100, "height": 80},
            "annotation_count": 1,
            "dontcare_count": 0,
            "preprocessing": {
                "mode": "letterbox",
                "original_width": 100,
                "original_height": 80,
                "output_width": 200,
                "output_height": 200,
                "resized_width": transform.resized_width,
                "resized_height": transform.resized_height,
                "scale_x": transform.scale_x,
                "scale_y": transform.scale_y,
                "pad_x": transform.pad_x,
                "pad_y": transform.pad_y,
                "pad_right": transform.pad_right,
                "pad_bottom": transform.pad_bottom,
            },
            "transformed_p2": transformed_p2.tolist(),
        })

    return manifest_frames


def _create_manifest(root: Path, manifest_frames: list, manifest_path: Path) -> None:
    """Create a manifest JSON file."""
    manifest = {
        "format": "kitti-preparation-manifest-v1",
        "paths_are_relative_to": "dataset_root",
        "preprocessing_mode": "letterbox",
        "output_image_size": {"width": 200, "height": 200},
        "split": {
            "validation_ratio": 0.33,
            "seed": 42,
            "train_frame_ids": [f["frame_id"] for f in manifest_frames if f["split"] == "train"],
            "validation_frame_ids": [f["frame_id"] for f in manifest_frames if f["split"] == "validation"],
        },
        "frames": manifest_frames,
        "invalid_samples": [],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def test_dataset_construction() -> None:
    """Test dataset can be constructed from manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        frame_ids = ["000000", "000001", "000002"]
        manifest_frames = _create_synthetic_kitti_structure(root, frame_ids)
        _create_manifest(root, manifest_frames, manifest_path)

        # Construct dataset
        dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="train",
            class_mapping=["Car"],
        )

        assert len(dataset) == 2
        assert dataset.num_classes == 1
        assert dataset.class_to_idx == {"Car": 0}
        assert dataset.idx_to_class == ["Car"]


def test_dataset_getitem_returns_correct_types() -> None:
    """Test that __getitem__ returns KittiSample with correct tensor types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        frame_ids = ["000000", "000001", "000002"]
        manifest_frames = _create_synthetic_kitti_structure(root, frame_ids)
        _create_manifest(root, manifest_frames, manifest_path)

        dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="train",
            class_mapping=["Car"],
        )

        sample = dataset[0]

        # Check frame_id
        assert sample.frame_id == "000000"

        # Check image tensor
        assert isinstance(sample.image, torch.Tensor)
        assert sample.image.dtype == torch.float32
        assert sample.image.shape == (3, 200, 200)  # C, H, W
        assert sample.image.min() >= 0.0 and sample.image.max() <= 1.0

        # Check target
        target = sample.target
        assert target.class_ids.dtype == torch.long
        assert target.class_ids.shape == (1,)
        assert target.class_ids[0].item() == 0
        assert target.bboxes_2d.dtype == torch.float32
        assert target.bboxes_2d.shape == (1, 4)
        assert target.dimensions_3d.dtype == torch.float32
        assert target.dimensions_3d.shape == (1, 3)
        assert target.locations_3d.dtype == torch.float32
        assert target.locations_3d.shape == (1, 3)
        assert target.rotation_y.dtype == torch.float32
        assert target.rotation_y.shape == (1,)

        # Check bbox is normalised to [0, 1]
        bbox = target.bboxes_2d[0]
        assert (bbox >= 0.0).all() and (bbox <= 1.0).all()

        # Check dimensions match original (H=1.5, W=1.6, L=3.8)
        dims = target.dimensions_3d[0]
        assert torch.allclose(dims, torch.tensor([1.5, 1.6, 3.8]))

        # Check location matches original (X=1.0, Y=1.5, Z=20.0)
        loc = target.locations_3d[0]
        assert torch.allclose(loc, torch.tensor([1.0, 1.5, 20.0]))

        # Check rotation_y
        assert torch.allclose(target.rotation_y[0], torch.tensor(0.2))

        # Check transformed_p2
        assert target.transformed_p2.shape == (3, 4)
        assert target.transformed_p2.dtype == torch.float32


def test_dataset_split_filtering() -> None:
    """Test that train/validation split correctly filters frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        frame_ids = ["000000", "000001", "000002"]
        manifest_frames = _create_synthetic_kitti_structure(root, frame_ids)
        _create_manifest(root, manifest_frames, manifest_path)

        train_dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="train",
            class_mapping=["Car"],
        )
        val_dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="validation",
            class_mapping=["Car"],
        )

        assert len(train_dataset) == 2
        assert len(val_dataset) == 1
        assert train_dataset[0].frame_id != val_dataset[0].frame_id
        assert train_dataset[1].frame_id != val_dataset[0].frame_id


def test_dataset_multiple_objects() -> None:
    """Test dataset handles multiple objects per frame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        frame_ids = ["000000"]
        for d in ("image_2", "label_2", "calib"):
            (root / d).mkdir(parents=True, exist_ok=True)

        # Create image
        img = Image.new("RGB", (100, 80), color=(10, 20, 30))
        img.save(root / "image_2" / "000000.png")

        # Create label with TWO objects
        label_content = (
            "Car 0.0 0 0.1 10.0 8.0 50.0 40.0 1.5 1.6 3.8 1.0 1.5 20.0 0.2\n"
            "Pedestrian 0.0 0 0.1 20.0 10.0 40.0 50.0 1.7 0.5 0.6 2.0 1.0 15.0 1.0\n"
        )
        (root / "label_2" / "000000.txt").write_text(label_content, encoding="utf-8")

        # Create calibration
        proj = "100 0 50 10 0 120 40 20 0 0 1 0"
        calib_content = (
            f"P0: {proj}\n"
            f"P1: {proj}\n"
            f"P2: {proj}\n"
            f"P3: {proj}\n"
            "R0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        )
        (root / "calib" / "000000.txt").write_text(calib_content, encoding="utf-8")

        transform = make_letterbox_transform(100, 80, 200, 200)
        p2_orig = np.array([
            [100.0, 0.0, 50.0, 10.0],
            [0.0, 120.0, 40.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        transformed_p2 = transform_projection_matrix(p2_orig, transform)

        manifest_frames = [{
            "frame_id": "000000",
            "image_path": "image_2/000000.png",
            "label_path": "label_2/000000.txt",
            "calibration_path": "calib/000000.txt",
            "split": "train",
            "original_image_size": {"width": 100, "height": 80},
            "annotation_count": 2,
            "dontcare_count": 0,
            "preprocessing": {
                "mode": "letterbox",
                "original_width": 100,
                "original_height": 80,
                "output_width": 200,
                "output_height": 200,
                "resized_width": transform.resized_width,
                "resized_height": transform.resized_height,
                "scale_x": transform.scale_x,
                "scale_y": transform.scale_y,
                "pad_x": transform.pad_x,
                "pad_y": transform.pad_y,
                "pad_right": transform.pad_right,
                "pad_bottom": transform.pad_bottom,
            },
            "transformed_p2": transformed_p2.tolist(),
        }]

        _create_manifest(root, manifest_frames, manifest_path)

        dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="train",
            class_mapping=["Car", "Pedestrian"],
        )

        sample = dataset[0]
        assert sample.target.class_ids.shape == (2,)
        assert set(sample.target.class_ids.tolist()) == {0, 1}
        assert sample.target.bboxes_2d.shape == (2, 4)
        assert sample.target.dimensions_3d.shape == (2, 3)
        assert sample.target.locations_3d.shape == (2, 3)


def test_dataset_excludes_dontcare() -> None:
    """Test that DontCare objects are excluded from targets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        for d in ("image_2", "label_2", "calib"):
            (root / d).mkdir(parents=True, exist_ok=True)

        # Create image
        img = Image.new("RGB", (100, 80), color=(10, 20, 30))
        img.save(root / "image_2" / "000000.png")

        # Create label with Car + DontCare
        label_content = (
            "Car 0.0 0 0.1 10.0 8.0 50.0 40.0 1.5 1.6 3.8 1.0 1.5 20.0 0.2\n"
            "DontCare -1 -1 -10 0 0 50 40 -1 -1 -1 -1000 -1000 -1000 -10\n"
        )
        (root / "label_2" / "000000.txt").write_text(label_content, encoding="utf-8")

        # Create calibration
        proj = "100 0 50 10 0 120 40 20 0 0 1 0"
        calib_content = f"P2: {proj}\n"
        (root / "calib" / "000000.txt").write_text(calib_content, encoding="utf-8")

        transform = make_letterbox_transform(100, 80, 200, 200)
        p2_orig = np.array([
            [100.0, 0.0, 50.0, 10.0],
            [0.0, 120.0, 40.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        transformed_p2 = transform_projection_matrix(p2_orig, transform)

        manifest_frames = [{
            "frame_id": "000000",
            "image_path": "image_2/000000.png",
            "label_path": "label_2/000000.txt",
            "calibration_path": "calib/000000.txt",
            "split": "train",
            "original_image_size": {"width": 100, "height": 80},
            "annotation_count": 1,
            "dontcare_count": 1,
            "preprocessing": {
                "mode": "letterbox",
                "original_width": 100,
                "original_height": 80,
                "output_width": 200,
                "output_height": 200,
                "resized_width": transform.resized_width,
                "resized_height": transform.resized_height,
                "scale_x": transform.scale_x,
                "scale_y": transform.scale_y,
                "pad_x": transform.pad_x,
                "pad_y": transform.pad_y,
                "pad_right": transform.pad_right,
                "pad_bottom": transform.pad_bottom,
            },
            "transformed_p2": transformed_p2.tolist(),
        }]

        _create_manifest(root, manifest_frames, manifest_path)

        dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="train",
            class_mapping=["Car"],
        )

        sample = dataset[0]
        assert sample.target.class_ids.shape == (1,)  # Only Car, not DontCare


def test_collate_fn() -> None:
    """Test collate function produces correct batch structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        frame_ids = ["000000", "000001"]
        manifest_frames = _create_synthetic_kitti_structure(root, frame_ids)
        _create_manifest(root, manifest_frames, manifest_path)

        dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="train",
            class_mapping=["Car"],
        )

        # Get two samples
        samples = [dataset[0], dataset[1]]

        # Collate
        batch = kitti_collate_fn(samples)

        assert "images" in batch
        assert "targets" in batch
        assert "frame_ids" in batch
        assert "original_sizes" in batch
        assert "calibrations" in batch

        assert batch["images"].shape == (2, 3, 200, 200)
        assert len(batch["targets"]) == 2
        assert len(batch["frame_ids"]) == 2
        assert len(batch["original_sizes"]) == 2
        assert len(batch["calibrations"]) == 2

        # Each target should be valid
        for target in batch["targets"]:
            assert target.class_ids.shape == (1,)
            assert target.bboxes_2d.shape == (1, 4)


def test_dataset_unknown_class_raises() -> None:
    """Test that unknown class in annotations raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        for d in ("image_2", "label_2", "calib"):
            (root / d).mkdir(parents=True, exist_ok=True)

        # Create image
        img = Image.new("RGB", (100, 80), color=(10, 20, 30))
        img.save(root / "image_2" / "000000.png")

        # Create label with UNKNOWN class
        label_content = "Truck 0.0 0 0.1 10.0 8.0 50.0 40.0 1.5 1.6 3.8 1.0 1.5 20.0 0.2\n"
        (root / "label_2" / "000000.txt").write_text(label_content, encoding="utf-8")

        # Create calibration
        proj = "100 0 50 10 0 120 40 20 0 0 1 0"
        (root / "calib" / "000000.txt").write_text(f"P2: {proj}\n", encoding="utf-8")

        transform = make_letterbox_transform(100, 80, 200, 200)
        p2_orig = np.array([
            [100.0, 0.0, 50.0, 10.0],
            [0.0, 120.0, 40.0, 20.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        transformed_p2 = transform_projection_matrix(p2_orig, transform)

        manifest_frames = [{
            "frame_id": "000000",
            "image_path": "image_2/000000.png",
            "label_path": "label_2/000000.txt",
            "calibration_path": "calib/000000.txt",
            "split": "train",
            "original_image_size": {"width": 100, "height": 80},
            "annotation_count": 1,
            "dontcare_count": 0,
            "preprocessing": {
                "mode": "letterbox",
                "original_width": 100,
                "original_height": 80,
                "output_width": 200,
                "output_height": 200,
                "resized_width": transform.resized_width,
                "resized_height": transform.resized_height,
                "scale_x": transform.scale_x,
                "scale_y": transform.scale_y,
                "pad_x": transform.pad_x,
                "pad_y": transform.pad_y,
                "pad_right": transform.pad_right,
                "pad_bottom": transform.pad_bottom,
            },
            "transformed_p2": transformed_p2.tolist(),
        }]

        _create_manifest(root, manifest_frames, manifest_path)

        dataset = KittiManifestDataset(
            manifest_path=manifest_path,
            dataset_root=root,
            split="train",
            class_mapping=["Car"],  # Truck not in mapping
        )

        try:
            _ = dataset[0]
            assert False, "Should have raised ValueError for unknown class"
        except ValueError as e:
            assert "Truck" in str(e)
            assert "not in class mapping" in str(e)


def test_dataset_empty_split_raises() -> None:
    """Test that empty split raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "kitti"
        manifest_path = Path(tmpdir) / "manifest.json"

        for d in ("image_2", "label_2", "calib"):
            (root / d).mkdir(parents=True, exist_ok=True)

        # Create manifest with no frames
        manifest = {
            "format": "kitti-preparation-manifest-v1",
            "paths_are_relative_to": "dataset_root",
            "preprocessing_mode": "letterbox",
            "output_image_size": {"width": 200, "height": 200},
            "split": {
                "validation_ratio": 0.2,
                "seed": 42,
                "train_frame_ids": [],
                "validation_frame_ids": [],
            },
            "frames": [],
            "invalid_samples": [],
        }
        manifest_path.write_text(json.dumps(manifest))

        try:
            KittiManifestDataset(
                manifest_path=manifest_path,
                dataset_root=root,
                split="train",
                class_mapping=["Car"],
            )
            assert False, "Should have raised ValueError for empty split"
        except ValueError as e:
            assert "No frames found for split" in str(e)