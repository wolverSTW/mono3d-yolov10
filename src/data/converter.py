"""EXP-002 dataset preparation, frame-level splitting, and manifest generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.data.calibration import KittiCalibrationError, parse_calibration_file
from src.data.kitti_audit import (
    CALIBRATION_DIRECTORY_NAME,
    IMAGE_DIRECTORY_NAME,
    IMAGE_EXTENSIONS,
    LABEL_DIRECTORY_NAME,
    TEXT_EXTENSIONS,
    discover_frame_files,
)
from src.data.kitti_parser import KittiAnnotationError, parse_label_file, validate_kitti_object
from src.data.transforms import (
    ImageTransform,
    PreprocessingMode,
    make_image_transform,
    preprocess_image,
    transform_bbox,
    transform_projection_matrix,
)


@dataclass(frozen=True)
class PreparationIssue:
    """A frame-specific preparation problem retained in the output manifest."""

    frame_id: str | None
    reason: str

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-safe issue record."""
        return {"frame_id": self.frame_id, "reason": self.reason}


@dataclass(frozen=True)
class PreparedFrame:
    """A validated frame ready for a later dataset layer to consume."""

    frame_id: str
    image_path: str
    label_path: str
    calibration_path: str
    original_width: int
    original_height: int
    annotation_count: int
    dontcare_count: int
    transform: ImageTransform
    transformed_p2: np.ndarray

    def to_manifest_entry(self, split: Literal["train", "validation"]) -> dict[str, Any]:
        """Create a portable, JSON-safe manifest entry."""
        return {
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "label_path": self.label_path,
            "calibration_path": self.calibration_path,
            "split": split,
            "original_image_size": {"width": self.original_width, "height": self.original_height},
            "annotation_count": self.annotation_count,
            "dontcare_count": self.dontcare_count,
            "preprocessing": self.transform.to_dict(),
            "transformed_p2": self.transformed_p2.tolist(),
        }


@dataclass(frozen=True)
class PreparationResult:
    """In-memory manifest plus validation diagnostics from dataset preparation."""

    manifest: dict[str, Any]
    issues: tuple[PreparationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Whether every discovered frame was valid and split successfully."""
        return not self.issues and bool(self.manifest["frames"])


def generate_train_validation_split(
    frame_ids: list[str],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Create reproducible non-overlapping frame-level train/validation IDs.

    At least two valid frames are needed to guarantee that each split has a
    frame.  The nearest feasible validation count is used while preserving at
    least one frame in each split.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("frame_ids must be unique before splitting")
    if len(frame_ids) < 2:
        raise ValueError("at least two valid frames are required for a train/validation split")

    shuffled_ids = sorted(frame_ids)
    random.Random(seed).shuffle(shuffled_ids)
    validation_count = round(len(shuffled_ids) * validation_ratio)
    validation_count = max(1, min(validation_count, len(shuffled_ids) - 1))
    validation_ids = sorted(shuffled_ids[:validation_count])
    validation_id_set = set(validation_ids)
    train_ids = [frame_id for frame_id in sorted(frame_ids) if frame_id not in validation_id_set]
    return train_ids, validation_ids


def _relative_path(file_path: Path, dataset_root: Path) -> str:
    """Produce a portable path relative to the dataset root."""
    return file_path.relative_to(dataset_root).as_posix()


def _open_image_size(file_path: Path) -> tuple[int, int, int, Image.Image]:
    """Load an image and return its dimensions, channels, and open image object."""
    try:
        image = Image.open(file_path)
        image.load()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"unreadable image: {error}") from error
    width, height = image.size
    return width, height, len(image.getbands()), image


def _discover_file_groups(dataset_root: Path) -> dict[str, Any]:
    """Discover relevant files using the EXP-001 frame-ID convention."""
    return {
        "images": discover_frame_files(dataset_root / IMAGE_DIRECTORY_NAME, IMAGE_EXTENSIONS),
        "labels": discover_frame_files(dataset_root / LABEL_DIRECTORY_NAME, TEXT_EXTENSIONS),
        "calibration": discover_frame_files(dataset_root / CALIBRATION_DIRECTORY_NAME, TEXT_EXTENSIONS),
    }


def prepare_kitti_dataset(
    dataset_root: str | Path,
    *,
    output_width: int,
    output_height: int,
    preprocessing_mode: PreprocessingMode,
    validation_ratio: float,
    seed: int,
    expected_image_channels: int | None = 3,
    padding_value: int = 114,
) -> PreparationResult:
    """Validate frames and construct a split manifest without writing KITTI data.

    Original annotations and calibration files are read only.  The declared
    image transform is applied in memory to verify image processing and then
    recorded alongside the consistently transformed P2 matrix in the manifest.
    """
    dataset_root = Path(dataset_root)
    groups = _discover_file_groups(dataset_root)
    all_frame_ids = sorted(set().union(*(group.frame_ids for group in groups.values())))
    issues: list[PreparationIssue] = []
    prepared_frames: list[PreparedFrame] = []

    for frame_id in all_frame_ids:
        frame_paths: dict[str, Path] = {}
        frame_reasons: list[str] = []
        for modality, group in groups.items():
            paths = group.files_by_id.get(frame_id, ())
            if not paths:
                frame_reasons.append(f"missing {modality} file")
            elif len(paths) > 1:
                frame_reasons.append(f"duplicate {modality} files")
            else:
                frame_paths[modality] = paths[0]
        if frame_reasons:
            issues.append(PreparationIssue(frame_id=frame_id, reason="; ".join(frame_reasons)))
            continue

        image: Image.Image | None = None
        try:
            image_width, image_height, channels, image = _open_image_size(frame_paths["images"])
            if expected_image_channels is not None and channels != expected_image_channels:
                raise ValueError(f"expected {expected_image_channels} image channels, found {channels}")

            annotations = parse_label_file(frame_paths["labels"], frame_id=frame_id)
            for annotation in annotations:
                validation = validate_kitti_object(annotation, (image_width, image_height))
                if not validation.valid:
                    raise ValueError("; ".join(validation.errors))

            transform = make_image_transform(
                preprocessing_mode,
                image_width,
                image_height,
                output_width,
                output_height,
            )
            transformed_image = preprocess_image(image, transform, padding_value=padding_value)
            try:
                if transformed_image.size != (output_width, output_height):
                    raise ValueError("preprocessing output dimensions do not match the configured input size")
            finally:
                transformed_image.close()
            for annotation in annotations:
                if annotation.is_dontcare:
                    continue
                transformed_bbox = transform_bbox(annotation.bbox_2d, transform)
                transformed_validation = validate_kitti_object(
                    replace(annotation, bbox_2d=transformed_bbox),
                    (output_width, output_height),
                )
                if not transformed_validation.valid:
                    raise ValueError("transformed annotation invalid: " + "; ".join(transformed_validation.errors))

            calibration = parse_calibration_file(frame_paths["calibration"])
            transformed_p2 = transform_projection_matrix(calibration.p2, transform)
            prepared_frames.append(
                PreparedFrame(
                    frame_id=frame_id,
                    image_path=_relative_path(frame_paths["images"], dataset_root),
                    label_path=_relative_path(frame_paths["labels"], dataset_root),
                    calibration_path=_relative_path(frame_paths["calibration"], dataset_root),
                    original_width=image_width,
                    original_height=image_height,
                    annotation_count=sum(not annotation.is_dontcare for annotation in annotations),
                    dontcare_count=sum(annotation.is_dontcare for annotation in annotations),
                    transform=transform,
                    transformed_p2=transformed_p2,
                )
            )
        except (KittiAnnotationError, KittiCalibrationError, ValueError) as error:
            issues.append(PreparationIssue(frame_id=frame_id, reason=str(error)))
        finally:
            if image is not None:
                image.close()

    train_ids: list[str] = []
    validation_ids: list[str] = []
    if prepared_frames:
        try:
            train_ids, validation_ids = generate_train_validation_split(
                [frame.frame_id for frame in prepared_frames],
                validation_ratio=validation_ratio,
                seed=seed,
            )
        except ValueError as error:
            issues.append(PreparationIssue(frame_id=None, reason=str(error)))

    train_id_set = set(train_ids)
    validation_id_set = set(validation_ids)
    frames = [
        frame.to_manifest_entry("train" if frame.frame_id in train_id_set else "validation")
        for frame in sorted(prepared_frames, key=lambda item: item.frame_id)
        if frame.frame_id in train_id_set or frame.frame_id in validation_id_set
    ]
    manifest = {
        "format": "kitti-preparation-manifest-v1",
        "paths_are_relative_to": "dataset_root",
        "preprocessing_mode": preprocessing_mode,
        "output_image_size": {"width": output_width, "height": output_height},
        "split": {
            "validation_ratio": validation_ratio,
            "seed": seed,
            "train_frame_ids": train_ids,
            "validation_frame_ids": validation_ids,
        },
        "frames": frames,
        "invalid_samples": [issue.to_dict() for issue in issues],
    }
    return PreparationResult(manifest=manifest, issues=tuple(issues))


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Save a preparation manifest as formatted JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def format_preparation_summary(result: PreparationResult) -> str:
    """Return a human-readable preparation summary for the CLI."""
    split = result.manifest["split"]
    lines = [
        f"KITTI preparation: {'PASSED' if result.is_valid else 'FAILED'}",
        f"Prepared frames: {len(result.manifest['frames'])}",
        f"Train frames: {len(split['train_frame_ids'])}",
        f"Validation frames: {len(split['validation_frame_ids'])}",
        f"Invalid samples: {len(result.issues)}",
    ]
    for issue in result.issues:
        prefix = f"{issue.frame_id}: " if issue.frame_id is not None else ""
        lines.append(f"- {prefix}{issue.reason}")
    return "\n".join(lines)
