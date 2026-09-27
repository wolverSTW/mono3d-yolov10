"""Non-destructive structural auditing for KITTI 3D Object Detection data.

This module intentionally validates dataset files only.  It does not resize
images, transform coordinates, construct training targets, or modify dataset
contents.  Those concerns belong to later experiments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


IMAGE_DIRECTORY_NAME = "image_2"
LABEL_DIRECTORY_NAME = "label_2"
CALIBRATION_DIRECTORY_NAME = "calib"

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
TEXT_EXTENSIONS = frozenset({".txt"})

# These are the annotation types documented for the KITTI object-detection
# labels.  They are not a model class mapping.
KITTI_OBJECT_TYPES = frozenset(
    {
        "Car",
        "Van",
        "Truck",
        "Pedestrian",
        "Person_sitting",
        "Cyclist",
        "Tram",
        "Misc",
        "DontCare",
    }
)

# P2 maps rectified camera-2 coordinates to image-2 pixels, which is the
# minimum calibration information needed by the later monocular pipeline.
DEFAULT_REQUIRED_CALIBRATION_MATRICES: dict[str, tuple[int, int]] = {"P2": (3, 4)}
KNOWN_CALIBRATION_MATRICES: dict[str, tuple[int, int]] = {
    "P0": (3, 4),
    "P1": (3, 4),
    "P2": (3, 4),
    "P3": (3, 4),
    "R0_rect": (3, 3),
    "Tr_velo_to_cam": (3, 4),
    "Tr_imu_to_velo": (3, 4),
}


@dataclass(frozen=True)
class FrameDiscovery:
    """Files discovered in one KITTI modality, grouped by frame identifier."""

    files_by_id: dict[str, tuple[Path, ...]]

    @property
    def frame_ids(self) -> set[str]:
        """Return every discovered frame identifier."""
        return set(self.files_by_id)

    @property
    def duplicate_ids(self) -> dict[str, tuple[Path, ...]]:
        """Return identifiers represented by more than one file."""
        return {
            frame_id: paths
            for frame_id, paths in self.files_by_id.items()
            if len(paths) > 1
        }

    @property
    def file_count(self) -> int:
        """Return the number of files, including duplicate identifiers."""
        return sum(len(paths) for paths in self.files_by_id.values())


def discover_frame_files(directory: str | Path, extensions: frozenset[str]) -> FrameDiscovery:
    """Discover files by stem without assuming matching directory order.

    Multiple files with the same stem are preserved so that callers can report
    duplicates instead of silently choosing one.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return FrameDiscovery(files_by_id={})

    files_by_id: defaultdict[str, list[Path]] = defaultdict(list)
    for file_path in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
        if file_path.is_file() and file_path.suffix.casefold() in extensions:
            files_by_id[file_path.stem].append(file_path)

    return FrameDiscovery(
        files_by_id={
            frame_id: tuple(sorted(paths, key=lambda path: path.name.casefold()))
            for frame_id, paths in sorted(files_by_id.items())
        }
    )


def _relative_path(file_path: Path, dataset_root: Path) -> str:
    """Return a report-friendly path relative to the audited dataset root."""
    return file_path.relative_to(dataset_root).as_posix()


def _issue(frame_id: str, file_path: Path, dataset_root: Path, reason: str, **extra: Any) -> dict[str, Any]:
    """Create a serialisable validation issue record."""
    issue: dict[str, Any] = {
        "frame_id": frame_id,
        "file": _relative_path(file_path, dataset_root),
        "reason": reason,
    }
    issue.update(extra)
    return issue


def _parse_finite_float(value: str, field_name: str) -> str | None:
    """Return an explanatory error when *value* is not a finite float."""
    try:
        parsed_value = float(value)
    except ValueError:
        return f"{field_name} is not numeric: {value!r}"
    if not math.isfinite(parsed_value):
        return f"{field_name} is not finite: {value!r}"
    return None


def _validate_label_file(
    file_path: Path,
    frame_id: str,
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    """Validate one label file and return issues, observed, and valid type counts."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        return (
            [_issue(frame_id, file_path, dataset_root, f"could not read label file: {error}")],
            Counter(),
            Counter(),
        )

    issues: list[dict[str, Any]] = []
    observed_class_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    numeric_field_names = (
        "truncated",
        "occluded",
        "alpha",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "height",
        "width",
        "length",
        "location_x",
        "location_y",
        "location_z",
        "rotation_y",
    )

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        fields = line.split()
        if len(fields) != 15:
            issues.append(
                _issue(
                    frame_id,
                    file_path,
                    dataset_root,
                    f"expected 15 fields, found {len(fields)}",
                    line=line_number,
                )
            )
            continue

        reasons: list[str] = []
        object_type = fields[0]
        observed_class_counts[object_type] += 1
        if object_type not in KITTI_OBJECT_TYPES:
            reasons.append(f"unknown KITTI object type: {object_type!r}")

        for field_name, value in zip(numeric_field_names, fields[1:], strict=True):
            numeric_error = _parse_finite_float(value, field_name)
            if numeric_error is not None:
                reasons.append(numeric_error)

        # KITTI encodes occlusion as an integer category.  -1 is allowed for
        # unknown values, which commonly appears in DontCare annotations.
        if not reasons:
            try:
                occlusion = int(fields[2])
            except ValueError:
                reasons.append(f"occluded must be an integer category: {fields[2]!r}")
            else:
                if occlusion not in {-1, 0, 1, 2, 3}:
                    reasons.append(f"occluded is outside KITTI categories: {occlusion}")

        if reasons:
            issues.append(
                _issue(
                    frame_id,
                    file_path,
                    dataset_root,
                    "; ".join(reasons),
                    line=line_number,
                )
            )
            continue

        class_counts[object_type] += 1

    return issues, observed_class_counts, class_counts


def _normalise_required_matrices(
    required_matrices: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, int]]:
    """Validate and normalise required calibration matrix shapes."""
    if required_matrices is None:
        return dict(DEFAULT_REQUIRED_CALIBRATION_MATRICES)

    normalised: dict[str, tuple[int, int]] = {}
    for key, shape in required_matrices.items():
        if len(shape) != 2 or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape):
            raise ValueError(
                f"Calibration matrix {key!r} must have a two-dimensional positive integer shape."
            )
        normalised[str(key)] = (shape[0], shape[1])
    return normalised


def _validate_calibration_file(
    file_path: Path,
    frame_id: str,
    dataset_root: Path,
    required_matrices: Mapping[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Validate a KITTI calibration file without applying transformations."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        return [_issue(frame_id, file_path, dataset_root, f"could not read calibration file: {error}")]

    issues: list[dict[str, Any]] = []
    entries: dict[str, list[str]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if ":" not in line:
            issues.append(
                _issue(
                    frame_id,
                    file_path,
                    dataset_root,
                    "expected a 'key: values' calibration entry",
                    line=line_number,
                )
            )
            continue

        key, raw_values = line.split(":", maxsplit=1)
        key = key.strip()
        values = raw_values.split()
        if not key:
            issues.append(
                _issue(frame_id, file_path, dataset_root, "calibration key is empty", line=line_number)
            )
            continue
        if key in entries:
            issues.append(
                _issue(
                    frame_id,
                    file_path,
                    dataset_root,
                    f"duplicate calibration entry: {key}",
                    line=line_number,
                )
            )
            continue

        entries[key] = values
        for value in values:
            numeric_error = _parse_finite_float(value, f"{key} value")
            if numeric_error is not None:
                issues.append(_issue(frame_id, file_path, dataset_root, numeric_error, line=line_number))
                break

    expected_shapes = dict(KNOWN_CALIBRATION_MATRICES)
    expected_shapes.update(required_matrices)
    for key, shape in expected_shapes.items():
        if key not in entries:
            if key in required_matrices:
                issues.append(
                    _issue(
                        frame_id,
                        file_path,
                        dataset_root,
                        f"required calibration matrix is missing: {key}",
                    )
                )
            continue

        expected_value_count = shape[0] * shape[1]
        actual_value_count = len(entries[key])
        if actual_value_count != expected_value_count:
            issues.append(
                _issue(
                    frame_id,
                    file_path,
                    dataset_root,
                    f"{key} must contain {expected_value_count} values for shape {shape}, "
                    f"found {actual_value_count}",
                )
            )

    return issues


def _validate_image_file(
    file_path: Path,
    frame_id: str,
    dataset_root: Path,
    expected_channels: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Read image metadata and return unreadable/channel issues when present."""
    try:
        with Image.open(file_path) as image:
            image.verify()
        with Image.open(file_path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            channels = len(image.getbands())
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return None, _issue(frame_id, file_path, dataset_root, f"unreadable image: {error}"), None

    image_details = {
        "frame_id": frame_id,
        "file": _relative_path(file_path, dataset_root),
        "width": width,
        "height": height,
        "mode": mode,
        "channels": channels,
    }
    channel_issue: dict[str, Any] | None = None
    if expected_channels is not None and channels != expected_channels:
        channel_issue = _issue(
            frame_id,
            file_path,
            dataset_root,
            f"expected {expected_channels} image channels, found {channels}",
        )
    return image_details, None, channel_issue


def _serialise_duplicates(discovery: FrameDiscovery, dataset_root: Path) -> dict[str, list[str]]:
    """Convert duplicate paths to JSON-safe paths relative to the dataset root."""
    return {
        frame_id: [_relative_path(file_path, dataset_root) for file_path in paths]
        for frame_id, paths in discovery.duplicate_ids.items()
    }


def audit_kitti_dataset(
    dataset_root: str | Path,
    *,
    configured_classes: Sequence[str] | None = None,
    required_calibration_matrices: Mapping[str, Sequence[int]] | None = None,
    expected_image_channels: int | None = 3,
) -> dict[str, Any]:
    """Audit a KITTI dataset directory and return a JSON-serialisable report.

    The function does not write files and continues after malformed inputs where
    possible, allowing callers to receive all discovered diagnostics at once.
    """
    dataset_root = Path(dataset_root)
    required_matrices = _normalise_required_matrices(required_calibration_matrices)
    directories = {
        "dataset_root": dataset_root.is_dir(),
        IMAGE_DIRECTORY_NAME: (dataset_root / IMAGE_DIRECTORY_NAME).is_dir(),
        LABEL_DIRECTORY_NAME: (dataset_root / LABEL_DIRECTORY_NAME).is_dir(),
        CALIBRATION_DIRECTORY_NAME: (dataset_root / CALIBRATION_DIRECTORY_NAME).is_dir(),
    }

    images = discover_frame_files(dataset_root / IMAGE_DIRECTORY_NAME, IMAGE_EXTENSIONS)
    labels = discover_frame_files(dataset_root / LABEL_DIRECTORY_NAME, TEXT_EXTENSIONS)
    calibrations = discover_frame_files(dataset_root / CALIBRATION_DIRECTORY_NAME, TEXT_EXTENSIONS)

    image_ids = images.frame_ids
    label_ids = labels.frame_ids
    calibration_ids = calibrations.frame_ids
    matched_frame_ids = image_ids & label_ids & calibration_ids

    unreadable_images: list[dict[str, Any]] = []
    unexpected_image_channels: list[dict[str, Any]] = []
    image_details: list[dict[str, Any]] = []
    for frame_id, paths in images.files_by_id.items():
        for file_path in paths:
            details, unreadable_issue, channel_issue = _validate_image_file(
                file_path, frame_id, dataset_root, expected_image_channels
            )
            if details is not None:
                image_details.append(details)
            if unreadable_issue is not None:
                unreadable_images.append(unreadable_issue)
            if channel_issue is not None:
                unexpected_image_channels.append(channel_issue)

    malformed_labels: list[dict[str, Any]] = []
    observed_class_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for frame_id, paths in labels.files_by_id.items():
        for file_path in paths:
            issues, file_observed_class_counts, file_class_counts = _validate_label_file(
                file_path, frame_id, dataset_root
            )
            malformed_labels.extend(issues)
            observed_class_counts.update(file_observed_class_counts)
            class_counts.update(file_class_counts)

    malformed_calibrations: list[dict[str, Any]] = []
    for frame_id, paths in calibrations.files_by_id.items():
        for file_path in paths:
            malformed_calibrations.extend(
                _validate_calibration_file(file_path, frame_id, dataset_root, required_matrices)
            )

    configured_class_list = list(configured_classes) if configured_classes is not None else []
    observed_classes = sorted(observed_class_counts)
    observed_detection_classes = [name for name in observed_classes if name != "DontCare"]
    unconfigured_observed_classes = (
        sorted(set(observed_detection_classes) - set(configured_class_list))
        if configured_classes is not None
        else []
    )

    matching = {
        "matched_frame_ids": sorted(matched_frame_ids),
        "image_ids_missing_labels": sorted(image_ids - label_ids),
        "image_ids_missing_calibration": sorted(image_ids - calibration_ids),
        "label_ids_missing_images": sorted(label_ids - image_ids),
        "label_ids_missing_calibration": sorted(label_ids - calibration_ids),
        "calibration_ids_missing_images": sorted(calibration_ids - image_ids),
        "calibration_ids_missing_labels": sorted(calibration_ids - label_ids),
    }
    missing_images = sorted((label_ids | calibration_ids) - image_ids)
    missing_labels = sorted((image_ids | calibration_ids) - label_ids)
    missing_calibration = sorted((image_ids | label_ids) - calibration_ids)
    duplicate_ids = {
        "images": _serialise_duplicates(images, dataset_root),
        "labels": _serialise_duplicates(labels, dataset_root),
        "calibration": _serialise_duplicates(calibrations, dataset_root),
    }

    missing_directories = [name for name, is_present in directories.items() if not is_present]
    empty_modalities = [
        name
        for name, count in {
            "images": images.file_count,
            "labels": labels.file_count,
            "calibration": calibrations.file_count,
        }.items()
        if count == 0
    ]
    failure_reasons: list[str] = []
    if missing_directories:
        failure_reasons.append("required dataset directories are missing")
    if empty_modalities:
        failure_reasons.append("one or more required modalities contain no recognised files")
    if any(matching[key] for key in matching if key != "matched_frame_ids"):
        failure_reasons.append("frame identifiers are not matched across all required modalities")
    if any(duplicate_ids.values()):
        failure_reasons.append("duplicate frame identifiers were found")
    if unreadable_images:
        failure_reasons.append("one or more images are unreadable")
    if unexpected_image_channels:
        failure_reasons.append("one or more images have unexpected channel counts")
    if malformed_labels:
        failure_reasons.append("one or more label files are malformed")
    if malformed_calibrations:
        failure_reasons.append("one or more calibration files are malformed")

    return {
        "dataset_path": str(dataset_root),
        "overall_status": "passed" if not failure_reasons else "failed",
        "is_valid": not failure_reasons,
        "failure_reasons": failure_reasons,
        "directories": directories,
        "file_counts": {
            "images": images.file_count,
            "labels": labels.file_count,
            "calibration": calibrations.file_count,
        },
        "frame_ids": {
            "images": sorted(image_ids),
            "labels": sorted(label_ids),
            "calibration": sorted(calibration_ids),
        },
        "frame_matching": matching,
        "missing": {
            "images": missing_images,
            "labels": missing_labels,
            "calibration": missing_calibration,
        },
        "duplicate_ids": duplicate_ids,
        "image_details": sorted(image_details, key=lambda item: (item["frame_id"], item["file"])),
        "unreadable_images": unreadable_images,
        "unexpected_image_channels": unexpected_image_channels,
        "malformed_labels": malformed_labels,
        "malformed_calibrations": malformed_calibrations,
        "class_distribution": {
            "observed_classes": observed_classes,
            "class_counts": dict(sorted(observed_class_counts.items())),
            "valid_class_counts": dict(sorted(class_counts.items())),
            "dontcare_count": observed_class_counts["DontCare"],
            "class_mapping_configured": configured_classes is not None,
            "configured_classes": configured_class_list,
            "observed_detection_classes_not_configured": unconfigured_observed_classes,
        },
        "required_calibration_matrices": {
            key: list(shape) for key, shape in sorted(required_matrices.items())
        },
    }


def write_audit_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write an audit report as formatted JSON and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def format_audit_summary(report: Mapping[str, Any]) -> str:
    """Build a concise human-readable summary for CLI output."""
    file_counts = report["file_counts"]
    frame_matching = report["frame_matching"]
    lines = [
        f"KITTI audit: {str(report['overall_status']).upper()}",
        f"Dataset: {report['dataset_path']}",
        "Files: "
        f"{file_counts['images']} images, {file_counts['labels']} labels, "
        f"{file_counts['calibration']} calibration files",
        f"Matched frames: {len(frame_matching['matched_frame_ids'])}",
        f"Unreadable images: {len(report['unreadable_images'])}",
        f"Malformed labels: {len(report['malformed_labels'])}",
        f"Malformed calibrations: {len(report['malformed_calibrations'])}",
        f"DontCare annotations: {report['class_distribution']['dontcare_count']}",
    ]
    if report["failure_reasons"]:
        lines.append("Failure reasons:")
        lines.extend(f"- {reason}" for reason in report["failure_reasons"])
    else:
        lines.append("All structural and content checks passed.")
    return "\n".join(lines)
