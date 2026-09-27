"""Command-line entry point for the EXP-001 KITTI dataset audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data.kitti_audit import audit_kitti_dataset, format_audit_summary, write_audit_report


def _project_relative_path(value: str | Path) -> Path:
    """Resolve a configured project-relative path without machine-specific paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _argument_path(value: str | Path) -> Path:
    """Resolve a path supplied directly by the user relative to the current directory."""
    return Path(value).resolve()


def _load_audit_settings(
    config: dict[str, Any],
) -> tuple[Path, Path, list[str] | None, dict[str, list[int]] | None]:
    """Extract the small subset of configuration used by the audit command."""
    dataset_config = config.get("dataset", {})
    if not isinstance(dataset_config, dict):
        raise ValueError("dataset configuration section must be a mapping")
    audit_config = dataset_config.get("audit", {})
    if not isinstance(audit_config, dict):
        raise ValueError("dataset.audit configuration section must be a mapping")

    dataset_root = _project_relative_path(dataset_config["root"])
    report_path = _project_relative_path(audit_config["report_path"])
    configured_classes = dataset_config.get("classes")
    if configured_classes is not None and not isinstance(configured_classes, list):
        raise ValueError("dataset.classes must be a list when configured")

    required_matrices = audit_config.get("required_calibration_matrices")
    if required_matrices is not None and not isinstance(required_matrices, dict):
        raise ValueError("dataset.audit.required_calibration_matrices must be a mapping")
    return dataset_root, report_path, configured_classes, required_matrices


def parse_arguments() -> argparse.Namespace:
    """Parse the deliberately small audit CLI surface."""
    parser = argparse.ArgumentParser(description="Audit a KITTI 3D Object Detection dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override the configured KITTI dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the configured JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the audit, save its report, and return a meaningful process code."""
    arguments = parse_arguments()
    config = load_config(arguments.config)
    dataset_root, report_path, configured_classes, required_matrices = _load_audit_settings(config)

    if arguments.data_root is not None:
        dataset_root = _argument_path(arguments.data_root)
    if arguments.output is not None:
        report_path = _argument_path(arguments.output)

    report = audit_kitti_dataset(
        dataset_root,
        configured_classes=configured_classes,
        required_calibration_matrices=required_matrices,
        expected_image_channels=config.get("input", {}).get("channels", 3),
    )
    saved_path = write_audit_report(report, report_path)
    print(format_audit_summary(report))
    print(f"JSON report: {saved_path}")
    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
