"""Command-line entry point for the EXP-002 KITTI preparation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data.converter import format_preparation_summary, prepare_kitti_dataset, write_manifest


def _project_relative_path(value: str | Path) -> Path:
    """Resolve config paths relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _argument_path(value: str | Path) -> Path:
    """Resolve directly supplied paths relative to the current directory."""
    return Path(value).resolve()


def parse_arguments() -> argparse.Namespace:
    """Parse configuration-driven preparation options."""
    parser = argparse.ArgumentParser(description="Prepare a validated KITTI manifest for EXP-002.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, help="Override the configured KITTI root.")
    parser.add_argument("--output", type=Path, help="Override the configured manifest path.")
    parser.add_argument("--validation-ratio", type=float, help="Override the configured validation ratio.")
    parser.add_argument("--seed", type=int, help="Override project.seed for this split.")
    parser.add_argument(
        "--preprocessing-mode",
        choices=("resize", "letterbox"),
        help="Override the configured preprocessing mode.",
    )
    return parser.parse_args()


def _settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract only EXP-002 settings from the existing YAML convention."""
    dataset = config.get("dataset")
    input_config = config.get("input")
    preparation = dataset.get("preparation") if isinstance(dataset, dict) else None
    preprocessing = config.get("preprocessing")
    if not isinstance(dataset, dict) or not isinstance(input_config, dict):
        raise ValueError("dataset and input configuration sections must be mappings")
    if not isinstance(preparation, dict) or not isinstance(preprocessing, dict):
        raise ValueError("dataset.preparation and preprocessing configuration sections are required")
    return {
        "data_root": _project_relative_path(dataset["root"]),
        "output": _project_relative_path(preparation["manifest_path"]),
        "validation_ratio": preparation["validation_ratio"],
        "seed": config["project"]["seed"],
        "preprocessing_mode": preprocessing["mode"],
        "padding_value": preprocessing["padding_value"],
        "output_width": input_config["image_width"],
        "output_height": input_config["image_height"],
        "expected_image_channels": input_config.get("channels", 3),
    }


def main() -> int:
    """Prepare a manifest, print diagnostics, and signal invalid input clearly."""
    arguments = parse_arguments()
    settings = _settings_from_config(load_config(arguments.config))
    if arguments.data_root is not None:
        settings["data_root"] = _argument_path(arguments.data_root)
    if arguments.output is not None:
        settings["output"] = _argument_path(arguments.output)
    if arguments.validation_ratio is not None:
        settings["validation_ratio"] = arguments.validation_ratio
    if arguments.seed is not None:
        settings["seed"] = arguments.seed
    if arguments.preprocessing_mode is not None:
        settings["preprocessing_mode"] = arguments.preprocessing_mode

    result = prepare_kitti_dataset(
        settings["data_root"],
        output_width=settings["output_width"],
        output_height=settings["output_height"],
        preprocessing_mode=settings["preprocessing_mode"],
        validation_ratio=settings["validation_ratio"],
        seed=settings["seed"],
        expected_image_channels=settings["expected_image_channels"],
        padding_value=settings["padding_value"],
    )
    manifest_path = write_manifest(result.manifest, settings["output"])
    print(format_preparation_summary(result))
    print(f"Manifest: {manifest_path}")
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
