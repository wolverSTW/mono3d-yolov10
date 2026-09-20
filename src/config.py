"""Configuration loading utilities for the Mono3D YOLOv10 project."""

from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the project configuration from a YAML file."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config
