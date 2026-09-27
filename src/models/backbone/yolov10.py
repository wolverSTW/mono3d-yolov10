"""YOLOv10 backbone adapter for monocular 3D object detection.

This module provides a clean interface to the YOLOv10 feature extractor
from the ultralytics package. It loads the model architecture from
a YAML configuration (avoiding pretrained weight downloads during
development) and exposes intermediate feature maps for custom 3D
prediction heads.

The adapter uses forward hooks to capture the feature pyramid outputs
(P3, P4, P5) from the backbone+neck, letting the ultralytics model's
forward method handle all skip connections properly.

Configuration:
- variant: one of "yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"
- pretrained: whether to load pretrained weights (requires download)
- input_channels: number of input channels (default 3 for RGB)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
except ImportError as e:
    raise ImportError(
        "ultralytics package is required for YOLOv10 integration. "
        "Install with: pip install ultralytics"
    ) from e


Yolov10Variant = Literal["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"]


@dataclass(frozen=True)
class Yolov10Config:
    """Configuration for YOLOv10 backbone."""
    variant: Yolov10Variant = "yolov10n"
    pretrained: bool = False
    input_channels: int = 3


class Yolov10Backbone(nn.Module):
    """YOLOv10 backbone feature extractor.

    This module wraps the ultralytics YOLOv10 DetectionModel and
    exposes the backbone + neck feature pyramid (P3, P4, P5)
    using forward hooks on the appropriate layers.

    The forward pass returns a tuple of feature maps:
        (P3, P4, P5) where:
        - P3: stride 8, channels depend on variant (e.g., 256 for yolov10n)
        - P4: stride 16, channels depend on variant (e.g., 512 for yolov10n)
        - P5: stride 32, channels depend on variant (e.g., 1024 for yolov10n)

    These feature maps are suitable for connecting custom 3D prediction heads.
    """

    # Expected output channels for each variant at P3, P4, P5
    # Base channels (256, 512, 1024) scaled by width multiplier from YAML scales:
    # n: width=0.25, s: width=0.5, m: width=0.75, l: width=1.0, x: width=1.25
    _VARIANT_CHANNELS = {
        "yolov10n": (64, 128, 256),
        "yolov10s": (128, 256, 512),
        "yolov10m": (192, 384, 768),
        "yolov10l": (256, 512, 1024),
        "yolov10x": (320, 640, 1280),
    }

    # Layer indices (in model.model Sequential) that produce P3, P4, P5
    # Based on YOLOv10n architecture:
    # P3: C2f after second upsample+concat (index 16)
    # P4: C2f after third upsample+concat (index 19)
    # P5: C2fCIB after fourth concat (index 22)
    _FEATURE_LAYER_INDICES = (16, 19, 22)

    def __init__(
        self,
        variant: Yolov10Variant = "yolov10n",
        pretrained: bool = False,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.pretrained = pretrained
        self.input_channels = input_channels

        # Get the YAML config path for the variant
        yaml_path = self._get_yaml_path(variant)

        # Load model from YAML (architecture only, no pretrained weights unless requested)
        if pretrained:
            # Load from pretrained .pt file
            self._model = YOLO(f"{variant}.pt").model
        else:
            # Build from YAML config
            self._model = DetectionModel(yaml_path, ch=input_channels, nc=80)

        # Register forward hooks on feature layers
        self._feature_outputs: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_feature_hooks()

        # Store feature map info
        self._feature_channels = self._VARIANT_CHANNELS[variant]
        self._strides = (8, 16, 32)

    def _get_yaml_path(self, variant: str) -> Path:
        """Get the YAML config path for a YOLOv10 variant."""
        import ultralytics
        base = Path(ultralytics.__file__).parent / "cfg" / "models" / "v10"
        yaml_path = base / f"{variant}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"YOLOv10 config not found: {yaml_path}")
        return yaml_path

    def _register_feature_hooks(self) -> None:
        """Register forward hooks on layers that produce P3, P4, P5."""
        self._feature_outputs = []
        self._hooks = []

        # Clear any existing hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        model_seq = self._model.model
        for idx in self._FEATURE_LAYER_INDICES:
            if idx < len(model_seq):
                layer = model_seq[idx]
                hook = layer.register_forward_hook(self._make_hook())
                self._hooks.append(hook)
            else:
                raise RuntimeError(
                    f"Feature layer index {idx} out of range for model with {len(model_seq)} layers"
                )

    def _make_hook(self):
        """Create a hook function that captures the layer output."""
        def hook(module, input, output):
            self._feature_outputs.append(output)
        return hook

    def _clear_feature_outputs(self) -> None:
        """Clear captured feature outputs."""
        self._feature_outputs.clear()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning feature pyramid (P3, P4, P5).

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Tuple of (P3, P4, P5) feature maps.
            P3: [B, C3, H/8, W/8]
            P4: [B, C4, H/16, W/16]
            P5: [B, C5, H/32, W/32]
        """
        self._clear_feature_outputs()

        # Run the full model forward (handles all skip connections internally)
        # The detection head will run but we ignore its output
        _ = self._model(x)

        # Hooks have populated self._feature_outputs in layer order
        if len(self._feature_outputs) != 3:
            raise RuntimeError(
                f"Expected 3 feature maps from hooks, got {len(self._feature_outputs)}"
            )

        # Return as tuple (P3, P4, P5)
        return tuple(self._feature_outputs)

    @property
    def feature_channels(self) -> tuple[int, int, int]:
        """Return (C3, C4, C5) output channels for P3, P4, P5."""
        return self._feature_channels

    @property
    def strides(self) -> tuple[int, int, int]:
        """Return strides for P3, P4, P5."""
        return self._strides

    @property
    def num_parameters(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __del__(self):
        """Clean up hooks on deletion."""
        for hook in self._hooks:
            hook.remove()


def create_yolov10_backbone(config: Yolov10Config | None = None, **kwargs) -> Yolov10Backbone:
    """Factory function to create YOLOv10 backbone from config."""
    if config is None:
        config = Yolov10Config()
    # Allow kwargs to override config
    for key, value in kwargs.items():
        if hasattr(config, key):
            object.__setattr__(config, key, value)
    return Yolov10Backbone(
        variant=config.variant,
        pretrained=config.pretrained,
        input_channels=config.input_channels,
    )