"""Tests for YOLOv10 backbone integration."""

from __future__ import annotations

import torch

from src.models.backbone.yolov10 import Yolov10Backbone, Yolov10Config, create_yolov10_backbone
from src.models.baseline import Baseline3DDetector, BaselineConfig, create_baseline_model


def test_yolov10_config_defaults() -> None:
    """Test YOLOv10 config defaults."""
    config = Yolov10Config()
    assert config.variant == "yolov10n"
    assert config.pretrained is False
    assert config.input_channels == 3


def test_yolov10_config_custom() -> None:
    """Test YOLOv10 config with custom values."""
    config = Yolov10Config(variant="yolov10s", pretrained=True, input_channels=3)
    assert config.variant == "yolov10s"
    assert config.pretrained is True
    assert config.input_channels == 3


def test_yolov10_backbone_construction() -> None:
    """Test YOLOv10 backbone can be constructed from YAML."""
    backbone = Yolov10Backbone(variant="yolov10n", pretrained=False, input_channels=3)
    assert isinstance(backbone, Yolov10Backbone)
    assert backbone.variant == "yolov10n"


def test_yolov10_backbone_all_variants() -> None:
    """Test all YOLOv10 variants can be constructed."""
    variants = ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"]
    for variant in variants:
        backbone = Yolov10Backbone(variant=variant, pretrained=False, input_channels=3)
        assert backbone.variant == variant
        assert backbone.feature_channels is not None
        assert len(backbone.feature_channels) == 3


def test_yolov10_backbone_feature_channels() -> None:
    """Test feature channels are correct for each variant."""
    # yolov10n should have (64, 128, 256) channels at P3, P4, P5 (width=0.25)
    backbone = Yolov10Backbone(variant="yolov10n", pretrained=False, input_channels=3)
    assert backbone.feature_channels == (64, 128, 256)
    assert backbone.strides == (8, 16, 32)


def test_yolov10_backbone_forward_pass() -> None:
    """Test forward pass produces expected feature maps."""
    backbone = Yolov10Backbone(variant="yolov10n", pretrained=False, input_channels=3)
    backbone.eval()

    # Input: [B, 3, 640, 640]
    x = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        p3, p4, p5 = backbone(x)

    # Check output shapes (yolov10n: width=0.25 -> P3=64, P4=128, P5=256)
    assert p3.shape == (2, 64, 80, 80)   # 640/8 = 80
    assert p4.shape == (2, 128, 40, 40)  # 640/16 = 40
    assert p5.shape == (2, 256, 20, 20)  # 640/32 = 20

    # Check no NaN/Inf
    assert torch.isfinite(p3).all()
    assert torch.isfinite(p4).all()
    assert torch.isfinite(p5).all()


def test_yolov10_backbone_batch_sizes() -> None:
    """Test forward pass with different batch sizes."""
    backbone = Yolov10Backbone(variant="yolov10n", pretrained=False, input_channels=3)
    backbone.eval()

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 3, 320, 320)
        with torch.no_grad():
            p3, p4, p5 = backbone(x)
        assert p3.shape[0] == batch_size
        assert p4.shape[0] == batch_size
        assert p5.shape[0] == batch_size


def test_yolov10_backbone_input_resolutions() -> None:
    """Test forward pass with different input resolutions."""
    backbone = Yolov10Backbone(variant="yolov10n", pretrained=False, input_channels=3)
    backbone.eval()

    resolutions = [(320, 320), (416, 416), (512, 512), (640, 640)]
    for h, w in resolutions:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            p3, p4, p5 = backbone(x)
        assert p3.shape[2:] == (h // 8, w // 8)
        assert p4.shape[2:] == (h // 16, w // 16)
        assert p5.shape[2:] == (h // 32, w // 32)


def test_yolov10_backbone_parameter_count() -> None:
    """Test parameter count is reasonable and non-zero."""
    backbone = Yolov10Backbone(variant="yolov10n", pretrained=False, input_channels=3)
    total_params = backbone.num_parameters
    trainable_params = backbone.num_trainable_parameters

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params

    # yolov10n should have ~2-3M parameters
    assert 1_000_000 < total_params < 10_000_000


def test_yolov10_config_factory() -> None:
    """Test factory function."""
    config = Yolov10Config(variant="yolov10s", pretrained=False, input_channels=3)
    backbone = create_yolov10_backbone(config)
    assert backbone.variant == "yolov10s"
    assert backbone.feature_channels == (128, 256, 512)


def test_yolov10_factory_kwargs_override() -> None:
    """Test factory kwargs override config."""
    config = Yolov10Config(variant="yolov10n")
    backbone = create_yolov10_backbone(config, variant="yolov10s")
    assert backbone.variant == "yolov10s"


def test_baseline_model_construction() -> None:
    """Test baseline model can be constructed."""
    model = Baseline3DDetector()
    assert isinstance(model, Baseline3DDetector)
    assert model.config.num_classes == 3


def test_baseline_model_forward() -> None:
    """Test baseline model forward pass."""
    model = Baseline3DDetector()
    model.eval()

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        outputs = model(x)

    # Check structured output
    assert hasattr(outputs, 'detection_2d')
    assert hasattr(outputs, 'dimensions_3d')
    assert hasattr(outputs, 'locations_3d')
    assert hasattr(outputs, 'orientation')
    assert hasattr(outputs, 'backbone_features')

    # Check backbone features
    bf = outputs.backbone_features
    assert "p3" in bf
    assert "p4" in bf
    assert "p5" in bf
    assert "backbone_channels" in bf
    assert "strides" in bf

    assert bf["p3"].shape == (2, 64, 80, 80)
    assert bf["p4"].shape == (2, 128, 40, 40)
    assert bf["p5"].shape == (2, 256, 20, 20)
    assert bf["backbone_channels"] == (64, 128, 256)
    assert bf["strides"] == (8, 16, 32)


def test_baseline_model_finite_outputs() -> None:
    """Test baseline model outputs are finite."""
    model = Baseline3DDetector()
    model.eval()

    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        outputs = model(x)

    # Check backbone features are finite
    for key in ["p3", "p4", "p5"]:
        assert torch.isfinite(outputs.backbone_features[key]).all(), f"NaN/Inf in {key}"


def test_baseline_config_custom() -> None:
    """Test baseline model with custom config."""
    config = BaselineConfig(
        backbone=Yolov10Config(variant="yolov10s"),
        num_classes=5
    )
    model = Baseline3DDetector(config)
    assert model.config.num_classes == 5
    assert model.backbone.variant == "yolov10s"


def test_baseline_factory() -> None:
    """Test baseline model factory function."""
    model = create_baseline_model()
    assert isinstance(model, Baseline3DDetector)


def test_baseline_parameter_count() -> None:
    """Test baseline model parameter count."""
    model = Baseline3DDetector()
    total = model.num_parameters
    trainable = model.num_trainable_parameters
    assert total > 0
    assert trainable > 0
    assert trainable <= total