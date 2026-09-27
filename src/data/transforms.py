"""Image-plane preprocessing transformations that preserve camera geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from PIL import Image

from src.data.kitti_parser import BoundingBox2D


PreprocessingMode = Literal["resize", "letterbox"]


@dataclass(frozen=True)
class ImageTransform:
    """Explicit original-to-output image transformation metadata.

    ``pad_x`` and ``pad_y`` are left and top padding.  The full matrix maps
    homogeneous original image coordinates to output coordinates and is used to
    transform a complete 3 × 4 projection matrix as ``P' = A @ P``.
    """

    mode: PreprocessingMode
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    resized_width: int
    resized_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    pad_right: int
    pad_bottom: int

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3 × 3 scale-and-translation image-plane matrix A."""
        return np.asarray(
            [
                [self.scale_x, 0.0, float(self.pad_x)],
                [0.0, self.scale_y, float(self.pad_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        """Return JSON-safe transform metadata for manifests."""
        return asdict(self)


def _validate_sizes(original_width: int, original_height: int, output_width: int, output_height: int) -> None:
    """Reject zero or negative input/output dimensions."""
    if min(original_width, original_height, output_width, output_height) <= 0:
        raise ValueError("original and output image dimensions must be positive")


def make_resize_transform(
    original_width: int,
    original_height: int,
    output_width: int,
    output_height: int,
) -> ImageTransform:
    """Create an anisotropic direct-resize transformation."""
    _validate_sizes(original_width, original_height, output_width, output_height)
    return ImageTransform(
        mode="resize",
        original_width=original_width,
        original_height=original_height,
        output_width=output_width,
        output_height=output_height,
        resized_width=output_width,
        resized_height=output_height,
        scale_x=output_width / original_width,
        scale_y=output_height / original_height,
        pad_x=0,
        pad_y=0,
        pad_right=0,
        pad_bottom=0,
    )


def make_letterbox_transform(
    original_width: int,
    original_height: int,
    output_width: int,
    output_height: int,
) -> ImageTransform:
    """Create an aspect-ratio-preserving resize plus centred padding transform."""
    _validate_sizes(original_width, original_height, output_width, output_height)
    nominal_scale = min(output_width / original_width, output_height / original_height)
    resized_width = max(1, round(original_width * nominal_scale))
    resized_height = max(1, round(original_height * nominal_scale))
    total_pad_x = output_width - resized_width
    total_pad_y = output_height - resized_height
    pad_x = total_pad_x // 2
    pad_y = total_pad_y // 2
    return ImageTransform(
        mode="letterbox",
        original_width=original_width,
        original_height=original_height,
        output_width=output_width,
        output_height=output_height,
        resized_width=resized_width,
        resized_height=resized_height,
        scale_x=resized_width / original_width,
        scale_y=resized_height / original_height,
        pad_x=pad_x,
        pad_y=pad_y,
        pad_right=total_pad_x - pad_x,
        pad_bottom=total_pad_y - pad_y,
    )


def make_image_transform(
    mode: PreprocessingMode,
    original_width: int,
    original_height: int,
    output_width: int,
    output_height: int,
) -> ImageTransform:
    """Create the configured preprocessing transformation."""
    if mode == "resize":
        return make_resize_transform(original_width, original_height, output_width, output_height)
    if mode == "letterbox":
        return make_letterbox_transform(original_width, original_height, output_width, output_height)
    raise ValueError(f"unsupported preprocessing mode: {mode!r}")


def transform_bbox(bbox: BoundingBox2D, transform: ImageTransform) -> BoundingBox2D:
    """Map a 2D box from original image coordinates to output coordinates."""
    return BoundingBox2D(
        x1=transform.scale_x * bbox.x1 + transform.pad_x,
        y1=transform.scale_y * bbox.y1 + transform.pad_y,
        x2=transform.scale_x * bbox.x2 + transform.pad_x,
        y2=transform.scale_y * bbox.y2 + transform.pad_y,
    )


def inverse_transform_bbox(bbox: BoundingBox2D, transform: ImageTransform) -> BoundingBox2D:
    """Map an output-coordinate box back to original image coordinates."""
    return BoundingBox2D(
        x1=(bbox.x1 - transform.pad_x) / transform.scale_x,
        y1=(bbox.y1 - transform.pad_y) / transform.scale_y,
        x2=(bbox.x2 - transform.pad_x) / transform.scale_x,
        y2=(bbox.y2 - transform.pad_y) / transform.scale_y,
    )


def transform_projection_matrix(projection_matrix: np.ndarray, transform: ImageTransform) -> np.ndarray:
    """Transform a 3 × 4 projection matrix using ``P' = A @ P``."""
    projection_matrix = np.asarray(projection_matrix, dtype=np.float64)
    if projection_matrix.shape != (3, 4):
        raise ValueError(f"projection matrix must have shape (3, 4), found {projection_matrix.shape}")
    return transform.matrix @ projection_matrix


def preprocess_image(image: Image.Image, transform: ImageTransform, *, padding_value: int = 114) -> Image.Image:
    """Apply a declared transform in memory without writing dataset files."""
    if image.size != (transform.original_width, transform.original_height):
        raise ValueError("image dimensions do not match the transformation metadata")
    resized = image.resize((transform.resized_width, transform.resized_height), Image.Resampling.BILINEAR)
    if transform.mode == "resize":
        return resized

    if image.mode == "RGBA":
        padding_color: int | tuple[int, ...] = (padding_value, padding_value, padding_value, 255)
    elif len(image.getbands()) > 1:
        padding_color = tuple(padding_value for _ in image.getbands())
    else:
        padding_color = padding_value
    canvas = Image.new(image.mode, (transform.output_width, transform.output_height), color=padding_color)
    canvas.paste(resized, (transform.pad_x, transform.pad_y))
    return canvas
