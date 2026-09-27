"""Parsing and coordinate validation for KITTI 3D object annotations.

KITTI dimensions are retained in their on-disk order: height, width, length.
Locations remain in the rectified camera coordinate system: X is horizontal,
Y is vertical, and Z is forward depth.  ``alpha`` and ``rotation_y`` are
stored separately because they describe different orientation quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox2D:
    """An image-coordinate box in ``(x1, y1, x2, y2)`` order."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Dimensions3D:
    """KITTI dimensions in explicit ``(height, width, length)`` order."""

    height: float
    width: float
    length: float


@dataclass(frozen=True)
class CameraLocation3D:
    """Rectified camera location where X is horizontal, Y vertical, Z forward."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class KittiObject:
    """One parsed KITTI object annotation without target encoding."""

    class_name: str
    truncated: float
    occluded: int
    alpha: float
    bbox_2d: BoundingBox2D
    dimensions_3d: Dimensions3D
    location_3d: CameraLocation3D
    rotation_y: float

    @property
    def is_dontcare(self) -> bool:
        """Whether this record is a KITTI ``DontCare`` region."""
        return self.class_name == "DontCare"


@dataclass(frozen=True)
class CoordinateValidation:
    """Structured validation result for a parsed annotation."""

    valid: bool
    errors: tuple[str, ...]


class KittiAnnotationError(ValueError):
    """A malformed annotation with source context suitable for diagnostics."""

    def __init__(self, message: str, *, frame_id: str | None = None, line_number: int | None = None) -> None:
        context: list[str] = []
        if frame_id is not None:
            context.append(f"frame {frame_id}")
        if line_number is not None:
            context.append(f"line {line_number}")
        prefix = f"{' '.join(context)}: " if context else ""
        super().__init__(prefix + message)


def _parse_finite_float(value: str, field_name: str, *, frame_id: str | None, line_number: int | None) -> float:
    """Parse a finite label value or raise a context-rich annotation error."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise KittiAnnotationError(
            f"{field_name} is not numeric: {value!r}",
            frame_id=frame_id,
            line_number=line_number,
        ) from error
    if not math.isfinite(parsed):
        raise KittiAnnotationError(
            f"{field_name} is not finite: {value!r}",
            frame_id=frame_id,
            line_number=line_number,
        )
    return parsed


def validate_bbox(bbox: BoundingBox2D, image_size: tuple[int, int] | None = None) -> CoordinateValidation:
    """Validate ordering and, when supplied, image-coordinate boundaries."""
    errors: list[str] = []
    values = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
    if not all(math.isfinite(value) for value in values):
        errors.append("bounding box coordinates must be finite")
    if bbox.x1 >= bbox.x2:
        errors.append("bounding box requires x1 < x2")
    if bbox.y1 >= bbox.y2:
        errors.append("bounding box requires y1 < y2")

    if image_size is not None:
        image_width, image_height = image_size
        if image_width <= 0 or image_height <= 0:
            errors.append("image dimensions must be positive")
        elif bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > image_width or bbox.y2 > image_height:
            errors.append("bounding box lies outside image boundaries")
    return CoordinateValidation(valid=not errors, errors=tuple(errors))


def validate_dimensions(dimensions: Dimensions3D) -> CoordinateValidation:
    """Require finite, positive KITTI height, width, and length values."""
    values = (dimensions.height, dimensions.width, dimensions.length)
    errors: list[str] = []
    if not all(math.isfinite(value) for value in values):
        errors.append("3D dimensions must be finite")
    if dimensions.height <= 0:
        errors.append("height must be positive")
    if dimensions.width <= 0:
        errors.append("width must be positive")
    if dimensions.length <= 0:
        errors.append("length must be positive")
    return CoordinateValidation(valid=not errors, errors=tuple(errors))


def validate_location(location: CameraLocation3D) -> CoordinateValidation:
    """Require finite camera coordinates without imposing a positive-Z policy."""
    if all(math.isfinite(value) for value in (location.x, location.y, location.z)):
        return CoordinateValidation(valid=True, errors=())
    return CoordinateValidation(valid=False, errors=("camera location values must be finite",))


def validate_kitti_object(
    annotation: KittiObject,
    image_size: tuple[int, int] | None = None,
) -> CoordinateValidation:
    """Validate an annotation while respecting KITTI ``DontCare`` semantics.

    ``DontCare`` records do not represent physical 3D detection targets; KITTI
    commonly uses sentinel dimensions and location values for them.  They are
    therefore parsed and preserved but excluded from physical box/dimension
    constraints.  Regular objects must have valid 2D boxes and positive 3D
    dimensions.  Positive Z is deliberately not required at this stage.
    """
    errors: list[str] = []
    if not annotation.class_name:
        errors.append("class name must not be empty")
    if not math.isfinite(annotation.alpha):
        errors.append("alpha must be finite")
    if not math.isfinite(annotation.rotation_y):
        errors.append("rotation_y must be finite")
    if annotation.is_dontcare:
        return CoordinateValidation(valid=not errors, errors=tuple(errors))

    errors.extend(validate_bbox(annotation.bbox_2d, image_size).errors)
    errors.extend(validate_dimensions(annotation.dimensions_3d).errors)
    errors.extend(validate_location(annotation.location_3d).errors)
    return CoordinateValidation(valid=not errors, errors=tuple(errors))


def parse_annotation_line(
    line: str,
    *,
    frame_id: str | None = None,
    line_number: int | None = None,
) -> KittiObject:
    """Parse and validate one standard 15-field KITTI annotation line."""
    fields = line.split()
    if len(fields) != 15:
        raise KittiAnnotationError(
            f"expected 15 fields, found {len(fields)}",
            frame_id=frame_id,
            line_number=line_number,
        )

    class_name = fields[0]
    if not class_name:
        raise KittiAnnotationError("class name is empty", frame_id=frame_id, line_number=line_number)

    truncated = _parse_finite_float(fields[1], "truncated", frame_id=frame_id, line_number=line_number)
    try:
        occluded = int(fields[2])
    except ValueError as error:
        raise KittiAnnotationError(
            f"occluded must be an integer category: {fields[2]!r}",
            frame_id=frame_id,
            line_number=line_number,
        ) from error
    alpha = _parse_finite_float(fields[3], "alpha", frame_id=frame_id, line_number=line_number)
    bbox = BoundingBox2D(
        _parse_finite_float(fields[4], "bbox_left", frame_id=frame_id, line_number=line_number),
        _parse_finite_float(fields[5], "bbox_top", frame_id=frame_id, line_number=line_number),
        _parse_finite_float(fields[6], "bbox_right", frame_id=frame_id, line_number=line_number),
        _parse_finite_float(fields[7], "bbox_bottom", frame_id=frame_id, line_number=line_number),
    )
    dimensions = Dimensions3D(
        _parse_finite_float(fields[8], "height", frame_id=frame_id, line_number=line_number),
        _parse_finite_float(fields[9], "width", frame_id=frame_id, line_number=line_number),
        _parse_finite_float(fields[10], "length", frame_id=frame_id, line_number=line_number),
    )
    location = CameraLocation3D(
        _parse_finite_float(fields[11], "location_x", frame_id=frame_id, line_number=line_number),
        _parse_finite_float(fields[12], "location_y", frame_id=frame_id, line_number=line_number),
        _parse_finite_float(fields[13], "location_z", frame_id=frame_id, line_number=line_number),
    )
    rotation_y = _parse_finite_float(fields[14], "rotation_y", frame_id=frame_id, line_number=line_number)
    annotation = KittiObject(
        class_name=class_name,
        truncated=truncated,
        occluded=occluded,
        alpha=alpha,
        bbox_2d=bbox,
        dimensions_3d=dimensions,
        location_3d=location,
        rotation_y=rotation_y,
    )
    validation = validate_kitti_object(annotation)
    if not validation.valid:
        raise KittiAnnotationError(
            "; ".join(validation.errors), frame_id=frame_id, line_number=line_number
        )
    return annotation


def parse_label_file(label_path: str | Path, *, frame_id: str | None = None) -> list[KittiObject]:
    """Parse every non-empty line of a KITTI label file in source order."""
    label_path = Path(label_path)
    resolved_frame_id = frame_id if frame_id is not None else label_path.stem
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise KittiAnnotationError(f"could not read label file: {error}", frame_id=resolved_frame_id) from error

    annotations: list[KittiObject] = []
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            annotations.append(
                parse_annotation_line(line, frame_id=resolved_frame_id, line_number=line_number)
            )
    return annotations
