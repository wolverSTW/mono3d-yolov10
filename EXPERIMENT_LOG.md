## EXP-001 — KITTI Dataset Audit

**Status:** implemented and verified with the EXP-001 synthetic-fixture test suite
(`14 passed`).

**Inputs:** a KITTI-style dataset root containing `image_2`, `label_2`, and
`calib` directories.

**Outputs:** a JSON audit report containing file counts, frame matching,
missing/duplicate identifiers, image diagnostics, label/calibration diagnostics,
observed class counts, `DontCare` count, and an overall status.

**Validation rules:** image files must be readable with the configured channel
count; labels must have 15 fields, recognised KITTI types, and finite numeric
fields; calibration files must be readable, numeric, and include a valid 3 × 4
`P2` matrix. All three modalities must contain files and share frame IDs.

**Test strategy:** temporary synthetic fixtures only. The real dataset is not
present and is neither populated nor modified by this experiment.

**Limitations:** this experiment performs no preprocessing, coordinate
transformation, target encoding, split generation, model training, or metric
evaluation.

---

## EXP-002 — Preprocessing + Coordinate Validation

**Status:** implemented and verified with the EXP-002 synthetic-fixture test suite
(`20 passed`).

**Inputs:** a validated KITTI dataset root (output of EXP-001 audit) containing
`image_2`, `label_2`, and `calib` directories.

**Outputs:** a JSON preparation manifest (`kitti-preparation-manifest-v1`)
containing global preprocessing metadata, deterministic train/validation split,
and per-frame entries with relative paths, original image size, annotation
counts, preprocessing transform (scale, padding), and transformed P2 matrix.

**Validation rules:**
- Frame must have exactly one image, one label, one calibration file
- Image must be readable with expected channel count
- Labels must parse as valid 15-field KITTI annotations with finite numeric
  values; 2D boxes must satisfy x1 < x2, y1 < y2 within image bounds
- 3D dimensions (height, width, length) must be positive for non-DontCare objects
- 3D location (X, Y, Z) must be finite
- Calibration must contain valid P2 (3×4) matrix with finite values
- After preprocessing, transformed 2D boxes must remain valid within output
  image bounds
- Train/validation split must be non-overlapping and cover all valid frames

**Test strategy:** temporary synthetic fixtures only. Tests cover:
- Annotation parsing and coordinate validation (6 tests)
- Calibration parsing and intrinsic derivation (4 tests)
- Image transform mathematics: bbox, inverse bbox, projection matrix (5 tests)
- Image preprocessing (letterbox padding value, canvas size) (1 test)
- Deterministic split generation and manifest creation (4 tests)
- CLI integration (2 tests)

**Limitations:** The real KITTI dataset has not been processed. No PyTorch
Dataset/DataLoader, no target encoding for model input, no class mapping
configured, no data augmentation.

---

## EXP-003A — Dataset + Target Encoding

**Status:** implemented and verified with the EXP-003A synthetic-fixture test suite
(`19 passed` — 10 target_encoder + 9 dataset).

**Inputs:** EXP-002 preparation manifest (`kitti-preparation-manifest-v1`) and
KITTI dataset root with `image_2`, `label_2`, `calib` directories.

**Outputs:** PyTorch Dataset yielding samples with:
- Preprocessed image tensor [3, H, W] float32 in [0, 1]
- EncodedTarget: class_ids [N], bboxes_2d [N,4] normalised [0,1],
  dimensions_3d [N,3] (H,W,L metres), locations_3d [N,3] (X,Y,Z camera metres),
  rotation_y [N] (radians), image_size, transformed_p2 [3,4]
- Frame ID, original image size, calibration dict

**Validation rules:**
- Frame must exist in manifest for the requested split
- Image must be readable and convertible to RGB
- Manifest preprocessing parameters must match declared transform
- Annotations must parse as valid 15-field KITTI entries
- All annotation classes must be in configured/discovered class mapping
- DontCare annotations excluded from encoded targets
- Transformed 2D boxes remain valid in output image coordinates
- Collate function produces stacked images [B,3,H,W] and list of targets

**Test strategy:** temporary synthetic fixtures only. Tests cover:
- Class mapping: with/without config, DontCare exclusion, sorting (2 tests)
- Target encoding: basic, multiple objects, DontCare exclusion, empty, unknown
  class error, dtype preservation, P2 passthrough, bbox decode (8 tests)
- Dataset: construction, getitem types, split filtering, multiple objects,
  DontCare exclusion, collate, unknown class error, empty split error (9 tests)

**Limitations:** Real KITTI dataset not processed. No model, heads, losses,
training, or evaluation implemented.

---

## EXP-003B — YOLOv10 Model Integration

**Status:** implemented and verified with the EXP-003B synthetic-fixture test suite
(`17 passed`).

**Inputs:** RGB tensor [B, 3, H, W] (H, W from config, default 640×640).

**Outputs:** Feature pyramid dict with:
- `p3`: [B, C3, H/8, W/8] (e.g., 64 channels for yolov10n)
- `p4`: [B, C4, H/16, W/16] (e.g., 128 channels for yolov10n)
- `p5`: [B, C5, H/32, W/32] (e.g., 256 channels for yolov10n)
- `backbone_channels`: (C3, C4, C5)
- `strides`: (8, 16, 32)

**Validation rules:**
- Input tensor must be [B, 3, H, W] with H, W divisible by 32
- Output feature maps must be finite (no NaN/Inf)
- Feature map spatial dimensions must match expected strides
- Channel counts must match variant-specific expectations

**Test strategy:** temporary synthetic fixtures only. Tests cover:
- Config defaults and custom values (2 tests)
- Backbone construction for all 5 variants (2 tests)
- Feature channels correctness per variant (1 test)
- Forward pass shapes, batch sizes, input resolutions (4 tests)
- Parameter count sanity check (1 test)
- Factory functions (2 tests)
- Baseline model integration (5 tests)

**Limitations:** Real KITTI dataset not processed. No 3D prediction heads,
losses, training, or evaluation implemented. CUDA not available locally —
GPU verification pending. Pretrained weights not loaded (architecture-only
mode for development).

---

## EXP-003C — Baseline 3D Prediction Heads

**Status:** implemented and verified with synthetic-fixture test suite
(`70 passed` total).

**Inputs:** RGB tensor [B, 3, H, W] (H, W from config, default 640×640).

**Outputs:** Structured `BaselineOutput` dataclass with:

- `detection_2d`: `Detection2DOutput` — class_logits [B, N, C], bboxes_2d [B, N, 4]
  normalised [0,1], objectness [B, N]
- `dimensions_3d`: `Dimension3DOutput` — dimensions [B, N, 3] (h,w,l metres, exp transform),
  logits [B, N, 3]
- `locations_3d`: `Location3DOutput` — locations [B, N, 3] (X,Y,Z camera metres, direct regression),
  logits [B, N, 3]
- `orientation`: `OrientationOutput` — rotation_y [B, N] (radians, direct regression), logits [B, N, 1]
- `backbone_features`: dict with p3, p4, p5, channels, strides

**Coordinate Conventions (preserved from EXP-002):**
- 2D bbox: (x1, y1, x2, y2) normalised [0,1]
- 3D dims: (h, w, l) metres — exact h,w,l ordering
- 3D location: (X, Y, Z) rectified camera coords — X horizontal, Y vertical,
  Z forward depth (camera-axis depth, NOT Euclidean distance)
- Orientation: `rotation_y` in radians — NOT `alpha` (observation angle)

**Validation rules:**
- Input tensor [B, 3, H, W] with H, W divisible by 32
- Output feature maps finite (no NaN/Inf)
- Dimension outputs strictly positive (exp transform)
- Dimension ordering: h, w, l
- Location: X, Y, Z in camera coords; Z is depth not distance
- Orientation: rotation_y in radians; NOT alpha

**Test strategy:** synthetic fixtures only. Tests cover:
- 2D detection head: construction, output shapes, finite outputs
- Dimension head: construction, output shapes, positive dimensions, h/w/l ordering
- Location head: construction, output shapes, X/Y/Z convention
- Orientation head: construction, output shapes, rotation_y representation
- Full model integration: construction, forward pass, structured output keys,
  batch handling, finite outputs, parameter counts
- Multi-scale feature usage: P3/P4/P5 all consumed by heads
- Class mapping: num_classes from config respected

**Limitations:** Real KITTI dataset not processed. No losses, training,
evaluation, geometry-guided enhancement, Transformer/context, uncertainty,
distance refinement. CUDA not available locally — GPU verification pending.
