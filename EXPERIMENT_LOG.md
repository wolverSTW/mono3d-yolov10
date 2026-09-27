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
