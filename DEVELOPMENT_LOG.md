## EXP-001 — KITTI Dataset Audit

**Objective:** establish structural and annotation-level validity of a KITTI
3D Object Detection dataset before preprocessing or model development.

**Implementation:** added a non-destructive audit module and CLI. The audit
discovers image, label, and calibration files by frame identifier; detects
missing and duplicate identifiers; checks image readability and channel
metadata; validates the 15-field KITTI label structure and finite numeric
values; validates calibration readability, numeric values, known matrix shapes,
and the required `P2` 3 × 4 matrix; and writes a JSON report.

**Tests:** synthetic temporary fixtures cover valid matching, all directional
missing-modality cases, malformed labels and calibration, invalid numeric
fields, unreadable images, duplicate identifiers, class distribution,
`DontCare`, empty datasets, and CLI report generation.

**Limitations:** `data/raw/kitti` contains only empty directories in this
repository. The implementation has been tested only against synthetic fixtures;
no real KITTI dataset statistics or audit outcome are claimed.

**Next step:** EXP-002 — Preprocessing + Coordinate Validation.

---

## EXP-002 — Preprocessing + Coordinate Validation

**Objective:** parse, validate, and transform KITTI annotations and images into
a geometrically consistent, split manifest ready for training.

**Implementation:** added parsing and validation for KITTI 3D annotations
(`kitti_parser.py`), camera calibration (`calibration.py`), image-plane
transformations with geometric consistency (`transforms.py`), and a preparation
pipeline that produces a split manifest (`converter.py`, `prepare_kitti.py`).
Key features:
- 15-field KITTI label parsing with structured dataclasses preserving
  dimension order (H, W, L) and camera coordinates (X, Y, Z)
- Separate `alpha` (observation angle) and `rotation_y` (camera-frame yaw)
  handling
- `DontCare` regions parsed but excluded from physical constraints
- P2 projection matrix parsing with intrinsic derivation (fx, fy, cx, cy)
- Letterbox (aspect-preserving) and resize preprocessing modes
- 3×3 image-plane transform matrix `A` applied to both 2D boxes and P2
  as `P' = A @ P`
- Deterministic train/validation split with configurable seed
- JSON manifest with per-frame metadata including transformed P2

**Tests:** 20 synthetic-fixture tests covering annotation parsing, calibration
parsing, coordinate validation, transform mathematics (bbox, inverse bbox,
projection matrix), image preprocessing, deterministic splitting, manifest
generation, CLI success/failure paths, and invalid frame reporting.

**Limitations:** The real KITTI dataset has not been processed. No PyTorch
Dataset or DataLoader implemented yet. Class mapping not configured.

**Next step:** EXP-003 — YOLOv10 3D Baseline.
