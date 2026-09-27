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

**Next step:** EXP-003A — Dataset + Target Encoding.

---

## EXP-003A — Dataset + Target Encoding

**Objective:** implement PyTorch Dataset and target encoding for KITTI 3D
detection baseline, consuming EXP-002 manifest.

**Implementation:** added `target_encoder.py` (class mapping, annotation
encoding/decoding, bbox normalisation) and `dataset.py` (`KittiManifestDataset`,
`kitti_collate_fn`). Key features:
- Manifest-driven dataset loading with on-demand image/label/calib reading
- Letterbox/resize preprocessing applied from manifest parameters
- Target encoding: class indices, normalised 2D bboxes [0,1], absolute 3D dims
  (H,W,L in metres), absolute 3D location (X,Y,Z camera coords in metres),
  rotation_y in radians
- DontCare excluded from training targets
- Variable object counts handled via list-based collate function
- Class mapping from config or discovered from labels
- Preserves EXP-002 coordinate conventions (H/W/L, X/Y/Z, alpha≠rotation_y)

**Tests:** 19 synthetic-fixture tests (10 target_encoder + 9 dataset) covering
class mapping, encoding/decoding, DontCare exclusion, multiple objects,
unknown class handling, collate function, split filtering, deterministic
behaviour.

**Limitations:** Real KITTI dataset not processed. No model, heads, losses,
training, or evaluation yet.

**Next step:** EXP-003B — YOLOv10 Model Integration.

---

## EXP-003B — YOLOv10 Model Integration

**Objective:** integrate a verified YOLOv10 feature extractor as the backbone for
the monocular 3D detection baseline.

**Implementation:** added `yolov10.py` backbone adapter using ultralytics
`DetectionModel` built from official YAML configs, and `baseline.py` model
wrapper. Key features:
- Loads YOLOv10 architecture from official YAML configs (yolov10n/s/m/l/x)
- Uses forward hooks on backbone+neck layers (indices 16, 19, 22) to capture
  P3, P4, P5 feature maps, letting ultralytics forward handle skip connections
- Configuration-driven: variant, pretrained flag, input channels from config
- Returns feature pyramid dict (P3, P4, P5) for custom 3D heads (EXP-003C)
- Supports all 5 YOLOv10 variants with width-scaled channels

**Environment:** Python 3.11, PyTorch 2.4.0+cpu, torchvision 0.19.0+cpu,
ultralytics 8.4.163. CUDA not available locally (CPU-only development).

**Tests:** 17 synthetic-fixture tests covering config, construction, all variants,
forward pass shapes, batch sizes, input resolutions, parameter counts, factory
functions, baseline model integration. All pass.

**Limitations:** Real KITTI dataset not processed. No 3D prediction heads,
losses, training, or evaluation yet. CUDA not available locally — GPU
verification pending.

**Next step:** EXP-003C — 3D Prediction Heads.
