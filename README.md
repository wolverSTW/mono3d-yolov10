# A Lightweight YOLOv10-Based Geometry-Guided Framework for Monocular 3D Object Detection and Distance Estimation

## Project Overview

This project investigates a lightweight YOLOv10-based framework for monocular 3D object detection and metric distance estimation.

The proposed research focuses on integrating geometry-guided feature learning with a YOLOv10-based monocular 3D detection framework to improve 3D object localisation and distance estimation while maintaining computational efficiency.

The project is developed as part of the MSc Computer Science dissertation.

---

## Research Aim

The aim of this research is to investigate a lightweight YOLOv10-based geometry-guided framework for monocular 3D object detection and metric distance estimation, with the objective of improving 3D localisation performance while maintaining computational efficiency.

---

## Research Questions

### RQ1

How does a lightweight YOLOv10-based monocular 3D detection baseline perform in terms of 3D object detection and metric distance estimation on the KITTI 3D Object Detection Benchmark?

### RQ2

To what extent does the integration of geometry-guided feature learning improve 3D object localisation and metric distance estimation compared with the baseline model?

### RQ3

What is the trade-off between the performance improvements provided by the geometry-guided component and the additional computational cost of the proposed framework?

---

## Research Objectives

1. Establish a lightweight YOLOv10-based monocular 3D detection baseline for estimating object location, dimensions, orientation, and depth.

2. Design and integrate a geometry-guided feature learning mechanism into the baseline framework.

3. Investigate metric distance estimation using geometry-aware depth information.

4. Evaluate the proposed framework against the baseline using 3D detection, distance estimation, and computational efficiency metrics.

5. Conduct ablation experiments to determine the contribution of the geometry-guided components and analyse the accuracy-efficiency trade-off.

6. Assess the practical feasibility of the proposed framework using inference speed, model complexity, and computational resource requirements.

---

## Proposed Architecture

The conceptual architecture of the proposed framework is:

```text
Single RGB Image
        │
        ▼
YOLOv10-Based Feature Extraction
        │
        ▼
Geometry-Guided Feature Learning
        │
        ▼
Lightweight Feature Fusion
        │
        ├───────────────┐
        ▼               ▼
3D Detection Head    Depth & Distance
                     Estimation Head
        │               │
        └───────┬───────┘
                ▼
     3D Object Output
     + Estimated Distance
```

---

## Dataset Audit

EXP-001 implements a non-destructive audit for a KITTI 3D Object Detection
dataset. It validates the expected `image_2`, `label_2`, and `calib`
directories; discovers frames by identifier rather than file order; checks
cross-directory frame matching and duplicate identifiers; verifies image
readability, dimensions, and channel counts; validates KITTI label records;
and validates calibration entries required for later monocular geometry.

The audit currently requires the `P2` calibration matrix to contain 12 numeric
values (a 3 × 4 projection matrix). It reports observed label types and
`DontCare` separately. It does not establish a model class mapping: no such
mapping has yet been configured.

### Expected KITTI structure

```text
data/raw/kitti/
├── image_2/    # RGB images
├── label_2/    # KITTI 3D annotations
└── calib/      # KITTI calibration files
```

### Run the audit

From the repository root, after installing the project requirements:

```powershell
.\.venv\Scripts\python.exe scripts\audit_kitti.py
```

The default dataset root and JSON output path are configured in
`configs/config.yaml`. The default report path is
`logs/evaluation/kitti_audit.json`.

For an explicitly supplied dataset location and report path:

```powershell
.\.venv\Scripts\python.exe scripts\audit_kitti.py `
  --data-root D:\path\to\kitti `
  --output logs\evaluation\kitti_audit.json
```

The command prints a concise summary, writes a machine-readable JSON report,
and exits with code `0` only when all structural and content checks pass.
A failed audit records diagnostics and is expected, for example, when a
modality is missing, files are malformed, or the dataset directories are empty.

### Local development and GPU server execution

Use the same command locally for code development and synthetic-fixture tests.
After cloning the repository and making KITTI available on a GPU server, the
same audit command can be run there with `--data-root` pointing to the server's
dataset location. The GPU-server execution command is documented but has not
yet been verified on a GPU server; the audit itself does not require a GPU.

---

## Dataset Preparation (EXP-002)

EXP-002 implements preprocessing and coordinate validation for the KITTI dataset.
It builds on the EXP-001 audit to produce a validated, split manifest ready for
training.

### What EXP-002 does

1. **Frame validation** — verifies every frame has matching image, label, and
   calibration files with no duplicates
2. **Coordinate validation** — parses and validates KITTI 3D annotations (2D
   boxes, 3D dimensions, 3D location, orientation) against original image size
3. **Camera calibration parsing** — extracts and validates the P2 projection
   matrix and derives camera intrinsics (fx, fy, cx, cy)
4. **Image preprocessing** — supports two modes:
   - `resize` — anisotropic direct resize to target dimensions
   - `letterbox` — aspect-ratio-preserving resize with centred padding (default)
5. **Geometric consistency** — transforms 2D bounding boxes and the P2 projection
   matrix using the same image-plane transformation matrix `A`, ensuring
   `P' = A @ P` holds exactly
6. **Train/validation split** — deterministic, reproducible frame-level split
   using a configured seed
7. **Manifest generation** — writes a JSON manifest with all metadata needed for
   a PyTorch Dataset (relative paths, preprocessing parameters, transformed P2,
   split assignment)

### Expected KITTI structure

Same as EXP-001:
```text
data/raw/kitti/
├── image_2/    # RGB images
├── label_2/    # KITTI 3D annotations
└── calib/      # KITTI calibration files
```

### Run the preparation

From the repository root, after installing the project requirements:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_kitti.py
```

The default dataset root, manifest output path, validation ratio, seed,
preprocessing mode, and output image size are configured in
`configs/config.yaml`. The default manifest path is
`data/splits/kitti/exp002_manifest.json`.

For an explicitly supplied dataset location and manifest path:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_kitti.py `
  --data-root D:\path\to\kitti `
  --output data\splits\kitti\my_manifest.json `
  --validation-ratio 0.2 `
  --seed 42 `
  --preprocessing-mode letterbox
```

The command prints a concise summary, writes a machine-readable JSON manifest,
and exits with code `0` only when all frames are valid and split successfully.
A failed preparation records diagnostics for invalid frames in the manifest and
exits with code `1`.

### Manifest contents

The generated manifest (`kitti-preparation-manifest-v1` format) contains:

- **Global metadata**: preprocessing mode, output image size, split configuration
- **Per-frame entries**: relative paths to image/label/calibration, original
  image size, annotation counts, preprocessing transform (scale, padding), and
  the transformed 3×4 P2 projection matrix
- **Invalid samples**: frames that failed validation with reasons (not silently
  discarded)

### Local development and GPU server execution

Use the same command locally for code development and synthetic-fixture tests.
After cloning the repository and making KITTI available on a GPU server, the
same preparation command can be run there with `--data-root` pointing to the
server's dataset location. The GPU-server execution command is documented but
has not yet been verified on a GPU server; preparation itself does not require
a GPU.

---

## Dataset + Target Encoding (EXP-003A)

EXP-003A implements the PyTorch Dataset interface and target encoding for the
KITTI 3D object detection baseline. It consumes the EXP-002 manifest and
produces model-ready samples.

### What EXP-003A does

1. **PyTorch Dataset** — `KittiManifestDataset` loads images, annotations, and
   calibration on demand using the EXP-002 manifest
2. **Preprocessing application** — applies the declared letterbox/resize transform
   from the manifest to images
3. **Target encoding** — converts raw KITTI annotations to model-ready tensors:
   - Class indices (from configured class mapping)
   - Normalised 2D bounding boxes [0, 1] by output image size
   - 3D dimensions (height, width, length) in metres
   - 3D location (X, Y, Z) in rectified camera coordinates, metres
   - Orientation (rotation_y) in radians
4. **DontCare handling** — excludes DontCare annotations from training targets
5. **Collate function** — handles variable object counts per frame with list-based
   targets and stacked image tensors
6. **Coordinate conventions** — preserves EXP-002 conventions (H/W/L dims,
   X/Y/Z camera coords, alpha ≠ rotation_y)

### Expected KITTI structure

Same as EXP-001/002:
```text
data/raw/kitti/
├── image_2/    # RGB images
├── label_2/    # KITTI 3D annotations
└── calib/      # KITTI calibration files
```

### Run the dataset (local smoke test)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dataset.py -v
```

All tests use synthetic fixtures — no real KITTI data required.

### Target encoding details

The `EncodedTarget` dataclass contains:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `class_ids` | [N] | int64 | Class indices (0..C-1) |
| `bboxes_2d` | [N, 4] | float32 | Normalised (x1, y1, x2, y2) in [0, 1] |
| `dimensions_3d` | [N, 3] | float32 | (height, width, length) in metres |
| `locations_3d` | [N, 3] | float32 | (X, Y, Z) camera coords in metres |
| `rotation_y` | [N] | float32 | Camera-frame yaw in radians |
| `image_size` | - | tuple | (width, height) of preprocessed image |
| `transformed_p2` | [3, 4] | float32 | Transformed projection matrix |

**Normalisation choices (baseline defaults):**
- 2D bbox: divided by output image width/height → [0, 1]
- 3D dimensions: absolute metres (no scaling)
- 3D location: absolute metres (no scaling)
- Orientation: rotation_y in radians (no encoding)
- Class: integer index from configured class mapping

These are baseline defaults, not claimed to be optimal.

### Class mapping

If `dataset.classes` is configured in `configs/config.yaml`, it is used as the
authoritative mapping. Otherwise, the dataset discovers classes from label files
(excluding DontCare). Unknown classes raise an error at sample load time.

### Collate function

`kitti_collate_fn` returns a dict:
- `images`: [B, 3, H, W] float32 in [0, 1]
- `targets`: List[EncodedTarget] length B
- `frame_ids`: List[str] length B
- `original_sizes`: List[(width, height)] length B
- `calibrations`: List[dict] length B

Variable object counts per frame are handled by keeping targets as a list.

### Local development and GPU server execution

Use the same dataset class locally for code development and synthetic-fixture
tests. On a GPU server, instantiate `KittiManifestDataset` with the manifest
path and dataset root. The dataset itself does not require a GPU.

---

## YOLOv10 Model Integration (EXP-003B)

EXP-003B integrates a verified YOLOv10 feature extractor as the backbone for
the monocular 3D detection baseline. It uses the ultralytics implementation
via the official YAML configurations, avoiding pretrained weight downloads
during development.

### What EXP-003B does

1. **YOLOv10 backbone adapter** — `Yolov10Backbone` wraps the ultralytics
   `DetectionModel` built from official YAML configs (yolov10n/s/m/l/x)
2. **Feature pyramid extraction** — uses forward hooks to capture multi-scale
   feature maps (P3, P4, P5) from the backbone+neck, letting the ultralytics
   forward pass handle all skip connections (Concat, Upsample) correctly
3. **Configuration-driven** — variant, pretrained flag, input channels from config
4. **Clean interface** — returns feature pyramid dict suitable for custom 3D
   prediction heads (EXP-003C)

### Architecture

```text
Input [B, 3, H, W]
        │
        ▼
YOLOv10 Backbone (ultralytics DetectionModel)
        │
        ├── Conv stem (stride 2, 4)
        ├── C2f blocks + SCDown (backbone)
        ├── SPPF + PSA (neck)
        ├── Upsample + Concat + C2f (P3, stride 8)
        ├── Conv + Concat + C2f (P4, stride 16)
        └── SCDown + Concat + C2fCIB (P5, stride 32)
        │
        ▼
Feature Pyramid (P3, P4, P5) → Custom 3D Heads (EXP-003C)
```

### Supported variants

| Variant | Width | P3 channels | P4 channels | P5 channels | Params (approx) |
|---------|-------|-------------|-------------|-------------|-----------------|
| yolov10n | 0.25 | 64 | 128 | 256 | 2.8M |
| yolov10s | 0.50 | 128 | 256 | 512 | 8.1M |
| yolov10m | 0.75 | 192 | 384 | 768 | — |
| yolov10l | 1.00 | 256 | 512 | 1024 | — |
| yolov10x | 1.25 | 320 | 640 | 1280 | — |

Channels are base (256, 512, 1024) scaled by width multiplier.

### Run the model (local smoke test)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_yolov10.py -v
```

All tests use synthetic fixtures — no real KITTI data or pretrained weights required.

### Model interface

`Baseline3DDetector` returns a dict:

| Key | Shape | Description |
|-----|-------|-------------|
| `p3` | [B, C3, H/8, W/8] | P3 feature map |
| `p4` | [B, C4, H/16, W/16] | P4 feature map |
| `p5` | [B, C5, H/32, W/32] | P5 feature map |
| `backbone_channels` | tuple | (C3, C4, C5) |
| `strides` | tuple | (8, 16, 32) |

### Configuration

In `configs/config.yaml`:

```yaml
model:
  name: "yolov10"
  variant: "yolov10n"
  num_classes: 3
```

The backbone config is derived from the model section.

### Local development and GPU server execution

Use the same model class locally for code development and synthetic-fixture
tests. On a GPU server, instantiate `Baseline3DDetector` with the config.
The model runs on CPU or CUDA automatically.

---

## Baseline 3D Prediction Heads (EXP-003C)

EXP-003C implements the baseline 3D prediction heads on top of the YOLOv10
feature pyramid. It adds custom prediction heads for 2D detection, 3D
dimensions, 3D location, and orientation — completing the baseline
architecture.

### What EXP-003C does

1. **2D Detection Head** (`Detection2DHead`) — custom anchor-free head on the
   feature pyramid (P3, P4, P5) predicting class logits, normalised 2D
   bounding boxes, and objectness scores. This is a CUSTOM head, NOT the native
   YOLOv10 `v10Detect` head, to maintain a clean separation for the 3D baseline.
2. **3D Dimension Head** (`Dimension3DHead`) — predicts (height, width, length)
   in metres using an exponential transform to ensure positive dimensions.
3. **3D Location Head** (`Location3DHead`) — predicts (X, Y, Z) in rectified
   camera coordinates (metres) using direct regression. Z is camera-axis depth,
   NOT Euclidean distance.
4. **Orientation Head** (`OrientationHead`) — predicts `rotation_y` (camera-frame
   yaw) in radians via direct regression. Does NOT use `alpha` (observation angle).
5. **Structured Output** — `BaselineOutput` dataclass with explicit fields for
   each prediction type and backbone features.

### Architecture

```text
Input [B, 3, H, W]
        │
        ▼
YOLOv10 Backbone
        │
        ▼
Feature Pyramid (P3, P4, P5)
        │
        ├── 2D Detection Head (class_logits, bboxes_2d, objectness)
        ├── 3D Dimension Head (h, w, l in metres, exp transform)
        ├── 3D Location Head (X, Y, Z in metres, camera coords)
        └── Orientation Head (rotation_y in radians, direct regression)
        │
        ▼
Structured BaselineOutput
```

### Prediction Representations

| Head | Output | Shape | Convention |
|------|--------|-------|------------|
| 2D Detection | `class_logits` | [B, N, C] | Class logits per object |
| | `bboxes_2d` | [B, N, 4] | Normalised (x1,y1,x2,y2) in [0,1] |
| | `objectness` | [B, N] | Sigmoid confidence |
| 3D Dimension | `dimensions` | [B, N, 3] | (h, w, l) metres, exp(logits) |
| 3D Location | `locations` | [B, N, 3] | (X, Y, Z) camera coords, metres |
| Orientation | `rotation_y` | [B, N] | Camera-frame yaw, radians |

**Coordinate Conventions (preserved from EXP-002):**
- 2D bbox: (x1, y1, x2, y2) normalised to [0, 1]
- 3D dims: (height, width, length) in metres — exact h, w, l ordering
- 3D location: (X, Y, Z) rectified camera coords — X horizontal, Y vertical, Z forward depth
- Orientation: `rotation_y` in radians — NOT `alpha` (observation angle)
- Z is camera-axis depth; NOT Euclidean distance

### Run the model (local smoke test)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_yolov10.py -v
```

All tests use synthetic fixtures — no real KITTI data or pretrained weights required.

### Model Interface

`Baseline3DDetector` returns `BaselineOutput` dataclass:

| Key | Type | Description |
|-----|------|-------------|
| `detection_2d` | `Detection2DOutput` | Class logits, bboxes_2d, objectness |
| `dimensions_3d` | `Dimension3DOutput` | Dimensions (h,w,l), logits |
| `locations_3d` | `Location3DOutput` | Locations (X,Y,Z), logits |
| `orientation` | `OrientationOutput` | rotation_y, logits |
| `backbone_features` | dict | P3, P4, P5, channels, strides |

### Configuration

In `configs/config.yaml`:

```yaml
model:
  name: "yolov10"
  variant: "yolov10n"
  num_classes: 3
```

Head configurations can be customised via `BaselineConfig`.

### Local development and GPU server execution

Use the same model class locally for code development and synthetic-fixture
tests. On a GPU server, instantiate `Baseline3DDetector` with the config.
The model runs on CPU or CUDA automatically.

---

## Baseline Multi-Task Loss (EXP-003D)

EXP-003D implements the baseline multi-task loss functions required to train
the EXP-003C baseline model. It provides modular, configurable loss functions
for each prediction head and a combined multi-task loss aggregator.

### What EXP-003D does

1. **2D Detection Loss** (`DetectionLoss`) — combines:
   - Box regression: Smooth L1 on normalised (x1, y1, x2, y2)
   - Objectness: BCELoss on sigmoid probabilities (1=matched, 0=background)
   - Classification: CrossEntropyLoss on class logits
2. **3D Dimension Loss** (`DimensionLoss`) — Smooth L1 on physical dimensions
   (h, w, l) in metres. Model uses `exp(logits)` for positivity.
3. **3D Location Loss** (`LocationLoss`) — Smooth L1 on (X, Y, Z) in metres.
   Z is camera-axis depth, NOT Euclidean distance.
4. **Orientation Loss** (`OrientationLoss`) — angular difference with periodicity
   handling via `atan2(sin(pred-target), cos(pred-target))`, then Smooth L1.
5. **Total Loss** (`TotalLoss`) — weighted sum of all components with
   configurable weights. Returns total and individual components.

### Architecture

```text
Model Outputs + Targets
        │
        ▼
Assignment (fixed-order)
        │
        ├── Box Loss (Smooth L1 on [0,1] coords)
        ├── Objectness (BCELoss on sigmoid probs)
        ├── Classification (CrossEntropy on logits)
        ├── Dimension 3D (Smooth L1 on exp(logits) in metres)
        ├── Location 3D (Smooth L1 on X,Y,Z camera coords)
        ├── Orientation (Angular diff + Smooth L1)
        │
        ▼
TotalLoss = Σ λ_i * L_i
```

### Loss Formulas

| Loss | Formula | Target |
|------|---------|--------|
| Box | Smooth L1 on (x1,y1,x2,y2) ∈ [0,1] | Normalised coords |
| Objectness | BCELoss(p, y) where p∈[0,1], y∈{0,1} | Matched=1, bg=0 |
| Classification | CrossEntropy(logits, class_idx) | Class indices |
| Dimension | Smooth L1 on exp(logits) in metres | (h,w,l) metres |
| Location | Smooth L1 on direct regression | (X,Y,Z) camera metres |
| Orientation | Smooth L1 on atan2(sin(Δ),cos(Δ)) | rotation_y radians |

**Coordinate Conventions (preserved from EXP-002):**
- 2D bbox: (x1, y1, x2, y2) normalised to [0, 1]
- 3D dims: (height, width, length) in metres — exact h, w, l ordering
- 3D location: (X, Y, Z) rectified camera coords — X horizontal, Y vertical, Z forward depth
- Orientation: `rotation_y` in radians — NOT `alpha` (observation angle)
- Z is camera-axis depth; NOT Euclidean distance

### Run the loss tests (local smoke test)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_*_loss.py -v
```

All tests use synthetic fixtures — no real KITTI data required.

### Configuration

In `configs/config.yaml`:

```yaml
loss:
  box_weight: 1.0
  objectness_weight: 1.0
  classification_weight: 1.0
  dimension_3d_weight: 1.0
  location_3d_weight: 1.0
  orientation_weight: 1.0
  box_loss_beta: 1.0/9.0
```

### Local development and GPU server execution

Use the same loss modules locally for code development and synthetic-fixture
tests. On a GPU server, instantiate `TotalLoss` with the config. The loss
modules run on CPU or CUDA automatically.

---

## Development Status

- EXP-001 — KITTI Dataset Audit — implemented and covered by synthetic-fixture
  tests (14 passed). The repository's real KITTI directories are currently
  empty; no claim is made that the real dataset has passed this audit.
- EXP-002 — Preprocessing + Coordinate Validation — implemented and covered by
  synthetic-fixture tests (20 passed). The real dataset has not been processed.
- EXP-003A — Dataset + Target Encoding — implemented and covered by synthetic-fixture
  tests (19 passed). The real dataset has not been processed.
- EXP-003B — YOLOv10 Model Integration — implemented and covered by synthetic-fixture
  tests (17 passed). The real dataset has not been processed.
- EXP-003C — Baseline 3D Prediction Heads — implemented and covered by synthetic-fixture
  tests (70 total tests passed). The real dataset has not been processed.
- EXP-003D — Baseline Multi-Task Loss — implemented and covered by synthetic-fixture
  tests (108 total tests passed). The real dataset has not been processed.
- Geometry-guided enhancement, training, evaluation, and visualisation
  pipelines — not yet implemented.