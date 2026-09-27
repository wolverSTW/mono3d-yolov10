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

## Development Status

- EXP-001 — KITTI Dataset Audit — implemented and covered by synthetic-fixture
  tests. The repository's real KITTI directories are currently empty; no claim
  is made that the real dataset has passed this audit.
- EXP-002 — Preprocessing + Coordinate Validation — not started.
- YOLOv10 integration, 3D prediction, training, evaluation, and visualisation
  pipelines — not yet implemented.
