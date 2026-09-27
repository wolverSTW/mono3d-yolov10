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
