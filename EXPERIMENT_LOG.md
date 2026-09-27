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
