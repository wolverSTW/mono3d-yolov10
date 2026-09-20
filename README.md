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
