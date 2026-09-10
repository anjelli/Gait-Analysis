Copy-paste this directly into your GitHub `README.md`:

````markdown
# Biomechanical Gait Analysis Modeling

### Monocular Computer Vision Pipeline for Quantitative Gait Analysis

**Research Intern | Self-Directed Research Project**  
**February 2025 – April 2025**  
**Supervisor:** Professor Moataz Eltoukhy, University of Miami

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv)](https://opencv.org/) [![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) [![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-8CAAE6?logo=scipy)](https://scipy.org/) [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)

---

## Overview

This repository implements a **monocular video-based gait analysis pipeline** for extracting biomechanically meaningful gait events and spatiotemporal parameters from sagittal-plane walking videos.

The system converts raw video into structured gait measurements through the following pipeline:

**Video → Pose Landmarks → Coordinate Processing → Temporal Filtering → Gait Event Detection → Spatiotemporal Metrics**

The project was designed to investigate whether **low-cost RGB video and computer vision** can provide a practical alternative for preliminary gait assessment when laboratory-grade motion-capture infrastructure is unavailable.

The implementation combines **MediaPipe Pose**, OpenCV-based geometric processing, Savitzky–Golay temporal filtering, peak-based event detection, and stride-consistency constraints.

---

# Research Motivation

Traditional biomechanical gait laboratories commonly rely on marker-based motion capture and other specialized instrumentation. These systems provide high measurement fidelity but can be expensive, infrastructure-heavy, and difficult to deploy at scale.

This project explores a simpler acquisition model:

> **Can a conventional single-camera walking video be transformed into stable, interpretable gait measurements through geometric calibration and signal processing?**

The pipeline focuses on recovering estimates of:

- Heel-strike and toe-off events
- Cadence
- Gait-cycle time
- Stride length
- Step length
- Walking speed
- Stance time
- Swing time
- Double-support time

---

# System Architecture

```text
                     ┌─────────────────────┐
                     │   Sagittal Video    │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Frame Extraction   │
                     │      OpenCV         │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Pose Estimation   │
                     │     MediaPipe       │
                     └──────────┬──────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ Anatomical Landmark Series │
                  │      x(t), y(t), frame     │
                  └─────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
      ┌──────────────────────┐      ┌──────────────────────┐
      │ Geometric Calibration│      │ Relative Coordinates │
      │ Checkerboard /        │      │ Heel-to-Hip /        │
      │ Homography            │      │ Toe-to-Hip           │
      └──────────┬───────────┘      └──────────┬───────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                     ┌─────────────────────┐
                     │ Savitzky–Golay      │
                     │ Temporal Smoothing  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Gait Event Detection│
                     │ HS / TO             │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Event Consistency   │
                     │ / Stride Filtering  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Gait Metric         │
                     │ Computation         │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ CSV + Evaluation    │
                     └─────────────────────┘
````

---

# 1. Pose Estimation

The pipeline uses **MediaPipe Pose** to obtain frame-wise anatomical landmarks from monocular RGB video.

The current gait-event implementation extracts the bilateral:

* Left hip
* Right hip
* Left heel
* Right heel
* Left toe
* Right toe

The detected landmarks are converted from MediaPipe's normalized image coordinates into pixel coordinates using the frame dimensions.

For a landmark at frame `t`, its position is represented as:

**(x(t), y(t))**

For `N` tracked landmarks over `T` frames, the resulting trajectory data forms a `T × 2N` coordinate matrix.

This representation provides the temporal foundation for downstream gait-event detection and biomechanical feature extraction.

---

# 2. Geometric Calibration

The notebook implementation includes a **checkerboard-based planar calibration stage** for mapping pixel coordinates into a metric reference plane.

A checkerboard with known square dimensions is detected using OpenCV's chessboard-corner detection. Corresponding image coordinates and world-plane coordinates are then used to estimate a homography using **RANSAC**.

The transformation can be understood as:

**[x_world, y_world, 1] = H × [x_pixel, y_pixel, 1]**

where:

* `H` is the `3 × 3` homography matrix
* `(x_pixel, y_pixel)` are image coordinates
* `(x_world, y_world)` are coordinates on the calibrated reference plane
* the homogeneous coordinate accounts for projective scaling

The calibrated coordinates are then converted from centimeters to meters.

### Calibration Pipeline

```text
RGB Frame
   │
   ▼
Grayscale + Histogram Equalization
   │
   ▼
Checkerboard Detection
   │
   ▼
Known Physical Grid Coordinates
   │
   ▼
RANSAC Homography Estimation
   │
   ▼
Pixel → Ground-Plane Coordinates
```

The notebook uses an **8.5 cm checkerboard square size** and supports multiple checkerboard patterns for calibration robustness.

---

# 3. Relative Coordinate Normalization

The gait-event pipeline uses coordinates relative to the subject's hip rather than directly analyzing absolute foot coordinates.

The hip midpoint is calculated as:

**x_hip(t) = (x_left_hip(t) + x_right_hip(t)) / 2**

Hip velocity is then estimated from the temporal derivative of the hip trajectory:

**v_hip(t) = d(x_hip(t)) / dt**

The median hip velocity is used to determine walking direction.

For each side, the heel and toe signals are expressed relative to the hip:

**r_heel(t) = d × (x_heel(t) − x_hip(t))**

**r_toe(t) = d × (x_toe(t) − x_hip(t))**

where `d` is either `+1` or `−1` and normalizes both right-to-left and left-to-right recordings into the same direction convention.

### Why Relative Coordinates?

Raw trajectories are sensitive to:

* Camera position
* Walking direction
* Subject location in the frame
* Global body translation

Using hip-relative signals suppresses much of the global translation and makes gait-event detection more consistent across walking directions.

---

# 4. Temporal Signal Processing

Raw pose trajectories contain high-frequency fluctuations caused by landmark-estimation uncertainty.

The pipeline applies a **Savitzky–Golay filter** to smooth the trajectory signals while preserving their overall temporal shape.

Current configuration:

```text
Window length : 11 samples
Polynomial     : 3rd order
```

Conceptually:

**x_smooth(t) = SavitzkyGolay(x(t))**

The purpose of this step is to reduce frame-to-frame landmark jitter before numerical differentiation and gait-event detection.

This is particularly important because noisy trajectories can generate false peaks and, consequently, incorrect gait events.

---

# 5. Gait Event Detection

The pipeline identifies two primary gait events:

### Heel Strike (HS)

Heel-strike candidates are identified from peaks in the heel-to-hip relative trajectory.

**HS = peaks(r_heel)**

### Toe Off (TO)

Toe-off candidates are identified from valleys in the toe-to-hip relative trajectory.

**TO = peaks(−r_toe)**

The implementation uses `scipy.signal.find_peaks()` together with minimum temporal spacing and peak-prominence constraints to suppress spurious detections.

Current detection parameters:

```text
Minimum stride interval : 0.7 s
Minimum step interval   : 0.3 s
Peak prominence         : 6 px
```

---

# 6. Stride-Consistent Event Pairing

A major source of error in monocular gait analysis is false event detection caused by landmark jitter and small oscillations.

The pipeline therefore applies **temporal and spatial consistency checks** to candidate gait events.

For consecutive heel strikes, the system checks whether the detected events satisfy expected stride spacing.

The extended notebook implementation also applies stride-locking constraints based on:

**Δt ≥ minimum stride interval**

and

**Δx ≥ minimum stride-distance fraction**

where:

* `Δt` is the time between candidate heel strikes
* `Δx` is the spatial separation between events
* the reference stride distance is estimated from valid gait cycles
* the minimum stride fraction controls spatial consistency

Toe-off events are then selected from the valid interval between consecutive heel strikes.

This prevents isolated false peaks from propagating into the final gait metrics.

---

# 7. Spatiotemporal Gait Metrics

Once gait events are detected, the pipeline derives higher-level gait parameters.

## Gait Cycle Time

For consecutive heel strikes of the same foot:

**Gait cycle time = t_HS(next) − t_HS(current)**

The implementation uses the mean interval between successive heel strikes.

---

## Stride Length

For consecutive heel strikes:

**Stride length = |x_HS(next) − x_HS(current)|**

After spatial scaling:

**Stride length (m) = Stride length (px) × meters-per-pixel scale**

---

## Walking Speed

Walking speed is estimated from stride displacement divided by stride duration:

**Walking speed = Stride length / Gait cycle time**

The implementation averages stride-level speed estimates across valid gait cycles.

---

## Step Length

The current implementation approximates step length as:

**Step length = Stride length / 2**

---

## Cadence

Heel strikes from both feet are merged into chronological order.

For consecutive events:

**Step interval = t(next) − t(current)**

Cadence is estimated as:

**Cadence = 60 / mean(step interval)**

with the result expressed in **steps per minute**.

---

# 8. Stance and Swing Phase Estimation

The pipeline estimates stance and swing durations from detected heel-strike and toe-off events.

### Stance

**Stance time = Toe-off time − Heel-strike time**

### Swing

**Swing time = Next heel-strike time − Toe-off time**

These values are computed separately for the left and right sides and then aggregated across valid gait cycles.

---

# 9. Double-Support Estimation

Double-support time is estimated from the overlap between the stance intervals of the two limbs.

The implementation approximates this using the shorter concurrent stance interval:

**Double-support time = min(left stance time, right stance time)**

Because double-support estimation depends on correctly identifying events on both limbs, it is more sensitive to gait-event errors than metrics such as cadence or cycle time.

---

# 10. Spatial Scale Estimation

The pipeline converts image-space displacement into physical units using a spatial scale.

For labeled evaluation sequences, the meters-per-pixel scale can be estimated from a known reference stride:

**meters-per-pixel scale = ground-truth stride length / pixel stride length**

For sequences without a reference label, the implementation can estimate spatial scale using walking-speed information and the median hip velocity.

This creates a useful distinction between:

* **Temporal metrics**, which are largely derived directly from frame timing
* **Spatial metrics**, which depend on camera geometry and calibration

---

# 11. Evaluation Framework

The repository contains reference measurements for two walking sequences:

```text
brandon_01_RL
brandon_02_LR
```

Reference parameters include:

* Walking speed
* Cadence
* Cycle time
* Stride length
* Step length
* Stance time
* Swing time
* Double-support time

The evaluation module compares estimated values against ground-truth measurements using:

**Absolute error = |estimated − ground truth|**

**Percentage error = 100 × |estimated − ground truth| / |ground truth|**

This provides a quantitative framework for evaluating the reliability of the computer-vision pipeline.

---

# 12. Output

The command-line pipeline exports a CSV containing both frame-level gait-event annotations and summary gait metrics.

### Frame-Level Output

```text
frame
HS_left
HS_right
TO_left
TO_right
```

### Gait-Level Output

```text
speed
cadence
cycle_time
stride_length
step_length
stance_time
swing_time
double_support_time
```

This output structure makes the pipeline suitable for downstream statistical analysis, visualization, and machine-learning workflows.

### Example

```csv
frame,HS_left,HS_right,TO_left,TO_right,speed,cadence,cycle_time,stride_length,step_length,stance_time,swing_time,double_support_time
0,0,0,0,0,1.32,109.8,1.09,1.44,0.72,0.72,0.37,0.35
...
```

---

# 13. Repository Structure

```text
Gait-Analysis/
│
├── gait_pipeline.py
│   └── Main processing and gait-event pipeline
│
├── Gait_Analysis.py
│   └── Command-line runner for one or more videos
│
├── Gait_Analysis.ipynb
│   └── Experimental / calibration / visualization workflow
│
├── README.md
│
├── brandon_01_RL (1).MOV
├── brandon_02_LR (1).MOV
│
└── generated_metrics.csv
```

---

# 14. Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install opencv-python mediapipe numpy scipy pandas matplotlib
```

The notebook implementation uses **MediaPipe 0.10.21** together with OpenCV, NumPy, SciPy, Pandas, and Matplotlib.

---

# 15. Running the Pipeline

### Single Video

```bash
python gait_pipeline.py \
    "brandon_01_RL (1).MOV" \
    --label brandon_01_RL \
    --out brandon_01_metrics.csv
```

### Second Recording

```bash
python gait_pipeline.py \
    "brandon_02_LR (1).MOV" \
    --label brandon_02_LR \
    --out brandon_02_metrics.csv
```

### Multiple Videos

```bash
python Gait_Analysis.py \
    "brandon_01_RL (1).MOV" \
    "brandon_02_LR (1).MOV" \
    --out-dir results/
```

The command-line runner generates a metrics CSV for each input recording and can optionally perform ground-truth comparisons.

---

# 16. Technical Stack

| Component             | Technology                      |
| --------------------- | ------------------------------- |
| Video processing      | OpenCV                          |
| Pose estimation       | MediaPipe Pose                  |
| Numerical computation | NumPy                           |
| Signal processing     | SciPy                           |
| Peak detection        | `scipy.signal.find_peaks`       |
| Temporal smoothing    | Savitzky–Golay filter           |
| Data handling         | Pandas                          |
| Visualization         | Matplotlib                      |
| Geometric calibration | OpenCV Homography + RANSAC      |
| Interface             | Python / CLI / Jupyter Notebook |

---

# 17. Key Engineering Contributions

### Monocular Landmark-to-Gait Pipeline

Built an end-to-end computer-vision workflow that transforms conventional walking videos into structured gait events and quantitative spatiotemporal measurements.

### Perspective-Aware Coordinate Processing

Implemented checkerboard-based planar calibration and homography mapping to transform image-space observations toward physically meaningful reference-plane coordinates.

### Noise-Robust Event Detection

Developed gait-event detection using smoothed **heel-to-hip and toe-to-hip relative trajectories**, combined with peak prominence and temporal separation constraints.

### Direction-Invariant Processing

Automatically determines walking direction from hip motion and normalizes right-to-left and left-to-right recordings into a common coordinate convention.

### Stride-Consistent Temporal Reasoning

Applied temporal and spatial consistency constraints to reduce false gait events caused by landmark-estimation noise.

### Quantitative Evaluation

Implemented automated comparison between estimated gait parameters and available reference measurements using absolute and percentage error.

---

# 18. Research Relevance

The project provides a foundation for **low-cost and scalable biomechanical assessment from monocular video**.

Potential future applications include:

* Automated mobility assessment
* Rehabilitation progress monitoring
* Longitudinal gait tracking
* Abnormal gait characterization
* Remote biomechanical assessment
* Computer-assisted gait analysis

The current system is a **research prototype** and is not presented as a clinically validated diagnostic system.

---

# 19. Limitations

Monocular gait analysis introduces several important limitations.

### 2D Projection

A single RGB camera does not directly measure full 3D human motion.

### Camera Geometry

Absolute spatial measurements depend on calibration, camera placement, and the validity of the reference plane.

### Pose Estimation Uncertainty

Errors in detected landmarks propagate directly into gait-event timing and derived measurements.

### Occlusion

Lower-limb landmarks may become unreliable when the feet or legs are partially occluded.

### Event Sensitivity

Metrics such as stance, swing, and double-support time are highly dependent on accurate heel-strike and toe-off detection.

### Clinical Validation

Clinical deployment would require rigorous validation against established biomechanical measurement systems.

---

# 20. Future Work

```text
Current System
      │
      ├── Monocular pose estimation
      ├── Geometric calibration
      ├── Temporal filtering
      ├── HS / TO detection
      └── Spatiotemporal gait metrics
               │
               ▼
        Future Extensions
               │
      ├── Automatic gait-cycle segmentation
      ├── Joint-angle estimation
      ├── Left/right symmetry analysis
      ├── 3D pose reconstruction
      ├── Temporal deep-learning models
      ├── Abnormal-gait classification
      ├── Multi-camera fusion
      └── Validation against optical motion capture
```

---


# Citation

If this repository is used for research or educational purposes, please cite the repository and acknowledge the research supervision provided by the University of Miami.

```
```
