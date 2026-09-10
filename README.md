# Biomechanical Gait Analysis Modeling

### Monocular Computer Vision Pipeline for Quantitative Gait Analysis

**Research Intern | Self-Directed Research Project**
**February 2025 – April 2025**
**Supervisor:** Professor Moataz Eltoukhy, University of Miami

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv)](https://opencv.org/) [![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) [![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-8CAAE6?logo=scipy)](https://scipy.org/) [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)

---

## Overview

This repository implements a **monocular video-based gait analysis pipeline** for extracting biomechanically meaningful gait events and spatiotemporal parameters from sagittal-plane walking videos.

The system converts raw video into structured gait measurements through a sequence of:

$$
\boxed{
\text{Video}
\rightarrow
\text{Pose Landmarks}
\rightarrow
\text{Coordinate Processing}
\rightarrow
\text{Temporal Filtering}
\rightarrow
\text{Gait Event Detection}
\rightarrow
\text{Spatiotemporal Metrics}
}
$$

The project was designed to investigate whether **low-cost RGB video and computer vision** can provide a practical alternative for preliminary gait assessment when laboratory-grade motion-capture infrastructure is unavailable.

The implementation combines **MediaPipe Pose**, OpenCV-based geometric processing, Savitzky–Golay temporal filtering, peak/zero-crossing event detection, and rule-based stride consistency checks.

---

# Research Motivation

Traditional biomechanical gait laboratories commonly rely on marker-based motion capture and other specialized instrumentation. These systems offer high measurement fidelity but can be expensive, infrastructure-heavy, and difficult to deploy at scale.

This project explores a simpler acquisition model:

> **Can a conventional single-camera walking video be transformed into stable, interpretable gait measurements through geometric calibration and signal processing?**

The pipeline focuses on recovering robust estimates of:

* Heel-strike and toe-off events
* Cadence
* Gait-cycle time
* Stride length
* Step length
* Walking speed
* Stance time
* Swing time
* Double-support time

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
                  │    x(t), y(t), frame      │
                  └─────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
      ┌──────────────────────┐      ┌──────────────────────┐
      │ Geometric Calibration│      │ Relative Coordinates │
      │  Checkerboard /      │      │ heel - hip / toe -   │
      │  Homography          │      │ hip normalization    │
      └──────────┬───────────┘      └──────────┬───────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                     ┌─────────────────────┐
                     │ Savitzky–Golay     │
                     │ Temporal Smoothing │
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
```

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

MediaPipe landmark indices are explicitly defined in the processing pipeline, with hip, heel, and toe positions converted from normalized coordinates to pixel coordinates using the frame dimensions.

For landmark \(i\) at frame \(t\):

$$
\mathbf{p}_i(t)=
\begin{bmatrix}
x_i(t)\\
y_i(t)
\end{bmatrix}
$$

where \(x_i\) and \(y_i\) represent the image-space coordinates.

For \(N\) observed landmarks over \(T\) frames, the raw trajectory data can be represented as:

$$
\mathbf{X}\in\mathbb{R}^{T\times2N}
$$

---

# 2. Geometric Calibration

The notebook implementation includes a **checkerboard-based planar calibration stage** for mapping pixel coordinates into a metric reference plane.

A checkerboard with known square dimensions is detected using OpenCV's chessboard-corner detection. Corresponding image coordinates and world-plane coordinates are then used to estimate a homography using **RANSAC**.

The projective transformation is:

$$
s
\begin{bmatrix}
x_w\\
y_w\\
1
\end{bmatrix}
=
H
\begin{bmatrix}
x_p\\
y_p\\
1
\end{bmatrix}
$$

where:

* \(H\) = \(3\times3\) homography matrix
* \((x_p,y_p)\) = pixel coordinates
* \((x_w,y_w)\) = coordinates on the calibrated reference plane
* \(s\) = projective scale

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

The notebook explicitly uses an **8.5 cm checkerboard square size** and supports multiple checkerboard patterns for detection robustness.

---

# 3. Relative Coordinate Normalization

The CLI pipeline introduces a more robust event-detection representation by using coordinates relative to the subject's hip rather than directly analyzing raw heel trajectories.

The midpoint of the hips is computed as:

$$
x_{hip}(t)
=
\frac{x_{Lhip}(t)+x_{Rhip}(t)}{2}
$$

After smoothing, the system estimates walking direction from the median hip velocity:

$$
v_{hip}(t)=
\frac{d x_{hip}(t)}{dt}
$$

The sign of the median velocity determines the walking-direction normalization.

For each side:

$$
r_{heel}(t)
=
d\left(x_{heel}(t)-x_{hip}(t)\right)
$$

$$
r_{toe}(t)
=
d\left(x_{toe}(t)-x_{hip}(t)\right)
$$

where \(d\in\{-1,+1\}\) normalizes right-to-left and left-to-right recordings into the same coordinate convention.

### Why Relative Coordinates?

Raw trajectories are sensitive to:

* camera position
* walking direction
* subject location in the frame
* global body translation

Using hip-relative signals suppresses much of the global translation and makes gait-event detection more invariant to whether the subject walks **left-to-right or right-to-left**.

---

# 4. Temporal Signal Processing

Raw pose trajectories contain high-frequency fluctuations caused by landmark-estimation uncertainty.

The pipeline applies an **adaptive Savitzky–Golay filter** to the landmark signals.

The current configuration uses:

```text
Window length : 11 samples
Polynomial     : 3rd order
```

The filter locally approximates the trajectory using a polynomial while preserving the overall shape of the gait signal more effectively than simple moving-average smoothing.

Conceptually:

$$
x_{smooth}(t)=
\mathcal{SG}\left(x(t)\right)
$$

where \(\mathcal{SG}\) represents Savitzky–Golay filtering.

This produces smoother trajectories before numerical differentiation and event detection.

---

# 5. Gait Event Detection

The pipeline identifies two fundamental gait events:

### Heel Strike (HS)

Heel-strike candidates are detected from peaks in the heel-to-hip relative trajectory.

$$
HS = \operatorname{peaks}(r_{heel})
$$

### Toe Off (TO)

Toe-off candidates are detected from valleys in the toe-to-hip relative trajectory:

$$
TO = \operatorname{peaks}(-r_{toe})
$$

The implementation uses `scipy.signal.find_peaks()` with constraints on peak prominence and temporal separation to suppress spurious detections.

Current detection parameters include:

```text
Minimum stride interval : 0.7 s
Minimum step interval   : 0.3 s
Peak prominence         : 6 px
```

---

# 6. Stride-Consistent Event Pairing

A major source of error in gait analysis is false event detection caused by landmark jitter or small oscillations.

The system therefore imposes **temporal and spatial constraints** on detected gait events.

For candidate heel strikes \(HS_i\) and \(HS_{i+1}\), events are retained only when they satisfy the expected stride spacing.

The extended notebook implementation further applies stride locking based on:

$$
\Delta t \geq t_{min}
$$

and

$$
\Delta x \geq \alpha\tilde{S}
$$

where:

* \(\Delta t\) = elapsed time between candidate heel strikes
* \(\Delta x\) = spatial separation
* \(\tilde{S}\) = median stride distance
* \(\alpha\) = minimum acceptable stride fraction

The corresponding toe-off is then selected from the valid event interval between consecutive heel strikes.

This prevents isolated false peaks from propagating into the final gait metrics.

---

# 7. Spatiotemporal Gait Metrics

Once gait events are detected, the pipeline derives higher-level gait parameters.

## Gait Cycle Time

For consecutive heel strikes of the same foot:

$$
T_{cycle}
=
t_{HS,i+1}-t_{HS,i}
$$

The implementation uses the mean of successive left-foot heel-strike intervals.

---

## Stride Length

For consecutive heel strikes:

$$
L_{stride}
=
|x_{HS,i+1}-x_{HS,i}|
$$

After spatial scaling:

$$
L_{stride}^{m}
=
L_{stride}^{px}\cdot s
$$

where \(s\) is the estimated meters-per-pixel scale.

---

## Walking Speed

Walking speed is estimated from stride displacement divided by stride duration:

$$
v
=
\frac{L_{stride}}{T_{cycle}}
$$

The implementation averages stride-level speed estimates across valid cycles.

---

## Step Length

The current implementation approximates step length as half of mean stride length:

$$
L_{step}
=
\frac{L_{stride}}{2}
$$

---

## Cadence

Heel strikes from both feet are merged into a chronologically ordered sequence:

$$
t_1,t_2,\ldots,t_n
$$

with step intervals:

$$
\Delta t_i=t_{i+1}-t_i
$$

Cadence is then estimated as:

$$
C=
\frac{60}{\operatorname{mean}(\Delta t)}
$$

in steps/minute.

---

# 8. Stance and Swing Phase Estimation

The pipeline defines:

### Stance

$$
T_{stance}
=
t_{TO}-t_{HS}
$$

### Swing

$$
T_{swing}
=
t_{HS,next}-t_{TO}
$$

These are computed separately for left and right sides and then aggregated across valid gait cycles.

---

# 9. Double-Support Estimation

Double-support time is estimated from the overlap between the stance intervals of the two limbs:

$$
T_{DS}
=
\min(T_{stance,L},T_{stance,R})
$$

Because double support depends on correctly detecting events on **both sides simultaneously**, it is intrinsically more sensitive to landmark and event-detection errors than cadence or cycle time.

The repository therefore treats it as a lower-confidence monocular estimate.

---

# 10. Spatial Scale Estimation

The pipeline converts image-space displacement into physical units using a spatial scale.

For labeled evaluation sequences, the implementation can estimate the meters-per-pixel factor from the known stride length:

$$
s=
\frac{L_{stride}^{GT}}
     {L_{stride}^{px}}
$$

For sequences without an available reference label, the code falls back to an estimated walking speed and median hip velocity to obtain an approximate spatial scale.

This separation between **temporal gait features** and **spatial calibration** is important because timing metrics are less dependent on camera geometry than absolute distance measurements.

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

The evaluation module computes:

$$
e_{abs}=|\hat{y}-y_{GT}|
$$

and

$$
e_{\%}
=
100
\frac{|\hat{y}-y_{GT}|}{|y_{GT}|}
$$

for each gait parameter.

This creates a direct comparison between computer-vision-derived estimates and the available reference measurements.

---

# 12. Output

The command-line pipeline exports a CSV containing frame-level gait-event indicators and summary metrics.

### Frame-level columns

```text
frame
HS_left
HS_right
TO_left
TO_right
```

### Gait-level columns

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

The repository therefore produces both **event annotations** and **quantitative gait summaries** in a machine-readable format.

Example:

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

The current repository contains both a reusable Python pipeline and a Jupyter-based experimental implementation.

---

# 14. Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy scipy pandas matplotlib
```

The notebook implementation specifically uses **MediaPipe 0.10.21**, together with OpenCV, NumPy, SciPy, Pandas, and Matplotlib.

---

# 15. Running the Pipeline

### Single video

```bash
python gait_pipeline.py \
    "brandon_01_RL (1).MOV" \
    --label brandon_01_RL \
    --out brandon_01_metrics.csv
```

### Second recording

```bash
python gait_pipeline.py \
    "brandon_02_LR (1).MOV" \
    --label brandon_02_LR \
    --out brandon_02_metrics.csv
```

### Multiple videos

```bash
python Gait_Analysis.py \
    "brandon_01_RL (1).MOV" \
    "brandon_02_LR (1).MOV" \
    --out-dir results/
```

The command-line runner automatically generates one metrics CSV per input recording and can optionally perform ground-truth comparisons.

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

Built an end-to-end computer-vision workflow that converts ordinary walking videos into structured gait events and quantitative spatiotemporal measurements.

### Perspective-Aware Coordinate Processing

Implemented checkerboard-based planar calibration and homography mapping to transform image-space landmark observations toward physically meaningful coordinates.

### Noise-Robust Event Detection

Replaced naive raw-coordinate event detection with smoothed **heel-to-hip and toe-to-hip relative signals**, combined with prominence and minimum-distance constraints.

### Direction-Invariant Processing

Automatically determines walking direction from hip velocity, allowing the same event-detection logic to process both left-to-right and right-to-left recordings.

### Stride-Consistent Temporal Reasoning

Introduced temporal and spatial consistency checks so that isolated landmark fluctuations do not create implausible gait cycles.

### Quantitative Evaluation

Implemented automated comparison of estimated gait parameters against reference measurements using absolute and percentage error.

---

# 18. Research Relevance

The project provides a foundation for **low-cost, scalable biomechanical assessment** from monocular video.

A camera-based system of this type could support future research into:

* Automated mobility assessment
* Rehabilitation progress monitoring
* Longitudinal gait tracking
* Abnormal gait characterization
* Remote biomechanical assessment
* Computer-assisted clinical workflows

The current implementation should be viewed as a **research prototype rather than a clinically validated diagnostic system**.

---

# 19. Limitations

Monocular gait analysis introduces several fundamental constraints:

**2D projection:**
A single RGB camera does not directly measure full 3D biomechanics.

**Camera geometry:**
Absolute spatial measurements depend on calibration and camera placement.

**Pose-estimation uncertainty:**
Landmark errors directly propagate into event timing and derived metrics.

**Occlusion:**
Foot and lower-limb landmarks may become unreliable during partial occlusion.

**Event sensitivity:**
Stance, swing, and double-support estimates depend on accurate HS/TO detection.

**Clinical validity:**
Clinical deployment requires rigorous validation against established motion-capture and biomechanical measurement systems.

---

# 20. Future Work

```text
Current System
      │
      ├── Monocular pose estimation
      ├── Homography / spatial calibration
      ├── Temporal filtering
      ├── HS / TO detection
      └── Spatiotemporal metrics
               │
               ▼
        Future Extensions
               │
      ├── Automatic gait-cycle segmentation
      ├── Joint-angle estimation
      ├── Left/right symmetry metrics
      ├── 3D pose reconstruction
      ├── Temporal deep-learning models
      ├── Abnormal-gait classification
      ├── Multi-camera fusion
      └── Validation against optical motion capture
```

---

# Research Context

**Research Intern / Self-Directed Project**
**February 2025 – April 2025**

**Supervisor:** Professor Moataz Eltoukhy
**University of Miami**

The work investigates the intersection of:

$$
\text{Computer Vision}
+
\text{Signal Processing}
+
\text{Biomechanics}
$$

with the goal of developing more accessible quantitative gait-analysis methodologies.

---

# Citation

If this repository is used for research or educational work, please cite the repository and acknowledge the research supervision provided by the University of Miami.

---

# Disclaimer

This repository is intended for research and educational use. The implementation is not a clinically validated medical device and should not be used for diagnosis or treatment decisions without appropriate clinical validation.
