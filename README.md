# University of Miami - Gait Analysis

This repository now includes an improved gait-analysis pipeline tailored to sagittal walking videos and the Zeni-style event definitions discussed with your collaborators.

## What was improved

- **Event detection now uses relative coordinates** (heel-hip and toe-hip) instead of raw heel trajectories.
  - Heel strike: peak in `heel_x - hip_x`.
  - Toe off: valley in `toe_x - hip_x`.
- **Walking-direction normalization** is automatic using hip velocity, so RL/LR videos use the same event logic.
- **Robust peak filtering** uses minimum stride distance and prominence to reduce false peaks.
- **Stride-consistent HS/TO pairing** only keeps toe-off events between consecutive heel strikes.
- **Metric output format** includes frame-wise HS/TO flags plus summary metrics written to CSV.

## Main script

- `gait_pipeline.py`

## Usage

```bash
python gait_pipeline.py "brandon_01_RL (1).MOV" --label brandon_01_RL --out brandon_01_metrics.csv
python gait_pipeline.py "brandon_02_LR (1).MOV" --label brandon_02_LR --out brandon_02_metrics.csv
```

## Output CSV columns

- `frame`
- `HS_left`, `HS_right`, `TO_left`, `TO_right`
- `speed`, `cadence`, `cycle_time`, `stride_length`, `step_length`, `stance_time`, `swing_time`, `double_support_time`

## Notes on reliability

- Cadence and cycle time are usually the most reliable.
- Stance/swing are estimable from HS/TO pairs, but sensitive to landmark noise.
- Double support from monocular sagittal video is inherently lower-confidence.
