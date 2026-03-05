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

## Run without Jupyter Notebook

Use the plain Python script (no notebook required):

```bash
python Gait_Analysis.py "brandon_01_RL (1).MOV"
python Gait_Analysis.py "brandon_01_RL (1).MOV" "brandon_02_LR (1).MOV"
```

You can also set labels and output directory explicitly:

```bash
python Gait_Analysis.py "brandon_01_RL (1).MOV" "brandon_02_LR (1).MOV" --labels brandon_01_RL brandon_02_LR --out-dir outputs
```

## Output CSV columns

- `frame`
- `HS_left`, `HS_right`, `TO_left`, `TO_right`
- `speed`, `cadence`, `cycle_time`, `stride_length`, `step_length`, `stance_time`, `swing_time`, `double_support_time`

## Notes on reliability

- Cadence and cycle time are usually the most reliable.
- Stance/swing are estimable from HS/TO pairs, but sensitive to landmark noise.
- Double support from monocular sagittal video is inherently lower-confidence.


## Troubleshooting (Windows)

If your terminal shows `AttributeError: module "mediapipe" has no attribute "solutions"`, this project now includes a compatibility fallback in `gait_pipeline.py` for alternate MediaPipe package layouts.

Still failing? Recreate the environment with Python 3.11 x64 and reinstall deps inside `.venv`:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe numpy pandas scipy
```

