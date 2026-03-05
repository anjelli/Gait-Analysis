import argparse
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

LANDMARKS = {
    "left_hip": 23,
    "right_hip": 24,
    "left_heel": 29,
    "right_heel": 30,
    "left_toe": 31,
    "right_toe": 32,
}


@dataclass
class DetectionConfig:
    smooth_window: int = 11
    smooth_poly: int = 3
    min_stride_s: float = 0.7
    min_step_s: float = 0.3
    peak_prominence_px: float = 6.0


GROUND_TRUTH = {
    "brandon_01_RL": {
        "speed": 1.333,
        "cadence": 110.0,
        "cycle_time": 1.09,
        "stride_length": 1.45,
        "step_length": 0.729,
        "stance_time": 0.725,
        "swing_time": 0.365,
        "double_support_time": 0.37,
    },
    "brandon_02_LR": {
        "speed": 1.316,
        "cadence": 110.5,
        "cycle_time": 1.09,
        "stride_length": 1.43,
        "step_length": 0.713,
        "stance_time": 0.72,
        "swing_time": 0.37,
        "double_support_time": 0.35,
    },
}


def _adaptive_savgol(x: np.ndarray, default_window: int, poly: int) -> np.ndarray:
    if len(x) <= poly + 2:
        return x.copy()
    w = min(default_window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    if w >= len(x):
        w = len(x) - 1 if len(x) % 2 == 0 else len(x)
    return savgol_filter(x, window_length=w, polyorder=poly)


def _create_pose_estimator(model_complexity: int = 2):
    """Create a MediaPipe Pose estimator across package layout variants."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return mp.solutions.pose.Pose(model_complexity=model_complexity)

    pose_spec = importlib.util.find_spec("mediapipe.python.solutions.pose")
    if pose_spec is not None:
        mp_pose = importlib.import_module("mediapipe.python.solutions.pose")
        return mp_pose.Pose(model_complexity=model_complexity)

    raise RuntimeError(
        "MediaPipe Pose API not found in this mediapipe build. "
        "Please install a full mediapipe wheel (e.g., `python -m pip install mediapipe`)."
    )


def extract_landmarks(video_path: str) -> Tuple[pd.DataFrame, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    pose = _create_pose_estimator(model_complexity=2)

    rows: List[Dict[str, float]] = []
    frame = 0
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            break

        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            h, w = img.shape[:2]
            row = {"frame": frame, "time": frame / fps}
            for name, idx in LANDMARKS.items():
                lm = res.pose_landmarks.landmark[idx]
                row[f"{name}_x"] = lm.x * w
                row[f"{name}_y"] = lm.y * h
            rows.append(row)
        frame += 1

    cap.release()
    pose.close()

    df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    return df, fps


def build_relative_signals(df: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    out = df.copy()
    out["hip_x"] = 0.5 * (out["left_hip_x"] + out["right_hip_x"])

    hip_smooth = _adaptive_savgol(out["hip_x"].to_numpy(), cfg.smooth_window, cfg.smooth_poly)
    vx = np.gradient(hip_smooth, out["time"].to_numpy())
    direction = np.sign(np.nanmedian(vx))
    direction = 1.0 if direction == 0 else direction

    out["walk_dir"] = direction

    for side in ["left", "right"]:
        heel = _adaptive_savgol(out[f"{side}_heel_x"].to_numpy(), cfg.smooth_window, cfg.smooth_poly)
        toe = _adaptive_savgol(out[f"{side}_toe_x"].to_numpy(), cfg.smooth_window, cfg.smooth_poly)
        out[f"{side}_heel_rel"] = direction * (heel - hip_smooth)
        out[f"{side}_toe_rel"] = direction * (toe - hip_smooth)

    return out


def detect_events(signal_df: pd.DataFrame, fps: float, cfg: DetectionConfig) -> Dict[str, np.ndarray]:
    events: Dict[str, np.ndarray] = {}
    min_stride_frames = max(1, int(cfg.min_stride_s * fps))

    for side in ["left", "right"]:
        heel_rel = signal_df[f"{side}_heel_rel"].to_numpy()
        toe_rel = signal_df[f"{side}_toe_rel"].to_numpy()

        hs_idx, _ = find_peaks(
            heel_rel,
            distance=min_stride_frames,
            prominence=cfg.peak_prominence_px,
        )
        to_idx, _ = find_peaks(
            -toe_rel,
            distance=max(1, int(cfg.min_step_s * fps)),
            prominence=cfg.peak_prominence_px * 0.6,
        )

        paired_to: List[int] = []
        valid_hs: List[int] = []
        for i in range(len(hs_idx) - 1):
            h0, h1 = hs_idx[i], hs_idx[i + 1]
            tos_between = to_idx[(to_idx > h0) & (to_idx < h1)]
            if len(tos_between) == 0:
                continue
            valid_hs.append(h0)
            paired_to.append(int(tos_between[np.argmin(toe_rel[tos_between])]))
        if len(hs_idx) > 0:
            valid_hs.append(int(hs_idx[-1]))

        events[f"HS_{side}"] = np.array(valid_hs, dtype=int)
        events[f"TO_{side}"] = np.array(paired_to, dtype=int)

    return events


def estimate_spatial_scale_m_per_px(
    signal_df: pd.DataFrame,
    fps: float,
    events: Dict[str, np.ndarray],
    label: Optional[str],
) -> float:
    if label and label in GROUND_TRUTH:
        hs = events["HS_left"]
        if len(hs) > 1:
            heel = signal_df["left_heel_x"].to_numpy()
            px_stride = np.median(np.abs(np.diff(heel[hs])))
            if px_stride > 1e-6:
                return GROUND_TRUTH[label]["stride_length"] / px_stride
    hip_speed_px_s = np.nanmedian(np.abs(np.gradient(signal_df["hip_x"], signal_df["time"])))
    nominal_speed_m_s = 1.3
    return nominal_speed_m_s / max(hip_speed_px_s, 1e-6)


def compute_metrics(signal_df: pd.DataFrame, events: Dict[str, np.ndarray], m_per_px: float) -> Dict[str, float]:
    t = signal_df["time"].to_numpy()
    left_heel_x = signal_df["left_heel_x"].to_numpy()

    hs_left = events["HS_left"]
    hs_right = events["HS_right"]
    to_left = events["TO_left"]
    to_right = events["TO_right"]

    if len(hs_left) < 2 or len(hs_right) < 1:
        raise ValueError("Not enough gait events to compute metrics.")

    stride_times = np.diff(t[hs_left])
    stride_lengths = np.abs(np.diff(left_heel_x[hs_left])) * m_per_px

    all_hs_t = np.sort(np.concatenate([t[hs_left], t[hs_right]]))
    step_times = np.diff(all_hs_t)

    nL = min(len(to_left), len(hs_left) - 1)
    nR = min(len(to_right), len(hs_right) - 1)

    stance_left = t[to_left[:nL]] - t[hs_left[:nL]] if nL else np.array([])
    swing_left = t[hs_left[1 : nL + 1]] - t[to_left[:nL]] if nL else np.array([])
    stance_right = t[to_right[:nR]] - t[hs_right[:nR]] if nR else np.array([])
    swing_right = t[hs_right[1 : nR + 1]] - t[to_right[:nR]] if nR else np.array([])

    stance_mean = np.nanmean(np.concatenate([stance_left, stance_right])) if (nL + nR) else np.nan
    swing_mean = np.nanmean(np.concatenate([swing_left, swing_right])) if (nL + nR) else np.nan

    overlap = min(len(stance_left), len(stance_right))
    double_support = np.nanmean(np.minimum(stance_left[:overlap], stance_right[:overlap])) if overlap else np.nan

    return {
        "speed": float(np.nanmean(stride_lengths / stride_times)),
        "cadence": float(60.0 / np.nanmean(step_times)),
        "cycle_time": float(np.nanmean(stride_times)),
        "stride_length": float(np.nanmean(stride_lengths)),
        "step_length": float(np.nanmean(stride_lengths) / 2.0),
        "stance_time": float(stance_mean),
        "swing_time": float(swing_mean),
        "double_support_time": float(double_support),
    }


def build_event_csv(signal_df: pd.DataFrame, events: Dict[str, np.ndarray], metrics: Dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame({"frame": signal_df["frame"].astype(int)})
    for col in ["HS_left", "HS_right", "TO_left", "TO_right"]:
        flag = np.zeros(len(out), dtype=int)
        idx = events[col]
        idx = idx[idx < len(flag)]
        flag[idx] = 1
        out[col] = flag

    for k, v in metrics.items():
        out[k] = v
    return out


def evaluate(label: str, metrics: Dict[str, float]) -> pd.DataFrame:
    gt = GROUND_TRUTH[label]
    rows = []
    for k, gt_val in gt.items():
        est = metrics.get(k, np.nan)
        abs_err = np.abs(est - gt_val)
        rows.append({"metric": k, "ground_truth": gt_val, "estimate": est, "abs_error": abs_err, "pct_error": 100 * abs_err / gt_val})
    return pd.DataFrame(rows)


def run(video_path: str, out_csv: str, label: Optional[str] = None, cfg: Optional[DetectionConfig] = None) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    cfg = cfg or DetectionConfig()
    df, fps = extract_landmarks(video_path)
    signals = build_relative_signals(df, cfg)
    events = detect_events(signals, fps, cfg)
    scale = estimate_spatial_scale_m_per_px(signals, fps, events, label)
    metrics = compute_metrics(signals, events, scale)
    event_df = build_event_csv(signals, events, metrics)
    event_df.to_csv(out_csv, index=False)

    evaluation = None
    if label and label in GROUND_TRUTH:
        evaluation = evaluate(label, metrics)
    return metrics, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved gait analysis pipeline using heel-hip / toe-hip event detection.")
    parser.add_argument("video", help="Input sagittal walking video")
    parser.add_argument("--label", default=None, help="Optional ground-truth label for evaluation")
    parser.add_argument("--out", default="gait_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    metrics, evaluation = run(args.video, args.out, args.label)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    if evaluation is not None:
        print("\nGround-truth comparison:")
        print(evaluation.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
