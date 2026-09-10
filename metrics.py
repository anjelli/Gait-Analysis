from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GaitMetrics:
    cadence_spm: Optional[float]
    cycle_time_s: Optional[float]
    stride_length_m: Optional[float]
    step_length_m: Optional[float]
    walking_speed_mps: Optional[float]
    stance_time_s: Optional[float]
    swing_time_s: Optional[float]
    double_support_time_s: Optional[float]
    valid_cycles: int

    def to_dict(self) -> Dict[str, float | int | None]:
        return asdict(self)


def robust_median(values: np.ndarray) -> Optional[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def estimate_stride_lengths(
    tracks: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    coordinate_suffix: str = "_x_m",
) -> np.ndarray:
    if cycles.empty:
        return np.array([], dtype=float)

    values = []
    for _, row in cycles.iterrows():
        a = int(row["hs_frame"])
        b = int(row["next_hs_frame"])
        x0 = float(tracks.loc[a, f"pelvis{coordinate_suffix}"])
        x1 = float(tracks.loc[b, f"pelvis{coordinate_suffix}"])
        if np.isfinite(x0) and np.isfinite(x1):
            values.append(abs(x1 - x0))
    return np.asarray(values, dtype=float)


def compute_cadence(
    tracks: pd.DataFrame,
    events: Dict[str, np.ndarray],
) -> Optional[float]:
    all_hs = np.concatenate([
        np.asarray(events.get("HS_left", []), dtype=int),
        np.asarray(events.get("HS_right", []), dtype=int),
    ])
    if len(all_hs) < 2:
        return None

    all_hs = np.sort(np.unique(all_hs))
    times = tracks.loc[all_hs, "time_s"].to_numpy(dtype=float)
    dt = np.diff(times)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(60.0 / np.mean(dt)) if len(dt) else None


def estimate_double_support(cycles: pd.DataFrame) -> Optional[float]:
    if cycles.empty or not {"left", "right"}.issubset(set(cycles["side"])):
        return None

    left = cycles[cycles["side"] == "left"]
    right = cycles[cycles["side"] == "right"]

    overlaps = []
    for _, l in left.iterrows():
        for _, r in right.iterrows():
            start = max(float(l.hs_time_s), float(r.hs_time_s))
            end = min(float(l.to_time_s), float(r.to_time_s))
            if end > start:
                overlaps.append(end - start)

    return robust_median(np.asarray(overlaps)) if overlaps else None


def compute_gait_metrics(
    tracks: pd.DataFrame,
    cycles: pd.DataFrame,
    events: Dict[str, np.ndarray],
    *,
    meters_per_pixel: Optional[float] = None,
) -> GaitMetrics:
    if cycles.empty:
        return GaitMetrics(
            cadence_spm=compute_cadence(tracks, events),
            cycle_time_s=None,
            stride_length_m=None,
            step_length_m=None,
            walking_speed_mps=None,
            stance_time_s=None,
            swing_time_s=None,
            double_support_time_s=None,
            valid_cycles=0,
        )

    cycle_time = robust_median(cycles["cycle_time_s"].to_numpy())
    stance_time = robust_median(cycles["stance_time_s"].to_numpy())
    swing_time = robust_median(cycles["swing_time_s"].to_numpy())

    stride_length = None
    step_length = None
    speed = None

    if meters_per_pixel is not None:
        stride_px = []
        for _, row in cycles.iterrows():
            a = int(row["hs_frame"])
            b = int(row["next_hs_frame"])
            x0 = tracks.loc[a, "pelvis_x_px_smooth"]
            x1 = tracks.loc[b, "pelvis_x_px_smooth"]
            if np.isfinite(x0) and np.isfinite(x1):
                stride_px.append(abs(float(x1 - x0)))

        if stride_px and cycle_time is not None and cycle_time > 0:
            stride_length = float(np.median(stride_px) * meters_per_pixel)
            step_length = stride_length / 2.0
            speed = stride_length / cycle_time

    return GaitMetrics(
        cadence_spm=compute_cadence(tracks, events),
        cycle_time_s=cycle_time,
        stride_length_m=stride_length,
        step_length_m=step_length,
        walking_speed_mps=speed,
        stance_time_s=stance_time,
        swing_time_s=swing_time,
        double_support_time_s=estimate_double_support(cycles),
        valid_cycles=len(cycles),
    )


def compare_to_ground_truth(
    estimated: GaitMetrics,
    ground_truth: Dict[str, float],
) -> pd.DataFrame:
    est = estimated.to_dict()
    rows = []

    for metric, gt in ground_truth.items():
        value = est.get(metric)
        if value is None:
            continue
        gt = float(gt)
        value = float(value)
        abs_error = abs(value - gt)
        pct_error = 100.0 * abs_error / abs(gt) if gt != 0 else np.nan
        rows.append({
            "metric": metric,
            "estimated": value,
            "ground_truth": gt,
            "absolute_error": abs_error,
            "percent_error": pct_error,
        })

    return pd.DataFrame(rows)
