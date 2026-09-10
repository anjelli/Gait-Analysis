from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass(frozen=True)
class GaitEvent:
    frame: int
    time_s: float
    event_type: str
    side: str
    prominence: float


def estimate_walking_direction(tracks: pd.DataFrame) -> float:
    """Return +1 for increasing x and -1 for decreasing x."""
    t = tracks["time_s"].to_numpy(dtype=float)
    x = tracks["pelvis_x_smooth"].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(x)
    t, x = t[valid], x[valid]
    if len(t) < 3:
        return 1.0

    velocity = np.gradient(x, t)
    velocity = velocity[np.isfinite(velocity)]
    if len(velocity) == 0:
        return 1.0

    return 1.0 if float(np.median(velocity)) >= 0 else -1.0


def detect_heel_strikes(
    relative_heel: np.ndarray,
    *,
    distance_frames: int,
    prominence: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(relative_heel)
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    signal = relative_heel[valid]
    peaks, properties = find_peaks(
        signal,
        distance=distance_frames,
        prominence=prominence,
    )
    return idx[peaks], properties.get("prominences", np.array([], dtype=float))


def detect_toe_offs(
    relative_toe: np.ndarray,
    *,
    distance_frames: int,
    prominence: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(relative_toe)
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    signal = -relative_toe[valid]
    peaks, properties = find_peaks(
        signal,
        distance=distance_frames,
        prominence=prominence,
    )
    return idx[peaks], properties.get("prominences", np.array([], dtype=float))


def build_relative_signals(
    tracks: pd.DataFrame,
    direction: float,
) -> pd.DataFrame:
    out = tracks.copy()

    pelvis = out["pelvis_x_smooth"]
    for side in ("left", "right"):
        out[f"{side}_heel_rel_smooth"] = direction * (
            out[f"{side}_heel_x_px_smooth"] - pelvis
        )
        out[f"{side}_toe_rel_smooth"] = direction * (
            out[f"{side}_toe_x_px_smooth"] - pelvis
        )
    return out


def detect_gait_events(
    tracks: pd.DataFrame,
    *,
    min_stride_interval_s: float,
    min_step_interval_s: float,
    heel_prominence: float,
    toe_prominence: float,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], float]:
    t = tracks["time_s"].to_numpy(dtype=float)
    if len(t) < 3:
        raise ValueError("At least three frames are required.")

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("Non-increasing frame timestamps.")

    fps = 1.0 / dt
    stride_distance = max(1, round(min_stride_interval_s * fps))
    step_distance = max(1, round(min_step_interval_s * fps))

    direction = estimate_walking_direction(tracks)
    processed = build_relative_signals(tracks, direction)

    events: Dict[str, np.ndarray] = {}

    for side in ("left", "right"):
        hs_frames, hs_prom = detect_heel_strikes(
            processed[f"{side}_heel_rel_smooth"].to_numpy(),
            distance_frames=stride_distance,
            prominence=heel_prominence,
        )
        to_frames, to_prom = detect_toe_offs(
            processed[f"{side}_toe_rel_smooth"].to_numpy(),
            distance_frames=step_distance,
            prominence=toe_prominence,
        )

        events[f"HS_{side}"] = hs_frames
        events[f"TO_{side}"] = to_frames

    return processed, events, direction


def pair_gait_cycles(
    tracks: pd.DataFrame,
    events: Dict[str, np.ndarray],
    *,
    min_toe_off_fraction: float = 0.10,
    max_toe_off_fraction: float = 0.85,
) -> pd.DataFrame:
    time = tracks["time_s"].to_numpy(dtype=float)
    rows: List[dict] = []

    for side in ("left", "right"):
        hs = events[f"HS_{side}"]
        to = events[f"TO_{side}"]

        if len(hs) < 2:
            continue

        for i in range(len(hs) - 1):
            hs0, hs1 = int(hs[i]), int(hs[i + 1])
            cycle = time[hs1] - time[hs0]
            if cycle <= 0:
                continue

            candidates = to[(to > hs0) & (to < hs1)]
            if len(candidates) == 0:
                continue

            # Select a TO event in a physiologically plausible portion of the cycle.
            valid_candidates = []
            for to_frame in candidates:
                fraction = (time[int(to_frame)] - time[hs0]) / cycle
                if min_toe_off_fraction <= fraction <= max_toe_off_fraction:
                    valid_candidates.append(int(to_frame))

            if not valid_candidates:
                continue

            to_frame = valid_candidates[0]
            stance = time[to_frame] - time[hs0]
            swing = time[hs1] - time[to_frame]

            rows.append(
                {
                    "side": side,
                    "hs_frame": hs0,
                    "to_frame": to_frame,
                    "next_hs_frame": hs1,
                    "hs_time_s": time[hs0],
                    "to_time_s": time[to_frame],
                    "next_hs_time_s": time[hs1],
                    "cycle_time_s": cycle,
                    "stance_time_s": stance,
                    "swing_time_s": swing,
                    "toe_off_cycle_fraction": (time[to_frame] - time[hs0]) / cycle,
                }
            )

    return pd.DataFrame(rows)


def event_indicator_frame(
    n_frames: int,
    events: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = pd.DataFrame({"frame": np.arange(n_frames)})
    for name, frames in events.items():
        indicator = np.zeros(n_frames, dtype=np.int8)
        valid = np.asarray(frames, dtype=int)
        valid = valid[(valid >= 0) & (valid < n_frames)]
        indicator[valid] = 1
        out[name] = indicator
    return out
