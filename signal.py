from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def interpolate_1d(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate internal missing values; edge gaps use nearest valid value."""
    y = np.asarray(x, dtype=float).copy()
    idx = np.arange(len(y))
    valid = np.isfinite(y)

    if valid.sum() == 0:
        return y
    if valid.sum() == 1:
        y[:] = y[valid][0]
        return y

    y[~valid] = np.interp(idx[~valid], idx[valid], y[valid])
    return y


def valid_savgol_window(length: int, requested: int, polyorder: int) -> int | None:
    if length < 5:
        return None

    window = min(requested, length if length % 2 else length - 1)

    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1

    if window < minimum:
        window = minimum

    if window > length:
        window = length if length % 2 else length - 1

    if window <= polyorder:
        return None

    return window


def smooth_series(
    x: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    y = interpolate_1d(x)
    window = valid_savgol_window(len(y), window_length, polyorder)
    if window is None:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


def smooth_landmark_columns(
    tracks: pd.DataFrame,
    columns: list[str],
    *,
    window_length: int,
    polyorder: int,
) -> pd.DataFrame:
    out = tracks.copy()
    for column in columns:
        values = out[column].to_numpy(dtype=float)
        smoothed = smooth_series(
            values,
            window_length=window_length,
            polyorder=polyorder,
        )
        out[f"{column}_smooth"] = smoothed
    return out
