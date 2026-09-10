from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HomographyCalibration:
    H: np.ndarray
    inlier_mask: np.ndarray
    reprojection_rmse: float
    source_points: np.ndarray
    target_points: np.ndarray
    unit_scale_to_m: float

    def transform(self, xy_px: np.ndarray) -> np.ndarray:
        xy_px = np.asarray(xy_px, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(xy_px, self.H).reshape(-1, 2)
        return transformed * self.unit_scale_to_m

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            H=self.H,
            inlier_mask=self.inlier_mask,
            reprojection_rmse=self.reprojection_rmse,
            source_points=self.source_points,
            target_points=self.target_points,
            unit_scale_to_m=self.unit_scale_to_m,
        )

    @classmethod
    def load(cls, path: Path) -> "HomographyCalibration":
        data = np.load(path)
        return cls(
            H=data["H"],
            inlier_mask=data["inlier_mask"],
            reprojection_rmse=float(data["reprojection_rmse"]),
            source_points=data["source_points"],
            target_points=data["target_points"],
            unit_scale_to_m=float(data["unit_scale_to_m"]),
        )


def estimate_homography(
    pixel_points: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]],
    *,
    unit_scale_to_m: float = 0.01,
    ransac_reprojection_threshold_px: float = 3.0,
    confidence: float = 0.995,
    max_iterations: int = 2000,
) -> HomographyCalibration:
    """Estimate a robust planar homography using OpenCV RANSAC."""
    src = np.asarray(pixel_points, dtype=np.float32)
    dst = np.asarray(world_points, dtype=np.float32)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("pixel_points and world_points must both have shape (N, 2).")
    if len(src) < 4:
        raise ValueError("At least four point correspondences are required.")

    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reprojection_threshold_px,
        maxIters=max_iterations,
        confidence=confidence,
    )
    if H is None or mask is None:
        raise RuntimeError("Homography estimation failed.")

    transformed = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(transformed - dst, axis=1)
    inliers = mask.ravel().astype(bool)

    if not np.any(inliers):
        raise RuntimeError("RANSAC rejected all calibration points.")

    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))

    return HomographyCalibration(
        H=H,
        inlier_mask=inliers,
        reprojection_rmse=rmse,
        source_points=src,
        target_points=dst,
        unit_scale_to_m=unit_scale_to_m,
    )


def calibrate_checkerboard(
    image: np.ndarray,
    checkerboard_shape: tuple[int, int],
    square_size_cm: float,
    *,
    image_preprocess: bool = True,
    **homography_kwargs,
) -> HomographyCalibration:
    """
    Detect checkerboard corners and build a planar calibration.

    checkerboard_shape is (columns, rows) of internal corners.
    """
    if image is None:
        raise ValueError("image must not be None.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if image_preprocess:
        gray = cv2.equalizeHist(gray)

    pattern_size = tuple(checkerboard_shape)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK,
    )
    if not found or corners is None:
        raise RuntimeError("Checkerboard corners could not be detected.")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-4,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    cols, rows = checkerboard_shape
    world = np.zeros((rows * cols, 2), dtype=np.float32)
    world[:, 0] = np.tile(np.arange(cols), rows) * square_size_cm
    world[:, 1] = np.repeat(np.arange(rows), cols) * square_size_cm

    return estimate_homography(
        corners.reshape(-1, 2),
        world,
        unit_scale_to_m=0.01,
        **homography_kwargs,
    )


def apply_calibration_to_tracks(
    tracks: pd.DataFrame,
    calibration: HomographyCalibration,
) -> pd.DataFrame:
    """Add calibrated x/y coordinates in metres for every available landmark."""
    out = tracks.copy()
    for prefix in ("pelvis", "left_knee", "right_knee", "left_ankle",
                   "right_ankle", "left_heel", "right_heel", "left_toe", "right_toe"):
        x_col, y_col = f"{prefix}_x_px", f"{prefix}_y_px"
        x_world = np.full(len(out), np.nan)
        y_world = np.full(len(out), np.nan)

        valid = out[[x_col, y_col]].notna().all(axis=1).to_numpy()
        if np.any(valid):
            pts = out.loc[valid, [x_col, y_col]].to_numpy(dtype=np.float32)
            world = calibration.transform(pts)
            x_world[valid] = world[:, 0]
            y_world[valid] = world[:, 1]

        out[f"{prefix}_x_m"] = x_world
        out[f"{prefix}_y_m"] = y_world

    return out
