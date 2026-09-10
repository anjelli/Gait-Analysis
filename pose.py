from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
import pandas as pd


# Nine biomechanically useful points:
# pelvis midpoint is derived from the bilateral hip landmarks.
MEDIAPIPE_LANDMARKS = {
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_toe": 31,
    "right_toe": 32,
}


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


class MediaPipePoseExtractor:
    """Extract 2-D lower-extremity trajectories from monocular video."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    @staticmethod
    def video_info(video_path: Path) -> VideoInfo:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        if fps <= 0:
            raise ValueError(f"Invalid FPS reported by video: {fps}")

        return VideoInfo(
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_s=frame_count / fps if frame_count > 0 else 0.0,
        )

    def extract(self, video_path: Path, visibility_threshold: float = 0.30) -> tuple[pd.DataFrame, VideoInfo]:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is required for pose extraction. "
                "Install dependencies from requirements.txt."
            ) from exc

        info = self.video_info(video_path)
        pose_api = mp.solutions.pose

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        records = []
        try:
            with pose_api.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as pose:
                frame_idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)

                    row = {
                        "frame": frame_idx,
                        "time_s": frame_idx / info.fps,
                    }

                    if result.pose_landmarks is None:
                        for name in MEDIAPIPE_LANDMARKS:
                            row[f"{name}_x_px"] = np.nan
                            row[f"{name}_y_px"] = np.nan
                            row[f"{name}_visibility"] = 0.0
                        row.update(self._missing_pelvis_row())
                    else:
                        lms = result.pose_landmarks.landmark
                        left_hip = lms[23]
                        right_hip = lms[24]

                        hip_ok = (
                            float(getattr(left_hip, "visibility", 0.0)) >= visibility_threshold
                            and float(getattr(right_hip, "visibility", 0.0)) >= visibility_threshold
                        )

                        if hip_ok:
                            pelvis_x = 0.5 * (left_hip.x + right_hip.x) * info.width
                            pelvis_y = 0.5 * (left_hip.y + right_hip.y) * info.height
                        else:
                            pelvis_x = pelvis_y = np.nan

                        for name, idx in MEDIAPIPE_LANDMARKS.items():
                            lm = lms[idx]
                            visibility = float(getattr(lm, "visibility", 0.0))
                            if visibility >= visibility_threshold:
                                row[f"{name}_x_px"] = float(lm.x * info.width)
                                row[f"{name}_y_px"] = float(lm.y * info.height)
                            else:
                                row[f"{name}_x_px"] = np.nan
                                row[f"{name}_y_px"] = np.nan
                            row[f"{name}_visibility"] = visibility

                        row["pelvis_x_px"] = float(pelvis_x) if np.isfinite(pelvis_x) else np.nan
                        row["pelvis_y_px"] = float(pelvis_y) if np.isfinite(pelvis_y) else np.nan
                        row["pelvis_visibility"] = (
                            0.5 * (float(left_hip.visibility) + float(right_hip.visibility))
                            if hip_ok else 0.0
                        )

                    records.append(row)
                    frame_idx += 1
        finally:
            cap.release()

        tracks = pd.DataFrame.from_records(records)
        if tracks.empty:
            raise RuntimeError("Pose extraction produced no frames.")

        return tracks, info

    @staticmethod
    def _missing_pelvis_row() -> Dict[str, float]:
        return {
            "pelvis_x_px": np.nan,
            "pelvis_y_px": np.nan,
            "pelvis_visibility": 0.0,
        }


def validate_tracks(tracks: pd.DataFrame, required_fraction: float = 0.50) -> None:
    """Fail early when too little pose data are present."""
    x_cols = [c for c in tracks.columns if c.endswith("_x_px")]
    if not x_cols:
        raise ValueError("No coordinate columns found.")

    valid_fraction = tracks[x_cols].notna().mean(axis=1).mean()
    if valid_fraction < required_fraction:
        raise RuntimeError(
            f"Insufficient pose coverage: {valid_fraction:.1%}. "
            f"Required at least {required_fraction:.1%}."
        )
