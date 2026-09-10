from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .events import detect_gait_events, event_indicator_frame, pair_gait_cycles
from .metrics import GaitMetrics, compare_to_ground_truth, compute_gait_metrics
from .pose import MediaPipePoseExtractor, validate_tracks
from .signal import smooth_landmark_columns


LOGGER = logging.getLogger(__name__)


class GaitAnalysisPipeline:
    """End-to-end research pipeline for monocular sagittal-plane gait analysis."""

    def __init__(self, config: Optional[AnalysisConfig] = None) -> None:
        self.config = config or AnalysisConfig()
        self.config.validate()

    def run(
        self,
        video_path: Path,
        *,
        label: Optional[str] = None,
        meters_per_pixel: Optional[float] = None,
        ground_truth: Optional[Dict[str, float]] = None,
        output_dir: Optional[Path] = None,
    ) -> GaitMetrics:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        label = label or video_path.stem.replace(" ", "_")
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Extracting pose from %s", video_path)
        extractor = MediaPipePoseExtractor(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.confidence_threshold,
            min_tracking_confidence=self.config.confidence_threshold,
        )
        tracks, video_info = extractor.extract(
            video_path,
            visibility_threshold=self.config.confidence_threshold,
        )
        validate_tracks(tracks)

        smooth_cols = [
            c for c in tracks.columns
            if c.endswith("_x_px") and c.startswith(("pelvis", "left_", "right_"))
        ]

        LOGGER.info("Smoothing %d coordinate channels", len(smooth_cols))
        tracks = smooth_landmark_columns(
            tracks,
            smooth_cols,
            window_length=self.config.smoothing.window_length,
            polyorder=self.config.smoothing.polyorder,
        )

        # Event detection requires smoothed pelvis/heel/toe signals.
        processed, events, direction = detect_gait_events(
            tracks,
            min_stride_interval_s=self.config.events.min_stride_interval_s,
            min_step_interval_s=self.config.events.min_step_interval_s,
            heel_prominence=self.config.events.heel_prominence_px,
            toe_prominence=self.config.events.toe_prominence_px,
        )

        cycles = pair_gait_cycles(
            processed,
            events,
            min_toe_off_fraction=self.config.events.min_toe_off_fraction_of_cycle,
            max_toe_off_fraction=self.config.events.max_toe_off_fraction_of_cycle,
        )

        metrics = compute_gait_metrics(
            processed,
            cycles,
            events,
            meters_per_pixel=meters_per_pixel,
        )

        evaluation = None
        if ground_truth:
            evaluation = compare_to_ground_truth(metrics, ground_truth)

        self._save(
            out_dir=out_dir,
            label=label,
            tracks=processed,
            events=events,
            cycles=cycles,
            metrics=metrics,
            evaluation=evaluation,
            direction=direction,
            meters_per_pixel=meters_per_pixel,
            video_info=video_info.__dict__,
        )

        LOGGER.info(
            "Completed %s: %d valid cycles, cadence=%s",
            label,
            metrics.valid_cycles,
            metrics.cadence_spm,
        )

        return metrics

    @staticmethod
    def _save(
        *,
        out_dir: Path,
        label: str,
        tracks: pd.DataFrame,
        events: Dict[str, np.ndarray],
        cycles: pd.DataFrame,
        metrics: GaitMetrics,
        evaluation: Optional[pd.DataFrame],
        direction: float,
        meters_per_pixel: Optional[float],
        video_info: dict,
    ) -> None:
        if True:
            out_dir.mkdir(parents=True, exist_ok=True)

        event_df = event_indicator_frame(len(tracks), events)
        event_df.insert(1, "time_s", tracks["time_s"].to_numpy())

        event_df.to_csv(out_dir / f"{label}_events.csv", index=False)
        cycles.to_csv(out_dir / f"{label}_cycles.csv", index=False)
        tracks.to_csv(out_dir / f"{label}_tracks.csv", index=False)

        payload = {
            "label": label,
            "video": video_info,
            "walking_direction": direction,
            "meters_per_pixel": meters_per_pixel,
            "metrics": metrics.to_dict(),
        }

        if evaluation is not None and not evaluation.empty:
            payload["evaluation"] = evaluation.to_dict(orient="records")

        (out_dir / f"{label}_summary.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
