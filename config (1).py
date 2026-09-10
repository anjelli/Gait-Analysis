from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SmoothingConfig:
    method: Literal["savgol"] = "savgol"
    window_length: int = 11
    polyorder: int = 3


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = False
    ransac_reprojection_threshold_px: float = 3.0
    ransac_confidence: float = 0.995
    ransac_max_iterations: int = 2000
    unit_scale_to_m: float = 0.01  # centimeters -> meters


@dataclass(frozen=True)
class EventConfig:
    min_stride_interval_s: float = 0.70
    min_step_interval_s: float = 0.30
    heel_prominence_px: float = 6.0
    toe_prominence_px: float = 6.0
    max_toe_off_fraction_of_cycle: float = 0.85
    min_toe_off_fraction_of_cycle: float = 0.10


@dataclass(frozen=True)
class AnalysisConfig:
    confidence_threshold: float = 0.30
    model_complexity: int = 1
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    events: EventConfig = field(default_factory=EventConfig)
    output_dir: Path = Path("results")
    save_processed_tracks: bool = True
    save_events: bool = True
    save_cycles: bool = True
    save_json_summary: bool = True

    def validate(self) -> None:
        if self.model_complexity not in (0, 1, 2):
            raise ValueError("model_complexity must be 0, 1, or 2.")

        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be in [0, 1].")

        s = self.smoothing
        if s.window_length < 5 or s.window_length % 2 == 0:
            raise ValueError("window_length must be odd and >= 5.")
        if s.polyorder < 1 or s.polyorder >= s.window_length:
            raise ValueError("polyorder must be >= 1 and < window_length.")

        e = self.events
        if e.min_stride_interval_s <= 0 or e.min_step_interval_s <= 0:
            raise ValueError("Event intervals must be positive.")
        if e.min_toe_off_fraction_of_cycle >= e.max_toe_off_fraction_of_cycle:
            raise ValueError("Toe-off fraction bounds are invalid.")

        c = self.calibration
        if c.ransac_reprojection_threshold_px <= 0:
            raise ValueError("RANSAC threshold must be positive.")
        if not 0 < c.ransac_confidence < 1:
            raise ValueError("RANSAC confidence must be between 0 and 1.")
        if c.ransac_max_iterations < 100:
            raise ValueError("RANSAC max iterations must be >= 100.")
        if c.unit_scale_to_m <= 0:
            raise ValueError("unit_scale_to_m must be positive.")
