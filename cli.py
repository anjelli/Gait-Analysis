from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import AnalysisConfig, EventConfig, SmoothingConfig
from .pipeline import GaitAnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gait-analyze",
        description="Research-grade monocular gait analysis pipeline.",
    )
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--meters-per-pixel", type=float, default=None)
    p.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--confidence", type=float, default=0.30)
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--polyorder", type=int, default=3)
    p.add_argument("--min-stride-s", type=float, default=0.70)
    p.add_argument("--min-step-s", type=float, default=0.30)
    p.add_argument("--heel-prominence-px", type=float, default=6.0)
    p.add_argument("--toe-prominence-px", type=float, default=6.0)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    config = AnalysisConfig(
        confidence_threshold=args.confidence,
        model_complexity=args.model_complexity,
        smoothing=SmoothingConfig(
            window_length=args.smooth_window,
            polyorder=args.polyorder,
        ),
        events=EventConfig(
            min_stride_interval_s=args.min_stride_s,
            min_step_interval_s=args.min_step_s,
            heel_prominence_px=args.heel_prominence_px,
            toe_prominence_px=args.toe_prominence_px,
        ),
        output_dir=args.out_dir,
    )

    pipeline = GaitAnalysisPipeline(config)

    for video in args.videos:
        label = video.stem.replace(" ", "_")
        pipeline.run(
            video,
            label=label,
            meters_per_pixel=args.meters_per_pixel,
        )


if __name__ == "__main__":
    main()
