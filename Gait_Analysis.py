"""Command-line runner for gait analysis without Jupyter dependencies."""

import argparse
from pathlib import Path

from gait_pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gait analysis on one or more videos and export CSV metrics."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="One or more input video files (e.g., 'brandon_01_RL (1).MOV').",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching videos (e.g., brandon_01_RL brandon_02_LR).",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for generated CSV files.",
    )
    return parser.parse_args()


def default_label_from_filename(video: str) -> str | None:
    name = Path(video).stem
    if "brandon_01_RL" in name:
        return "brandon_01_RL"
    if "brandon_02_LR" in name:
        return "brandon_02_LR"
    return None


def main() -> None:
    args = parse_args()

    labels = args.labels or []
    if labels and len(labels) != len(args.videos):
        raise ValueError("If --labels is provided, it must match the number of videos.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, video in enumerate(args.videos):
        label = labels[idx] if labels else default_label_from_filename(video)
        out_name = f"{Path(video).stem}_metrics.csv"
        out_csv = out_dir / out_name

        metrics, evaluation = run(video, str(out_csv), label)

        print(f"\nProcessed: {video}")
        print(f"Output: {out_csv}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        if evaluation is not None:
            print("\nGround-truth comparison:")
            print(evaluation.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
