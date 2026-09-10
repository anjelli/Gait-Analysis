from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def load_summary(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_directory(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_summary.json")):
        payload = load_summary(path)
        row = {"label": payload.get("label", path.stem)}
        row.update(payload.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate gait summary JSON files.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_directory(args.results_dir)
    if summary.empty:
        raise SystemExit("No *_summary.json files found.")

    out = args.out or (args.results_dir / "gait_summary.csv")
    summary.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
