"""Download official Orbit Wars assets or write a local summary."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SUMMARY_TEXT = """# Orbit Wars Official Mechanics Summary

## Core rules
- Board: circular map with a central sun hazard.
- Planets/fleets are represented as tuples in official observations.
- Production resolves in the official turn-order pipeline before travel/combat resolution.
- Fleet speed uses the official launch-size/speed rule from competition README.
- Comets appear as temporary capturable opportunities and expire.
- Combat resolves by ownership/ship counts on arrivals.
- Match length is 500 turns.
- Score is final ship-equivalent control/value under official scoring.

## Preserved research references
- Discussion 697725: RL instability/cost; simulator speed and reward shape dominate.
- Discussion 701894: daily episode datasets for replay mining.
- Discussion 697413: top-10% replay datasets (Bovard).
"""


def download_competition_assets(competition: str, output_dir: Path) -> Path:
    """Run Kaggle CLI to download official competition assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(output_dir)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Kaggle CLI not found. Install and authenticate it first: "
            "https://github.com/Kaggle/kaggle-api#installation"
        ) from exc
    return output_dir


def write_official_summary(output_dir: Path) -> Path:
    """Write a local no-network official summary file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "official_summary.md"
    summary_path.write_text(SUMMARY_TEXT, encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Orbit Wars competition assets")
    parser.add_argument("--competition", default="orbit-wars", help="Competition slug")
    parser.add_argument("--output-dir", default="data/raw/competition", help="Output directory")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write local summary only without network download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.summary_only:
        summary_path = write_official_summary(output_dir)
        print(f"Summary written: {summary_path}")
        return

    downloaded_to = download_competition_assets(args.competition, output_dir)
    print(f"Competition assets downloaded to: {downloaded_to}")


if __name__ == "__main__":
    main()
