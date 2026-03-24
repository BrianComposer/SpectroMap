from __future__ import annotations

import argparse
from pathlib import Path

from .config import SpectroMapConfig
from .pipeline import SpectroMapPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectromap", description="Constellation-based melodic similarity analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare all WAV files in a directory.")
    compare_parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing WAV files.")
    compare_parser.add_argument("--output-dir", type=Path, required=True, help="Directory where outputs will be saved.")
    compare_parser.add_argument("--fft-size", type=int, default=4096)
    compare_parser.add_argument("--hop-size", type=int, default=512)
    compare_parser.add_argument("--min-frequency", type=float, default=80.0)
    compare_parser.add_argument("--max-frequency", type=float, default=2000.0)
    compare_parser.add_argument("--resample-length", type=int, default=128)
    compare_parser.add_argument("--adaptive-threshold-db", type=float, default=6.0)
    compare_parser.add_argument("--save-plots", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compare":
        config = SpectroMapConfig(
            fft_size=args.fft_size,
            hop_size=args.hop_size,
            min_frequency=args.min_frequency,
            max_frequency=args.max_frequency,
            resample_length=args.resample_length,
            adaptive_threshold_db=args.adaptive_threshold_db,
            save_plots=args.save_plots,
        )
        pipeline = SpectroMapPipeline(config=config)
        results = pipeline.compare_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_plots=args.save_plots,
        )
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
