from pathlib import Path
from spectromap.config import SpectroMapConfig
from spectromap.pipeline import SpectroMapPipeline


def main() -> None:

    ROOT = Path(__file__).resolve().parents[1]
    input_dir = ROOT / "data" / "input_audio"
    output_dir = ROOT / "outputs"

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    config = SpectroMapConfig(
        fft_size=4096,
        hop_size=512,
        min_frequency=80.0,
        max_frequency=2000.0,
        resample_length=128,
        adaptive_threshold_db=6.0,
        save_plots=True,
    )

    pipeline = SpectroMapPipeline(config=config)

    results = pipeline.compare_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        save_plots=True,
    )

    print("\nRanked similarities:\n")
    print(results[["file_a", "file_b", "similarity"]].to_string(index=False))

if __name__ == "__main__":
    main()