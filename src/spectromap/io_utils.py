from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_output_folders(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "constellations": output_dir / "constellations",
        "results": output_dir / "results",
        "plots": output_dir / "plots",
        "plots_spectrograms": output_dir / "plots" / "spectrograms",
        "plots_constellations": output_dir / "plots" / "constellations",
        "plots_melody": output_dir / "plots" / "melody",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def save_constellation_file(
    output_file: Path,
    peak_mask: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    spectrogram_db: np.ndarray,
) -> None:
    row_idx, col_idx = np.where(peak_mask)
    values = np.column_stack(
        [
            times[col_idx],
            freqs[row_idx],
            spectrogram_db[row_idx, col_idx],
        ]
    )
    header = "time_seconds frequency_hz power_db"
    np.savetxt(output_file, values, fmt="%.8f", header=header, comments="")


def save_comparison_tables(results_df: pd.DataFrame, output_dir: Path) -> None:
    results_path = output_dir / "results"

    distances = results_df[["file_a", "file_b", "distance"]].copy()
    similarities = results_df[["file_a", "file_b", "similarity"]].copy()
    ranked = results_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    distances.to_csv(results_path / "pairwise_distances.csv", index=False)
    similarities.to_csv(results_path / "pairwise_similarities.csv", index=False)
    ranked.to_csv(results_path / "ranked_similarities.csv", index=False)
