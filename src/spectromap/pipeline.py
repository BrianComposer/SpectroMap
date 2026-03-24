from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from .audio import compute_spectrogram, load_wav
from .config import SpectroMapConfig
from .io_utils import ensure_output_folders, save_comparison_tables, save_constellation_file
from .logging_utils import configure_logger
from .melody import extract_melody_contour
from .models import AnalysisResult
from .peaks import detect_constellation_peaks
from .plotting import save_constellation_plot, save_melody_plot, save_spectrogram_plot
from .similarity import compare_representations
from .plotting import plot_similarity_matrix

class SpectroMapPipeline:
    def __init__(self, config: SpectroMapConfig | None = None) -> None:
        self.config = config or SpectroMapConfig()
        self.logger = configure_logger(log_file=self.config.log_file)

    def analyze_file(self, audio_path: Path, output_dir: Path, save_plots: bool = True) -> AnalysisResult:
        paths = ensure_output_folders(output_dir)

        waveform, sample_rate = load_wav(audio_path)
        freqs, times, spectrogram_db = compute_spectrogram(waveform, sample_rate, self.config)
        peak_mask, peak_values_db = detect_constellation_peaks(spectrogram_db, self.config)
        melody_points, normalized_representation = extract_melody_contour(
            spectrogram_db=spectrogram_db,
            freqs=freqs,
            times=times,
            peak_mask=peak_mask,
            config=self.config,
        )

        if self.config.generate_constellation_files:
            constellation_file = paths["constellations"] / f"{audio_path.stem}.cns"
            save_constellation_file(
                output_file=constellation_file,
                peak_mask=peak_mask,
                times=times,
                freqs=freqs,
                spectrogram_db=spectrogram_db,
            )

        if save_plots:
            save_spectrogram_plot(
                output_file=paths["plots_spectrograms"] / f"{audio_path.stem}_spectrogram.png",
                freqs=freqs,
                times=times,
                spectrogram_db=spectrogram_db,
                title=f"Spectrogram | {audio_path.stem}",
            )
            save_constellation_plot(
                output_file=paths["plots_constellations"] / f"{audio_path.stem}_constellation.png",
                freqs=freqs,
                times=times,
                spectrogram_db=spectrogram_db,
                peak_mask=peak_mask,
                title=f"Constellation map | {audio_path.stem}",
            )
            save_melody_plot(
                output_file=paths["plots_melody"] / f"{audio_path.stem}_melody.png",
                freqs=freqs,
                times=times,
                spectrogram_db=spectrogram_db,
                melody_points=melody_points,
                title=f"Melody extraction | {audio_path.stem}",
            )

        self.logger.info("Analyzed %s", audio_path.name)

        return AnalysisResult(
            audio_path=audio_path,
            sample_rate=sample_rate,
            waveform=waveform,
            time_axis=times,
            freqs=freqs,
            times=times,
            spectrogram_db=spectrogram_db,
            peak_mask=peak_mask,
            peak_values_db=peak_values_db,
            melody_points=melody_points,
            normalized_representation=normalized_representation,
        )

    def compare_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        save_plots: bool | None = None,
    ) -> pd.DataFrame:
        save_plots = self.config.save_plots if save_plots is None else save_plots

        input_dir = Path(input_dir).resolve()

        wav_files = sorted(
            [p for p in input_dir.iterdir() if p.suffix.lower() == ".wav"]
        )
        if len(wav_files) < 2:
            raise ValueError("At least two WAV files are required to compare similarities.")

        analyses = {}
        for wav_file in wav_files:
            analyses[wav_file.name] = self.analyze_file(
                audio_path=wav_file,
                output_dir=output_dir,
                save_plots=save_plots,
            )

        rows = []
        for file_a, file_b in combinations(sorted(analyses.keys()), 2):
            rep_a = analyses[file_a].normalized_representation
            rep_b = analyses[file_b].normalized_representation
            distance, similarity = compare_representations(
                rep_a,
                rep_b,
                sigma_ratio=self.config.sorting_sigma_ratio,
                gamma=self.config.similarity_gamma,
            )
            rows.append(
                {
                    "file_a": file_a,
                    "file_b": file_b,
                    "distance": distance,
                    "similarity": similarity,
                }
            )

        results_df = pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)
        ensure_output_folders(output_dir)
        save_comparison_tables(results_df=results_df, output_dir=output_dir)
        self.logger.info("Computed %d pairwise comparisons", len(results_df))

        # --- Plot similarity matrix ---
        plot_similarity_matrix(
            results_df,
            output_dir / "plots" / "similarity_matrix.png"
        )

        return results_df
