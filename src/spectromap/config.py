from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SpectroMapConfig:
    fft_size: int = 4096
    hop_size: int = 512
    window: str = "hann"

    min_frequency: float = 80.0
    max_frequency: float = 2000.0

    peak_neighborhood_time: int = 7
    peak_neighborhood_freq: int = 9
    adaptive_threshold_db: float = 6.0
    global_floor_percentile: float = 65.0
    smoothing_sigma_freq: float = 1.0
    smoothing_sigma_time: float = 1.0

    melody_jump_penalty: float = 0.35
    melody_energy_weight: float = 1.0
    melody_smooth_window: int = 5

    resample_length: int = 128
    transposition_reference: str = "median"

    similarity_gamma: float = 1.0
    sorting_sigma_ratio: float = 1 / 6

    generate_constellation_files: bool = True
    save_plots: bool = True

    log_file: Path = Path("spectromap.log")
