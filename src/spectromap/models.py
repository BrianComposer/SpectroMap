from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AnalysisResult:
    audio_path: Path
    sample_rate: int
    waveform: np.ndarray
    time_axis: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    spectrogram_db: np.ndarray
    peak_mask: np.ndarray
    peak_values_db: np.ndarray
    melody_points: np.ndarray
    normalized_representation: np.ndarray


@dataclass(slots=True)
class PairwiseResult:
    file_a: str
    file_b: str
    distance: float
    similarity: float
