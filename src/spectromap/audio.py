from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .config import SpectroMapConfig


def load_wav(file_path: Path) -> tuple[np.ndarray, int]:
    sample_rate, waveform = wavfile.read(str(file_path))

    waveform = np.asarray(waveform, dtype=np.float64)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    max_abs = np.max(np.abs(waveform))
    if max_abs > 0:
        waveform = waveform / max_abs

    return waveform, int(sample_rate)


def power_to_db(power: np.ndarray, amin: float = 1e-12) -> np.ndarray:
    safe = np.maximum(power, amin)
    return 10.0 * np.log10(safe)


def compute_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    config: SpectroMapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, power = signal.spectrogram(
        waveform,
        fs=sample_rate,
        window=config.window,
        nperseg=config.fft_size,
        noverlap=config.fft_size - config.hop_size,
        nfft=config.fft_size,
        mode="magnitude",
        scaling="spectrum",
    )

    power = np.square(power)
    spectrogram_db = power_to_db(power)
    return freqs, times, spectrogram_db


def hz_to_midi(freq_hz: np.ndarray) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    out = np.full(freq_hz.shape, np.nan, dtype=np.float64)
    valid = freq_hz > 0
    out[valid] = 69.0 + 12.0 * np.log2(freq_hz[valid] / 440.0)
    return out
