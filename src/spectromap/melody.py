from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from .audio import hz_to_midi
from .config import SpectroMapConfig


def _resample_contour(time_norm: np.ndarray, values: np.ndarray, target_length: int) -> np.ndarray:
    if len(time_norm) == 0:
        grid = np.linspace(0.0, 1.0, target_length)
        return np.column_stack([grid, np.zeros_like(grid)])

    if len(time_norm) == 1:
        grid = np.linspace(0.0, 1.0, target_length)
        return np.column_stack([grid, np.full_like(grid, values[0], dtype=np.float64)])

    grid = np.linspace(0.0, 1.0, target_length)
    interp = np.interp(grid, time_norm, values)
    return np.column_stack([grid, interp])


def _normalize_pitch(midi_values: np.ndarray, method: str = "median") -> np.ndarray:
    if len(midi_values) == 0:
        return midi_values

    if method == "mean":
        reference = float(np.mean(midi_values))
    else:
        reference = float(np.median(midi_values))

    return midi_values - reference


def extract_melody_contour(
    spectrogram_db: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    peak_mask: np.ndarray,
    config: SpectroMapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Melody extraction by time-wise candidate selection with continuity regularization.

    For each time frame, candidate frequencies are taken from detected peaks inside
    a configurable frequency band. If no peak is present, the strongest raw spectral
    bin in the valid band is used as fallback. The chosen path balances spectral
    energy and pitch continuity.
    """
    valid_rows = (freqs >= config.min_frequency) & (freqs <= config.max_frequency)

    if not np.any(valid_rows):
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, _resample_contour(np.array([]), np.array([]), config.resample_length)

    band_freqs = freqs[valid_rows]
    band_spec = spectrogram_db[valid_rows, :]
    band_peaks = peak_mask[valid_rows, :]

    selected_times: list[float] = []
    selected_midi: list[float] = []

    prev_midi: float | None = None

    for frame_idx in range(band_spec.shape[1]):
        column_db = band_spec[:, frame_idx]
        column_peaks = band_peaks[:, frame_idx]

        candidate_idx = np.flatnonzero(column_peaks)

        if candidate_idx.size == 0:
            fallback_idx = int(np.argmax(column_db))
            candidate_idx = np.array([fallback_idx], dtype=int)

        candidate_freqs = band_freqs[candidate_idx]
        candidate_midi = hz_to_midi(candidate_freqs)
        candidate_energy = column_db[candidate_idx]

        if prev_midi is None:
            scores = candidate_energy * config.melody_energy_weight
        else:
            jump_cost = np.abs(candidate_midi - prev_midi)
            scores = (candidate_energy * config.melody_energy_weight) - (
                config.melody_jump_penalty * jump_cost
            )

        best_local = int(np.argmax(scores))
        best_midi = float(candidate_midi[best_local])

        selected_times.append(float(times[frame_idx]))
        selected_midi.append(best_midi)
        prev_midi = best_midi

    midi_values = np.asarray(selected_midi, dtype=np.float64)
    time_values = np.asarray(selected_times, dtype=np.float64)

    kernel = config.melody_smooth_window
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(3, kernel)

    if len(midi_values) >= kernel:
        midi_values = medfilt(midi_values, kernel_size=kernel)

    if len(time_values) > 1:
        duration = time_values[-1] - time_values[0]
        if duration <= 0:
            time_norm = np.linspace(0.0, 1.0, len(time_values))
        else:
            time_norm = (time_values - time_values[0]) / duration
    else:
        time_norm = np.array([0.0], dtype=np.float64)

    normalized_pitch = _normalize_pitch(midi_values, method=config.transposition_reference)

    melody_points = np.column_stack([time_values, midi_values])
    normalized_representation = _resample_contour(
        time_norm=time_norm,
        values=normalized_pitch,
        target_length=config.resample_length,
    )
    return melody_points, normalized_representation
