from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter

from .config import SpectroMapConfig


def detect_constellation_peaks(
    spectrogram_db: np.ndarray,
    config: SpectroMapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust local-peak detection on a dB spectrogram.

    Strategy:
    - smooth the spectrogram
    - detect local maxima in a fixed time-frequency neighborhood
    - use a local adaptive threshold based on a median filter
    - apply a global percentile floor to suppress noise
    """
    smoothed = gaussian_filter(
        spectrogram_db,
        sigma=(config.smoothing_sigma_freq, config.smoothing_sigma_time),
    )

    neighborhood = (
        max(3, int(config.peak_neighborhood_freq)),
        max(3, int(config.peak_neighborhood_time)),
    )

    local_max = smoothed == maximum_filter(smoothed, size=neighborhood, mode="nearest")
    local_baseline = median_filter(smoothed, size=neighborhood, mode="nearest")
    adaptive_threshold = local_baseline + config.adaptive_threshold_db
    global_floor = np.percentile(smoothed, config.global_floor_percentile)

    peak_mask = local_max & (smoothed >= adaptive_threshold) & (smoothed >= global_floor)

    peak_values_db = np.zeros_like(spectrogram_db)
    peak_values_db[peak_mask] = spectrogram_db[peak_mask]
    return peak_mask, peak_values_db
