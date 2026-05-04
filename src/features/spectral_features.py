"""
PSD (Power Spectral Density) feature extraction via Welch's method.

For each (window, nperseg, overlap_ratio) combination, 7 features are computed:
    - 5 band powers: delta (0.5-4 Hz), theta (4-8), alpha (8-13),
                     beta (13-30), gamma (30-45)
    - spectral_edge_95: frequency below which 95% of total power resides
    - dominant_freq: frequency at maximum PSD

Grid: 4 windows × 4 nperseg × 3 overlap = 48 combinations → 48 × 7 = 336 rows.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sig

from src.features.grid_utils import run_multi_feature_grid

# Standard EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Bonn dataset sampling rate
DEFAULT_FS = 173.61


def compute_psd_features(
    signal_arr: np.ndarray,
    window: str,
    nperseg: int,
    overlap_ratio: float,
    fs: float = DEFAULT_FS,
) -> dict[str, float]:
    """Compute PSD-based spectral features for a single EEG segment.

    Args:
        signal_arr: 1-D array of EEG amplitudes.
        window: Window function name ('hamming', 'hanning', 'blackman',
                'rectangular').  'rectangular' is mapped to 'boxcar'.
        nperseg: Length of each Welch segment in samples.
        overlap_ratio: Fractional overlap (0-1). noverlap = ratio * nperseg.
        fs: Sampling frequency in Hz.

    Returns:
        Dict with 7 entries: 5 band powers + spectral_edge_95 + dominant_freq.
    """
    # Map user-friendly name to scipy window name
    _WINDOW_MAP = {"rectangular": "boxcar", "hanning": "hann"}
    win = _WINDOW_MAP.get(window, window)
    noverlap = int(overlap_ratio * nperseg)

    try:
        freqs, psd = sig.welch(
            signal_arr, fs=fs, window=win,
            nperseg=nperseg, noverlap=noverlap,
        )
    except Exception as e:
        warnings.warn(
            f"Welch PSD failed (window={window}, nperseg={nperseg}, "
            f"overlap={overlap_ratio}): {e}"
        )
        return _nan_features()

    features: dict[str, float] = {}

    # Band powers (integral of PSD over each band via trapezoidal rule)
    for band_name, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        if mask.any():
            features[band_name] = float(np.trapz(psd[mask], freqs[mask]))
        else:
            features[band_name] = 0.0

    # Spectral edge frequency (95th percentile)
    features["spectral_edge_95"] = _spectral_edge(freqs, psd, 0.95)

    # Dominant frequency
    features["dominant_freq"] = float(freqs[np.argmax(psd)])

    return features


def _spectral_edge(freqs: np.ndarray, psd: np.ndarray, ratio: float) -> float:
    """Find the frequency below which *ratio* fraction of total power lies.

    Args:
        freqs: Frequency array from Welch.
        psd: PSD array from Welch.
        ratio: Cumulative power fraction (e.g. 0.95).

    Returns:
        Edge frequency in Hz, or np.nan if PSD is all zeros.
    """
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total == 0:
        return np.nan
    idx = np.searchsorted(cumulative, ratio * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _nan_features() -> dict[str, float]:
    """Return NaN-filled feature dict."""
    feats = {band: np.nan for band in BANDS}
    feats["spectral_edge_95"] = np.nan
    feats["dominant_freq"] = np.nan
    return feats


def compute_psd_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run PSD feature extraction over all (window, nperseg, overlap) combos.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'window', 'nperseg', 'overlap_ratio'.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return run_multi_feature_grid(
        signals, grid,
        param_keys=["window", "nperseg", "overlap_ratio"],
        compute_fn=compute_psd_features,
        feature_name="psd",
        checkpoint_path=checkpoint_path,
    )
