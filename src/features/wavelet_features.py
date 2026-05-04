"""
DWT (Discrete Wavelet Transform) feature extraction with hyperparameter grid sweep.

For each (wavelet, level) combination, the signal is decomposed into (level+1)
sub-bands via pywt.wavedec.  Three statistics are computed per sub-band:
    - energy:  sum(coeff ** 2)
    - std:     standard deviation of coefficients
    - entropy: Shannon entropy of normalised squared coefficients

Grid: 9 wavelets × 4 levels = 36 combinations.
Output rows per combination: (level+1) sub-bands × 3 stats.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

from src.features.grid_utils import run_multi_feature_grid


def compute_dwt_features(signal: np.ndarray, wavelet: str, level: int) -> dict[str, float]:
    """Compute DWT sub-band statistics for a single EEG segment.

    Args:
        signal: 1-D array of EEG amplitudes.
        wavelet: Wavelet family name (e.g. 'db4', 'sym2', 'coif1').
        level: Decomposition level (3–6).

    Returns:
        Dict mapping feature names like 'cA3_energy', 'cD2_std' to float values.
        Returns NaN-filled dict on failure.
    """
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
    except Exception as e:
        warnings.warn(f"DWT failed (wavelet={wavelet}, level={level}): {e}")
        # Return NaN features for expected sub-band count
        return _nan_features(level)

    features: dict[str, float] = {}

    for i, c in enumerate(coeffs):
        # Name sub-bands: cA{level} for approximation, cD{level-i+1} for details
        if i == 0:
            band_name = f"cA{level}"
        else:
            band_name = f"cD{level - i + 1}"

        # Energy
        energy = float(np.sum(c ** 2))
        features[f"{band_name}_energy"] = energy

        # Standard deviation
        features[f"{band_name}_std"] = float(np.std(c))

        # Shannon entropy of normalised squared coefficients
        features[f"{band_name}_entropy"] = _shannon_entropy(c)

    return features


def _shannon_entropy(coeffs: np.ndarray) -> float:
    """Compute Shannon entropy of normalised energy distribution.

    Args:
        coeffs: 1-D wavelet coefficient array.

    Returns:
        Entropy value (non-negative float), or np.nan on failure.
    """
    sq = coeffs ** 2
    total = np.sum(sq)
    if total == 0:
        return 0.0
    p = sq / total
    # Avoid log(0) by filtering out zeros
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _nan_features(level: int) -> dict[str, float]:
    """Return NaN-filled feature dict for a given decomposition level."""
    features = {}
    bands = [f"cA{level}"] + [f"cD{level - i}" for i in range(level)]
    for band in bands:
        for stat in ["energy", "std", "entropy"]:
            features[f"{band}_{stat}"] = np.nan
    return features


def compute_dwt_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run DWT feature extraction over all (wavelet, level) combinations.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'wavelet' and 'level'.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return run_multi_feature_grid(
        signals, grid,
        param_keys=["wavelet", "level"],
        compute_fn=compute_dwt_features,
        feature_name="dwt",
        checkpoint_path=checkpoint_path,
    )
