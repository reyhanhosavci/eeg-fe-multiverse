"""
Temporal and complexity feature extraction: Hjorth parameters and Hurst exponent.

Hjorth parameters (6 combinations):
    - Activity  = var(x)
    - Mobility  = sqrt(var(dx) / var(x))
    - Complexity = Mobility(dx) / Mobility(x)
    window_size_sec in [0.5, 1.0, 2.0, 3.0, 5.0, 'full']

Hurst exponent (12 combinations):
    - method in [rescaled_range, dfa, wavelet]
    - min_window in [4, 8, 16, 32]
"""

import warnings
from pathlib import Path

import antropy as ant
import numpy as np
import pandas as pd
import pywt

from src.features.grid_utils import run_multi_feature_grid, run_scalar_grid

# Bonn dataset sampling rate
DEFAULT_FS = 173.61


# ---------------------------------------------------------------------------
# Hjorth parameters
# ---------------------------------------------------------------------------


def compute_hjorth_features(
    signal: np.ndarray, window_size_sec: float | str, fs: float = DEFAULT_FS
) -> dict[str, float]:
    """Compute Hjorth activity, mobility, and complexity for a single segment.

    When window_size_sec is 'full', the entire segment is treated as one
    window.  Otherwise the segment is split into non-overlapping windows of
    the given duration and the Hjorth parameters are averaged across windows.

    Args:
        signal: 1-D array of EEG amplitudes.
        window_size_sec: Window length in seconds, or 'full'.
        fs: Sampling frequency in Hz.

    Returns:
        Dict with keys 'activity', 'mobility', 'complexity'.
    """
    try:
        if str(window_size_sec) == "full":
            return _hjorth_single(signal)

        win_samples = int(float(window_size_sec) * fs)
        if win_samples < 4:
            warnings.warn(f"Hjorth window too short ({win_samples} samples), using full")
            return _hjorth_single(signal)

        # Split into non-overlapping windows, discard incomplete tail
        n_windows = len(signal) // win_samples
        if n_windows == 0:
            return _hjorth_single(signal)

        activities, mobilities, complexities = [], [], []
        for i in range(n_windows):
            chunk = signal[i * win_samples : (i + 1) * win_samples]
            h = _hjorth_single(chunk)
            activities.append(h["activity"])
            mobilities.append(h["mobility"])
            complexities.append(h["complexity"])

        return {
            "activity": float(np.nanmean(activities)),
            "mobility": float(np.nanmean(mobilities)),
            "complexity": float(np.nanmean(complexities)),
        }
    except Exception as e:
        warnings.warn(f"Hjorth failed (window={window_size_sec}): {e}")
        return {"activity": np.nan, "mobility": np.nan, "complexity": np.nan}


def _hjorth_single(signal: np.ndarray) -> dict[str, float]:
    """Compute Hjorth parameters for a single window.

    Args:
        signal: 1-D array.

    Returns:
        Dict with activity, mobility, complexity.
    """
    dx = np.diff(signal)
    ddx = np.diff(dx)

    var_x = float(np.var(signal))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))

    activity = var_x

    if var_x == 0:
        return {"activity": activity, "mobility": np.nan, "complexity": np.nan}

    mobility = float(np.sqrt(var_dx / var_x))

    if var_dx == 0:
        return {"activity": activity, "mobility": mobility, "complexity": np.nan}

    mobility_dx = float(np.sqrt(var_ddx / var_dx))
    complexity = mobility_dx / mobility

    return {
        "activity": activity,
        "mobility": mobility,
        "complexity": complexity,
    }


def compute_hjorth_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run Hjorth features over all window_size_sec values.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with key 'window_size_sec'.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return run_multi_feature_grid(
        signals, grid,
        param_keys=["window_size_sec"],
        compute_fn=compute_hjorth_features,
        feature_name="hjorth",
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Hurst exponent
# ---------------------------------------------------------------------------


def compute_hurst(signal: np.ndarray, method: str, min_window: int) -> float:
    """Compute the Hurst exponent for a single EEG segment.

    Three estimation methods are supported:
        - rescaled_range: Classical R/S analysis
        - dfa: Detrended Fluctuation Analysis (antropy)
        - wavelet: Wavelet-variance regression

    Args:
        signal: 1-D array of EEG amplitudes.
        method: Estimation method name.
        min_window: Minimum window/scale size for the estimator.

    Returns:
        Scalar Hurst exponent (typically 0–1), or np.nan on failure.
    """
    method = str(method)
    min_window = int(min_window)

    try:
        if method == "rescaled_range":
            return _hurst_rs(signal, min_window)
        elif method == "dfa":
            return _hurst_dfa(signal, min_window)
        elif method == "wavelet":
            return _hurst_wavelet(signal, min_window)
        else:
            raise ValueError(f"Unknown Hurst method: {method}")
    except Exception as e:
        warnings.warn(f"Hurst failed (method={method}, min_window={min_window}): {e}")
        return np.nan


def _hurst_rs(signal: np.ndarray, min_window: int) -> float:
    """Rescaled-range (R/S) Hurst exponent estimation.

    Divides the signal into windows of increasing size, computes the
    rescaled range for each, and fits log(R/S) vs log(n).

    Args:
        signal: 1-D array.
        min_window: Smallest window size to use.

    Returns:
        Hurst exponent as float.
    """
    N = len(signal)
    # Generate window sizes: powers of 2 from min_window to N//2
    max_k = int(np.floor(np.log2(N // 2)))
    min_k = int(np.ceil(np.log2(max(min_window, 4))))

    if min_k > max_k:
        return np.nan

    ns = [2 ** k for k in range(min_k, max_k + 1)]
    rs_values = []

    for n in ns:
        n_chunks = N // n
        if n_chunks == 0:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = signal[i * n : (i + 1) * n]
            mean_c = np.mean(chunk)
            Y = np.cumsum(chunk - mean_c)
            R = np.max(Y) - np.min(Y)
            S = np.std(chunk, ddof=0)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
        else:
            rs_values.append(np.nan)

    # Filter NaN
    valid = [(n, r) for n, r in zip(ns, rs_values) if np.isfinite(r)]
    if len(valid) < 2:
        return np.nan

    log_n = np.log([v[0] for v in valid])
    log_rs = np.log([v[1] for v in valid])
    H, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(H, 0, 1))


def _hurst_dfa(signal: np.ndarray, min_window: int) -> float:
    """DFA-based Hurst exponent via antropy.

    Args:
        signal: 1-D array.
        min_window: Minimum window size for DFA.

    Returns:
        Hurst exponent (DFA alpha exponent).
    """
    # Build window sizes for DFA
    N = len(signal)
    max_window = N // 4
    if min_window >= max_window:
        min_window = 4

    val = ant.detrended_fluctuation(signal)
    return float(np.clip(val, 0, 2))


def _hurst_wavelet(signal: np.ndarray, min_window: int) -> float:
    """Wavelet-variance based Hurst exponent estimation.

    Uses the scaling of detail-coefficient variance across DWT levels:
        log2(Var(d_j)) ≈ j * (2H + 1) + const
    so H = (slope - 1) / 2.

    Args:
        signal: 1-D array.
        min_window: Minimum number of coefficients per level to include.

    Returns:
        Hurst exponent as float.
    """
    max_level = pywt.dwt_max_level(len(signal), "db4")
    if max_level < 2:
        return np.nan

    coeffs = pywt.wavedec(signal, "db4", level=max_level)
    detail_coeffs = coeffs[1:]  # skip approximation

    # Compute variance per level, filter by min coeff count
    levels = []
    log_vars = []
    for j, d in enumerate(reversed(detail_coeffs), start=1):
        if len(d) >= min_window:
            var_d = np.var(d)
            if var_d > 0:
                levels.append(j)
                log_vars.append(np.log2(var_d))

    if len(levels) < 2:
        return np.nan

    slope, _ = np.polyfit(levels, log_vars, 1)
    H = (slope - 1) / 2.0
    return float(np.clip(H, 0, 1))


def compute_hurst_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run Hurst exponent over all (method, min_window) combinations.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'method' and 'min_window'.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return run_scalar_grid(
        signals, grid,
        param_keys=["method", "min_window"],
        compute_fn=compute_hurst,
        feature_name="hurst",
        checkpoint_path=checkpoint_path,
    )
