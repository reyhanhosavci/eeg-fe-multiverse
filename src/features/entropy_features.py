"""
Entropy and complexity feature extraction with full hyperparameter grid sweep.

Computes five feature families over all parameter combinations specified in the
YAML config:
    1. Sample Entropy (SampEn)      — EntropyHub
    2. Approximate Entropy (ApEn)   — EntropyHub
    3. Permutation Entropy (PermEn) — antropy
    4. Fuzzy Entropy (FuzzyEn)      — EntropyHub
    5. Lempel-Ziv Complexity (LZC)  — antropy

Each grid runner produces a DataFrame where rows = parameter combinations and
columns = segment values (seg_000 … seg_N-1).  Results are saved as CSV files
under results/tables/.

Checkpoint support: each combination is saved to disk immediately after
completion.  If a run is interrupted, restarting will skip already-computed
combinations by reading the existing CSV.
"""

import time
import warnings
from itertools import product
from pathlib import Path

import EntropyHub as EH
import antropy as ant
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Single-segment compute functions
# ---------------------------------------------------------------------------


def compute_sampen(signal: np.ndarray, m: int, r_multiplier: float) -> float:
    """Compute Sample Entropy for a single EEG segment.

    Args:
        signal: 1-D array of EEG amplitudes.
        m: Embedding dimension.
        r_multiplier: Tolerance as a fraction of signal std (r = multiplier * std).

    Returns:
        Scalar SampEn value, or np.nan on failure.
    """
    try:
        r = r_multiplier * np.std(signal)
        result = EH.SampEn(signal, m=m, r=r)
        return _sanitize(float(result[0][-1]))
    except Exception as e:
        warnings.warn(f"SampEn failed (m={m}, r_mult={r_multiplier}): {e}")
        return np.nan


def compute_apen(signal: np.ndarray, m: int, r_multiplier: float) -> float:
    """Compute Approximate Entropy for a single EEG segment.

    Args:
        signal: 1-D array of EEG amplitudes.
        m: Embedding dimension.
        r_multiplier: Tolerance as a fraction of signal std.

    Returns:
        Scalar ApEn value, or np.nan on failure.
    """
    try:
        r = r_multiplier * np.std(signal)
        result = EH.ApEn(signal, m=m, r=r)
        return _sanitize(float(result[0][-1]))
    except Exception as e:
        warnings.warn(f"ApEn failed (m={m}, r_mult={r_multiplier}): {e}")
        return np.nan


def compute_permen(signal: np.ndarray, order: int, delay: int) -> float:
    """Compute Permutation Entropy for a single EEG segment.

    Args:
        signal: 1-D array of EEG amplitudes.
        order: Embedding dimension (motif length).
        delay: Time delay between consecutive elements.

    Returns:
        Scalar normalised PermEn value, or np.nan on failure.
    """
    try:
        value = ant.perm_entropy(signal, order=order, delay=delay, normalize=True)
        return _sanitize(float(value))
    except Exception as e:
        warnings.warn(f"PermEn failed (order={order}, delay={delay}): {e}")
        return np.nan


def compute_fuzzen(
    signal: np.ndarray, m: int, r_multiplier: float, n: int
) -> float:
    """Compute Fuzzy Entropy for a single EEG segment.

    Uses EntropyHub's FuzzEn with exponential fuzzy membership function.

    Args:
        signal: 1-D array of EEG amplitudes.
        m: Embedding dimension.
        r_multiplier: Tolerance as a fraction of signal std.
        n: Fuzzy exponent (controls boundary softness).

    Returns:
        Scalar FuzzyEn value, or np.nan on failure.
    """
    try:
        r = r_multiplier * np.std(signal)
        result = EH.FuzzEn(signal, m=m, r=(r, n))
        return _sanitize(float(result[0][-1]))
    except Exception as e:
        warnings.warn(f"FuzzyEn failed (m={m}, r_mult={r_multiplier}, n={n}): {e}")
        return np.nan


def compute_lzc(signal: np.ndarray, threshold: str, multiscale: bool) -> float:
    """Compute Lempel-Ziv Complexity for a single EEG segment.

    The signal is binarised using the given threshold method, then LZC is
    computed via antropy.lziv_complexity.

    Args:
        signal: 1-D array of EEG amplitudes.
        threshold: Binarisation method — 'median', 'mean', or 'zero'.
        multiscale: If True, average LZC across [median, mean, zero] thresholds.

    Returns:
        Scalar LZC value (normalised), or np.nan on failure.
    """
    try:
        if multiscale:
            values = [
                ant.lziv_complexity(_binarise(signal, t), normalize=True)
                for t in ["median", "mean", "zero"]
            ]
            return _sanitize(float(np.mean(values)))
        else:
            binary = _binarise(signal, threshold)
            return _sanitize(float(ant.lziv_complexity(binary, normalize=True)))
    except Exception as e:
        warnings.warn(
            f"LZC failed (threshold={threshold}, multiscale={multiscale}): {e}"
        )
        return np.nan


def _binarise(signal: np.ndarray, method: str) -> np.ndarray:
    """Convert a continuous signal to a binary sequence.

    Args:
        signal: 1-D array of EEG amplitudes.
        method: 'median', 'mean', or 'zero'.

    Returns:
        Binary (0/1) array of same length.
    """
    if method == "median":
        thr = np.median(signal)
    elif method == "mean":
        thr = np.mean(signal)
    elif method == "zero":
        thr = 0.0
    else:
        raise ValueError(f"Unknown binarisation method: {method}")
    return (signal >= thr).astype(int)


def _sanitize(value: float) -> float:
    """Replace inf with NaN; pass finite values through."""
    if not np.isfinite(value):
        warnings.warn(f"Non-finite entropy value encountered: {value}")
        return np.nan
    return value


# ---------------------------------------------------------------------------
# Generic grid runner with checkpoint support
# ---------------------------------------------------------------------------


def _load_checkpoint(
    checkpoint_path: Path, param_keys: list[str]
) -> tuple[pd.DataFrame, set]:
    """Load existing checkpoint and return completed parameter combos.

    Deduplicates rows on param_keys to guard against partial writes
    or filesystem sync artefacts (e.g. Google Drive).

    Args:
        checkpoint_path: Path to the checkpoint CSV file.
        param_keys: List of parameter column names (e.g. ['m', 'r_multiplier']).

    Returns:
        Tuple of (existing_df_or_empty, set_of_completed_combo_tuples).
    """
    if not checkpoint_path.exists():
        return pd.DataFrame(), set()

    df = pd.read_csv(checkpoint_path)
    if df.empty:
        return df, set()

    # Normalise param columns to strings for reliable comparison.
    # (iterrows can silently upcast int->float when mixed with float cols)
    param_str = {k: df[k].astype(str) for k in param_keys}
    dedup_key = pd.DataFrame(param_str)

    # Remove duplicate rows (keep first) on parameter columns
    df = df.loc[~dedup_key.duplicated(keep="first")].reset_index(drop=True)
    dedup_key = dedup_key.loc[df.index].reset_index(drop=True)

    # Build a set of string-tuples for already-computed combos
    done = set()
    for i in range(len(dedup_key)):
        key = tuple(dedup_key.iloc[i])
        done.add(key)

    return df, done


def _run_grid(
    signals: np.ndarray,
    grid: dict,
    param_keys: list[str],
    compute_fn,
    feature_name: str,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generic grid runner with checkpoint support.

    Iterates over all parameter combinations from *grid*, computes
    *compute_fn* for every segment, and optionally checkpoints after
    each combination.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict mapping parameter names to lists of values.
        param_keys: Ordered list of parameter names in the grid.
        compute_fn: Callable(signal, **params) -> float.
        feature_name: Human-readable name for progress bar and timing log.
        checkpoint_path: If provided, save after each combo and skip
                         already-computed combos on restart.

    Returns:
        Tuple of (results_df, timing_info).
    """
    # Build full combo list
    param_values = [grid[k] for k in param_keys]
    all_combos = list(product(*param_values))

    n_seg = signals.shape[0]
    seg_cols = [f"seg_{i:03d}" for i in range(n_seg)]

    # Load checkpoint if available
    existing_df = pd.DataFrame()
    done_keys: set = set()
    n_skipped = 0

    if checkpoint_path is not None:
        existing_df, done_keys = _load_checkpoint(checkpoint_path, param_keys)
        n_skipped = len(done_keys)
        if n_skipped > 0:
            print(
                f"  Checkpoint loaded: {n_skipped}/{len(all_combos)} "
                f"combinations already done, resuming..."
            )

    rows = []
    actual_skipped = 0
    t_start = time.perf_counter()

    pbar = tqdm(all_combos, desc=f"{feature_name} grid")
    for combo in pbar:
        # Check if this combo was already computed
        combo_key = tuple(str(v) for v in combo)
        if combo_key in done_keys:
            actual_skipped += 1
            pbar.set_postfix_str("skip")
            continue

        # Build kwargs for compute_fn
        params = dict(zip(param_keys, combo))

        # Compute for all segments
        values = [compute_fn(signals[j], **params) for j in range(n_seg)]

        row = {**params, **dict(zip(seg_cols, values))}
        rows.append(row)

        # Checkpoint: merge with existing and save
        if checkpoint_path is not None:
            new_df = pd.DataFrame(rows)
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged.to_csv(checkpoint_path, index=False)
            pbar.set_postfix_str("saved")

    total_sec = time.perf_counter() - t_start

    # Final DataFrame: existing + newly computed, deduplicated
    if rows:
        new_df = pd.DataFrame(rows)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = existing_df

    # Safety: deduplicate on param columns (guards against sync artefacts)
    if not final_df.empty:
        str_cols = {k: final_df[k].astype(str) for k in param_keys}
        dedup_key = pd.DataFrame(str_cols)
        final_df = final_df.loc[
            ~dedup_key.duplicated(keep="first")
        ].reset_index(drop=True)

    # Timing covers only newly computed combos
    n_new = len(all_combos) - actual_skipped
    n_calcs = n_new * n_seg

    timing = {
        "feature": feature_name,
        "n_combinations": len(all_combos),
        "n_new_combinations": n_new,
        "n_skipped": actual_skipped,
        "n_segments": n_seg,
        "n_calculations": n_calcs,
        "total_sec": round(total_sec, 2),
        "per_segment_sec": round(total_sec / max(n_calcs, 1), 4),
    }

    return final_df, timing


# ---------------------------------------------------------------------------
# Feature-specific grid runners (thin wrappers around _run_grid)
# ---------------------------------------------------------------------------


def compute_sampen_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run SampEn over all (m, r_multiplier) combinations for every segment.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'm' and 'r_multiplier', each a list.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return _run_grid(
        signals, grid,
        param_keys=["m", "r_multiplier"],
        compute_fn=compute_sampen,
        feature_name="sampen",
        checkpoint_path=checkpoint_path,
    )


def compute_apen_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run ApEn over all (m, r_multiplier) combinations for every segment.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'm' and 'r_multiplier', each a list.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return _run_grid(
        signals, grid,
        param_keys=["m", "r_multiplier"],
        compute_fn=compute_apen,
        feature_name="apen",
        checkpoint_path=checkpoint_path,
    )


def compute_permen_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run PermEn over all (order, delay) combinations for every segment.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'order' and 'delay', each a list.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return _run_grid(
        signals, grid,
        param_keys=["order", "delay"],
        compute_fn=compute_permen,
        feature_name="permen",
        checkpoint_path=checkpoint_path,
    )


def compute_fuzzen_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run FuzzyEn over all (m, r_multiplier, n) combinations for every segment.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'm', 'r_multiplier', and 'n', each a list.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return _run_grid(
        signals, grid,
        param_keys=["m", "r_multiplier", "n"],
        compute_fn=compute_fuzzen,
        feature_name="fuzzen",
        checkpoint_path=checkpoint_path,
    )


def compute_lzc_grid(
    signals: np.ndarray, grid: dict, checkpoint_path: Path | None = None
) -> tuple[pd.DataFrame, dict]:
    """Run LZC over all (threshold, multiscale) combinations for every segment.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict with keys 'threshold' and 'multiscale', each a list.
        checkpoint_path: Optional path for incremental saves.

    Returns:
        Tuple of (results_df, timing_info).
    """
    return _run_grid(
        signals, grid,
        param_keys=["threshold", "multiscale"],
        compute_fn=compute_lzc,
        feature_name="lzc",
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

_FEATURE_REGISTRY = {
    "sample_entropy": ("sampen", ["m", "r_multiplier"], compute_sampen_grid),
    "approximate_entropy": ("apen", ["m", "r_multiplier"], compute_apen_grid),
    "permutation_entropy": ("permen", ["order", "delay"], compute_permen_grid),
    "fuzzy_entropy": ("fuzzen", ["m", "r_multiplier", "n"], compute_fuzzen_grid),
    "lempel_ziv_complexity": ("lzc", ["threshold", "multiscale"], compute_lzc_grid),
}


def compute_all_entropy_features(
    signals: np.ndarray,
    config_path: str,
    output_dir: str = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Compute all entropy/complexity features using grids from the YAML config.

    Checkpoint-enabled: each feature's CSV doubles as its checkpoint file.
    If the process is interrupted and restarted, already-computed combinations
    are loaded from disk and skipped automatically.

    Args:
        signals: Array of shape (n_segments, n_samples).
        config_path: Path to hyperparameter_grid.yaml.
        output_dir: Directory to save per-feature CSV files and timing log.

    Returns:
        Dict mapping feature names to their result DataFrames.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_timings = []

    for yaml_key, (csv_suffix, _param_keys, runner_fn) in _FEATURE_REGISTRY.items():
        grid = config[yaml_key]

        print(f"\n{'='*60}")
        print(f"Computing {yaml_key} ...")
        print(f"{'='*60}")

        csv_path = output_path / f"entropy_features_{csv_suffix}.csv"

        df, timing = runner_fn(signals, grid, checkpoint_path=csv_path)

        # Final save (ensures clean state even if no new combos were computed)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        print(
            f"Time: {timing['total_sec']:.1f}s total, "
            f"{timing['per_segment_sec']:.4f}s per calculation "
            f"({timing['n_skipped']} skipped from checkpoint)"
        )

        # Track NaN count
        seg_cols = [c for c in df.columns if c.startswith("seg_")]
        nan_count = int(df[seg_cols].isna().sum().sum()) if len(seg_cols) > 0 else 0
        timing["nan_count"] = nan_count
        if nan_count > 0:
            print(f"WARNING: {nan_count} NaN values in {yaml_key}")

        all_results[yaml_key] = df
        all_timings.append(timing)

    # Save computation times
    timing_df = pd.DataFrame(all_timings)
    timing_path = output_path / "entropy_computation_times.csv"
    timing_df.to_csv(timing_path, index=False)
    print(f"\nTiming log saved: {timing_path}")

    return all_results
