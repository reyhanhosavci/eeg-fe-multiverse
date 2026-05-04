"""
Shared grid runner utilities with checkpoint support for multi-feature outputs.

Entropy features produce 1 scalar per (combo, segment).  DWT, PSD, and Hjorth
produce *multiple* named features per (combo, segment).  This module provides
a generic grid runner that handles both cases and adds checkpoint / resume
capability so long-running sweeps survive interruptions.
"""

import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_checkpoint(
    checkpoint_path: Path, param_keys: list[str]
) -> tuple[pd.DataFrame, set]:
    """Load existing checkpoint CSV and identify completed parameter combos.

    Args:
        checkpoint_path: Path to the checkpoint CSV.
        param_keys: Parameter column names used to identify unique combos.

    Returns:
        (existing_df, set_of_completed_combo_tuples).  Combo tuples are
        stringified for reliable comparison across dtypes.
    """
    if not checkpoint_path.exists():
        return pd.DataFrame(), set()

    df = pd.read_csv(checkpoint_path)
    if df.empty:
        return df, set()

    # Build string keys for each unique combo
    done = set()
    for _, group in df.groupby(param_keys, sort=False):
        key = tuple(str(group[k].iloc[0]) for k in param_keys)
        done.add(key)

    return df, done


def run_multi_feature_grid(
    signals: np.ndarray,
    grid: dict,
    param_keys: list[str],
    compute_fn,
    feature_name: str,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generic grid runner for features that return a dict per segment.

    The *compute_fn* signature must be::

        compute_fn(signal_1d, **params) -> dict[str, float]

    Each key in the returned dict becomes a ``feature_name`` column value,
    producing one row per (combo, feature_name) pair.

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict mapping parameter names to lists of values.
        param_keys: Ordered list of parameter names in the grid.
        compute_fn: Callable(signal, **params) -> dict[str, float].
        feature_name: Human-readable label for progress bar / timing log.
        checkpoint_path: If set, save after each combo and skip completed ones.

    Returns:
        (results_df, timing_info).  results_df has columns
        [*param_keys, 'feature_name', seg_000, …, seg_N-1].
    """
    param_values = [grid[k] for k in param_keys]
    all_combos = list(product(*param_values))
    n_seg = signals.shape[0]
    seg_cols = [f"seg_{i:03d}" for i in range(n_seg)]

    # Load checkpoint
    existing_df = pd.DataFrame()
    done_keys: set = set()
    n_skipped = 0

    if checkpoint_path is not None:
        existing_df, done_keys = load_checkpoint(checkpoint_path, param_keys)
        n_skipped = len(done_keys)
        if n_skipped:
            print(
                f"  Checkpoint: {n_skipped}/{len(all_combos)} "
                f"combinations already done, resuming..."
            )

    rows: list[dict] = []
    actual_skipped = 0
    t_start = time.perf_counter()

    pbar = tqdm(all_combos, desc=f"{feature_name} grid")
    for combo in pbar:
        combo_key = tuple(str(v) for v in combo)
        if combo_key in done_keys:
            actual_skipped += 1
            pbar.set_postfix_str("skip")
            continue

        params = dict(zip(param_keys, combo))

        # Compute for every segment — each call returns a dict
        feat_dicts = [compute_fn(signals[j], **params) for j in range(n_seg)]

        # Pivot: one row per feature key
        feat_keys = list(feat_dicts[0].keys())
        for fk in feat_keys:
            values = [d[fk] for d in feat_dicts]
            row = {**params, "feature_name": fk, **dict(zip(seg_cols, values))}
            rows.append(row)

        # Incremental checkpoint
        if checkpoint_path is not None:
            new_df = pd.DataFrame(rows)
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged.to_csv(checkpoint_path, index=False)
            pbar.set_postfix_str("saved")

    total_sec = time.perf_counter() - t_start

    # Final merge
    if rows:
        new_df = pd.DataFrame(rows)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = existing_df

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


def run_scalar_grid(
    signals: np.ndarray,
    grid: dict,
    param_keys: list[str],
    compute_fn,
    feature_name: str,
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generic grid runner for features that return a single float per segment.

    The *compute_fn* signature must be::

        compute_fn(signal_1d, **params) -> float

    Args:
        signals: Array of shape (n_segments, n_samples).
        grid: Dict mapping parameter names to lists of values.
        param_keys: Ordered list of parameter names in the grid.
        compute_fn: Callable(signal, **params) -> float.
        feature_name: Human-readable label for progress bar / timing log.
        checkpoint_path: If set, save after each combo and skip completed ones.

    Returns:
        (results_df, timing_info).  results_df has columns
        [*param_keys, seg_000, …, seg_N-1].
    """
    param_values = [grid[k] for k in param_keys]
    all_combos = list(product(*param_values))
    n_seg = signals.shape[0]
    seg_cols = [f"seg_{i:03d}" for i in range(n_seg)]

    existing_df = pd.DataFrame()
    done_keys: set = set()
    n_skipped = 0

    if checkpoint_path is not None:
        existing_df, done_keys = load_checkpoint(checkpoint_path, param_keys)
        n_skipped = len(done_keys)
        if n_skipped:
            print(
                f"  Checkpoint: {n_skipped}/{len(all_combos)} "
                f"combinations already done, resuming..."
            )

    rows: list[dict] = []
    actual_skipped = 0
    t_start = time.perf_counter()

    pbar = tqdm(all_combos, desc=f"{feature_name} grid")
    for combo in pbar:
        combo_key = tuple(str(v) for v in combo)
        if combo_key in done_keys:
            actual_skipped += 1
            pbar.set_postfix_str("skip")
            continue

        params = dict(zip(param_keys, combo))
        values = [compute_fn(signals[j], **params) for j in range(n_seg)]
        row = {**params, **dict(zip(seg_cols, values))}
        rows.append(row)

        if checkpoint_path is not None:
            new_df = pd.DataFrame(rows)
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged.to_csv(checkpoint_path, index=False)
            pbar.set_postfix_str("saved")

    total_sec = time.perf_counter() - t_start

    if rows:
        new_df = pd.DataFrame(rows)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = existing_df

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
