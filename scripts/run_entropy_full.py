"""
Full entropy feature extraction for all 500 Bonn EEG segments.

Execution order (fast → slow) to catch errors early:
    1. PermEn   —  25 combos, ~6 sec
    2. LZC      —   6 combos, ~12 sec
    3. SampEn   —  15 combos, ~2 h
    4. ApEn     —  15 combos, ~2 h
    5. FuzzyEn  —  27 combos, ~4.5 h

Checkpoint-enabled: interrupt with Ctrl+C and re-run to resume.

Usage:
    python scripts/run_entropy_full.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is on the import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.data_loader import load_bonn_all, get_task_data
from src.features.entropy_features import (
    compute_permen_grid,
    compute_lzc_grid,
    compute_sampen_grid,
    compute_apen_grid,
    compute_fuzzen_grid,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / ".." / "EEG Data"
CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameter_grid.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

# Ordered pipeline: fast features first so errors surface quickly
PIPELINE = [
    ("permutation_entropy", "permen", ["order", "delay"], compute_permen_grid),
    ("lempel_ziv_complexity", "lzc", ["threshold", "multiscale"], compute_lzc_grid),
    ("sample_entropy", "sampen", ["m", "r_multiplier"], compute_sampen_grid),
    ("approximate_entropy", "apen", ["m", "r_multiplier"], compute_apen_grid),
    ("fuzzy_entropy", "fuzzen", ["m", "r_multiplier", "n"], compute_fuzzen_grid),
]


def main():
    """Load all segments, compute entropy features, save results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Load all 500 segments (5 sets × 100)
    print("Loading Bonn dataset...")
    data = load_bonn_all(str(RAW_DIR))
    signals = np.vstack([data[s] for s in ["A", "B", "C", "D", "E"]])
    print(f"Loaded: {signals.shape[0]} segments, {signals.shape[1]} samples each\n")

    # Run pipeline
    timings = []
    t_total_start = time.perf_counter()

    for yaml_key, csv_suffix, _param_keys, runner_fn in PIPELINE:
        grid = config[yaml_key]
        csv_path = OUTPUT_DIR / f"entropy_features_{csv_suffix}.csv"

        n_combos = 1
        for v in grid.values():
            if isinstance(v, list):
                n_combos *= len(v)

        print(f"{'='*60}")
        print(f"  {yaml_key}  ({n_combos} combinations × {signals.shape[0]} segments)")
        print(f"  Checkpoint: {csv_path}")
        print(f"{'='*60}")

        df, timing = runner_fn(signals, grid, checkpoint_path=csv_path)

        # Final save
        df.to_csv(csv_path, index=False)

        # NaN report
        seg_cols = [c for c in df.columns if c.startswith("seg_")]
        nan_count = int(df[seg_cols].isna().sum().sum()) if seg_cols else 0
        timing["nan_count"] = nan_count

        print(
            f"  Done: {timing['n_new_combinations']} new, "
            f"{timing['n_skipped']} skipped, "
            f"{timing['total_sec']:.1f}s, "
            f"{nan_count} NaN"
        )
        print(f"  Saved: {csv_path}\n")

        timings.append(timing)

    # Save timing log
    import pandas as pd

    timing_df = pd.DataFrame(timings)
    timing_path = OUTPUT_DIR / "entropy_computation_times.csv"
    timing_df.to_csv(timing_path, index=False)

    t_total = time.perf_counter() - t_total_start
    print(f"{'='*60}")
    print(f"  ALL DONE — total wall time: {t_total/60:.1f} min")
    print(f"  Timing log: {timing_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
