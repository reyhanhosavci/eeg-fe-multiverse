"""
Compute DWT, PSD, Hjorth, and Hurst features for all 500 Bonn EEG segments.

Execution order (fast → slow):
    1. Hjorth    —   6 combos ×  3 features  → 18 rows,  ~5 sec
    2. Hurst     —  12 combos ×  1 feature   → 12 rows,  ~30 sec
    3. PSD/Welch —  48 combos ×  7 features  → 336 rows, ~2 min
    4. DWT       —  36 combos × 12-21 features → 594 rows, ~5 min

Checkpoint-enabled: interrupt with Ctrl+C and re-run to resume.

Usage:
    python scripts/run_remaining_features.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is on the import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_bonn_all
from src.features.wavelet_features import compute_dwt_grid
from src.features.spectral_features import compute_psd_grid
from src.features.temporal_features import compute_hjorth_grid, compute_hurst_grid

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR = PROJECT_ROOT / ".." / "EEG Data"
CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameter_grid.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

# (yaml_key, csv_filename, grid_runner)
PIPELINE = [
    ("hjorth", "hjorth_features.csv", compute_hjorth_grid),
    ("hurst", "hurst_features.csv", compute_hurst_grid),
    ("psd_welch", "psd_features.csv", compute_psd_grid),
    ("dwt", "dwt_features.csv", compute_dwt_grid),
]


def _count_combos(grid: dict) -> int:
    """Count total parameter combinations in a grid."""
    n = 1
    for v in grid.values():
        if isinstance(v, list):
            n *= len(v)
    return n


def main():
    """Load all segments, compute remaining features, save results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Load all 500 segments
    print("Loading Bonn dataset...")
    data = load_bonn_all(str(RAW_DIR))
    signals = np.vstack([data[s] for s in ["A", "B", "C", "D", "E"]])
    print(f"Loaded: {signals.shape[0]} segments × {signals.shape[1]} samples\n")

    timings = []
    t_total_start = time.perf_counter()

    for yaml_key, csv_name, runner_fn in PIPELINE:
        grid = config[yaml_key]
        csv_path = OUTPUT_DIR / csv_name
        n_combos = _count_combos(grid)

        print(f"{'=' * 60}")
        print(f"  {yaml_key}  ({n_combos} combinations × {signals.shape[0]} segments)")
        print(f"  Checkpoint: {csv_path}")
        print(f"{'=' * 60}")

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
        print(f"  Saved: {csv_path}  ({len(df)} rows)\n")

        timings.append(timing)

    # Save timing log
    timing_df = pd.DataFrame(timings)
    timing_path = OUTPUT_DIR / "remaining_computation_times.csv"
    timing_df.to_csv(timing_path, index=False)

    t_total = time.perf_counter() - t_total_start
    print(f"{'=' * 60}")
    print(f"  ALL DONE — total wall time: {t_total / 60:.1f} min")
    print(f"  Timing log: {timing_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
