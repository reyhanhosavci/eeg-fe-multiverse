"""
Run LightGBM classification for all 9 feature families × 3 tasks = 27 jobs.

Execution order: fast features first (scalar/few combos → multi-feature/many combos).
Checkpoint-enabled: re-run safely to resume interrupted work.

Usage:
    python scripts/run_classification.py
"""

import os
import sys
import warnings

# Suppress sklearn / LightGBM warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time

import pandas as pd

from src.classification import classify_all_combinations

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameter_grid.yaml"
TASKS = ["binary", "ternary", "five"]
MODEL_TYPE = "lightgbm"
SEED = 42

# Ordered fast → slow (scalar/few combos first, multi-feature/many combos last)
FEATURE_FILES = [
    ("entropy_features_lzc.csv", "lzc"),          #  6 combos, scalar
    ("hurst_features.csv", "hurst"),               # 12 combos, scalar
    ("entropy_features_sampen.csv", "sampen"),      # 15 combos, scalar
    ("entropy_features_apen.csv", "apen"),          # 15 combos, scalar
    ("entropy_features_permen.csv", "permen"),      # 25 combos, scalar
    ("entropy_features_fuzzen.csv", "fuzzen"),      # 27 combos, scalar
    ("hjorth_features.csv", "hjorth"),              #  6 combos, 3 features
    ("hurst_features.csv", "hurst"),                # skip (already above)
    ("psd_features.csv", "psd"),                   # 48 combos, 7 features
    ("dwt_features.csv", "dwt"),                   # 36 combos, 12-21 features
]

# Deduplicate (hurst was listed twice above as a reminder)
_seen = set()
FEATURE_FILES_DEDUP = []
for fname, label in FEATURE_FILES:
    if label not in _seen:
        _seen.add(label)
        FEATURE_FILES_DEDUP.append((fname, label))


def main():
    """Run all feature × task classification jobs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    t_total = time.perf_counter()

    total_jobs = 0
    for fname, _ in FEATURE_FILES_DEDUP:
        if (FEATURE_DIR / fname).exists():
            total_jobs += len(TASKS)

    job_num = 0
    for fname, feat_name in FEATURE_FILES_DEDUP:
        csv_path = FEATURE_DIR / fname
        if not csv_path.exists():
            print(f"  SKIP: {csv_path.name} not found\n")
            continue

        for task in TASKS:
            job_num += 1
            out_csv = OUTPUT_DIR / f"classification_{feat_name}_{task}.csv"

            print(f"[{job_num}/{total_jobs}] {feat_name} × {task}", end="")
            sys.stdout.flush()

            t_start = time.perf_counter()
            df = classify_all_combinations(
                feature_csv_path=csv_path,
                task=task,
                model_type=MODEL_TYPE,
                config_path=str(CONFIG_PATH),
                checkpoint_path=out_csv,
                seed=SEED,
            )
            df.to_csv(out_csv, index=False)
            elapsed = time.perf_counter() - t_start

            best = df["accuracy"].max()
            worst = df["accuracy"].min()
            mean = df["accuracy"].mean()
            delta = best - worst

            print(
                f"  >>  best={best:.4f}  worst={worst:.4f}  "
                f"range={delta:.4f}  ({elapsed:.1f}s)"
            )

            summary_rows.append({
                "feature": feat_name,
                "task": task,
                "n_combos": len(df),
                "best_acc": round(best, 4),
                "worst_acc": round(worst, 4),
                "mean_acc": round(mean, 4),
                "acc_range": round(delta, 4),
                "best_f1_macro": round(df["f1_macro"].max(), 4),
                "elapsed_sec": round(elapsed, 1),
            })

    wall = time.perf_counter() - t_total

    # Summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "classification_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 78}")
    print(f"  CLASSIFICATION COMPLETE — {wall / 60:.1f} min total")
    print(f"{'=' * 78}\n")

    # Pretty-print summary
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary_df.to_string(index=False))

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
