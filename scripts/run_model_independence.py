"""
Model independence validation: run XGBoost and Random Forest on all
9 feature families x 3 tasks, then compare with existing LightGBM results.

Produces:
    - 54 classification CSVs (27 XGBoost + 27 RF)
    - classification_model_comparison.csv  (summary table)
    - fig9_model_independence.png / .pdf

Usage:
    python scripts/run_model_independence.py
"""

import os
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.classification import classify_all_combinations

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
FEATURE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameter_grid.yaml"
TASKS = ["binary", "ternary", "five"]
SEED = 42

FEATURE_FILES = [
    ("entropy_features_lzc.csv", "lzc"),
    ("hurst_features.csv", "hurst"),
    ("entropy_features_sampen.csv", "sampen"),
    ("entropy_features_apen.csv", "apen"),
    ("entropy_features_permen.csv", "permen"),
    ("entropy_features_fuzzen.csv", "fuzzen"),
    ("hjorth_features.csv", "hjorth"),
    ("psd_features.csv", "psd"),
    ("dwt_features.csv", "dwt"),
]

MODELS = [
    ("lightgbm", ""),           # suffix empty = original files
    ("xgboost", "_xgboost"),
    ("random_forest", "_rf"),
]

MODEL_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "RF"}


# -------------------------------------------------------------------------
# 1. Run classification for each model
# -------------------------------------------------------------------------


def run_all_models():
    """Run XGBoost and RF; LightGBM results are reused from checkpoint."""
    total_new = 0

    for model_type, suffix in MODELS:
        label = MODEL_LABELS[model_type]
        print(f"\n{'#' * 70}")
        print(f"  MODEL: {label}")
        print(f"{'#' * 70}")

        for fname, feat_name in FEATURE_FILES:
            csv_path = FEATURE_DIR / fname
            if not csv_path.exists():
                continue

            for task in TASKS:
                out_csv = OUTPUT_DIR / f"classification_{feat_name}_{task}{suffix}.csv"

                # Skip if already fully computed
                if out_csv.exists():
                    existing = pd.read_csv(out_csv)
                    # Quick check: compare expected combo count
                    feat_csv = pd.read_csv(csv_path)
                    seg_cols = [c for c in feat_csv.columns if c.startswith("seg_")]
                    other_cols = [c for c in feat_csv.columns
                                  if not c.startswith("seg_") and c != "feature_name"]
                    if "feature_name" in feat_csv.columns:
                        n_expected = feat_csv.groupby(other_cols).ngroups
                    else:
                        n_expected = len(feat_csv)
                    if len(existing) >= n_expected:
                        print(f"  [{label}] {feat_name} x {task}: skip (complete)")
                        continue

                print(f"  [{label}] {feat_name} x {task}...", end="", flush=True)
                t0 = time.perf_counter()

                df = classify_all_combinations(
                    feature_csv_path=csv_path,
                    task=task,
                    model_type=model_type,
                    config_path=str(CONFIG_PATH),
                    checkpoint_path=out_csv,
                    seed=SEED,
                )
                df.to_csv(out_csv, index=False)
                elapsed = time.perf_counter() - t0

                best = df["accuracy"].max()
                worst = df["accuracy"].min()
                print(
                    f"  best={best:.4f}  worst={worst:.4f}  "
                    f"range={best - worst:.4f}  ({elapsed:.1f}s)"
                )
                total_new += 1

    print(f"\nNew classification jobs computed: {total_new}")


# -------------------------------------------------------------------------
# 2. Build comparison summary
# -------------------------------------------------------------------------


def build_comparison() -> pd.DataFrame:
    """Merge best/mean accuracy from all 3 models into one table."""
    rows = []

    for fname, feat_name in FEATURE_FILES:
        for task in TASKS:
            row = {"feature": feat_name, "task": task}

            for model_type, suffix in MODELS:
                label = MODEL_LABELS[model_type].lower()
                csv_path = OUTPUT_DIR / f"classification_{feat_name}_{task}{suffix}.csv"
                if not csv_path.exists():
                    row[f"{label}_best"] = np.nan
                    row[f"{label}_mean"] = np.nan
                    row[f"{label}_range"] = np.nan
                    continue

                df = pd.read_csv(csv_path)
                row[f"{label}_best"] = round(df["accuracy"].max(), 4)
                row[f"{label}_mean"] = round(df["accuracy"].mean(), 4)
                row[f"{label}_range"] = round(
                    df["accuracy"].max() - df["accuracy"].min(), 4
                )

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "classification_model_comparison.csv", index=False)
    return summary


# -------------------------------------------------------------------------
# 3. Spearman rank correlations across models
# -------------------------------------------------------------------------


def compute_rank_correlations() -> pd.DataFrame:
    """For each (feature, task), compute Spearman rho between model rankings."""
    rows = []

    for fname, feat_name in FEATURE_FILES:
        for task in TASKS:
            # Load all three result CSVs
            dfs = {}
            for model_type, suffix in MODELS:
                csv_path = OUTPUT_DIR / f"classification_{feat_name}_{task}{suffix}.csv"
                if csv_path.exists():
                    dfs[model_type] = pd.read_csv(csv_path)

            if len(dfs) < 2:
                continue

            # Identify param columns (exclude metrics)
            metric_cols = {"accuracy", "f1_macro", "f1_weighted", "std_accuracy"}
            ref_df = list(dfs.values())[0]
            param_cols = [c for c in ref_df.columns if c not in metric_cols]

            # Create a common key for matching combos
            for key, df in dfs.items():
                dfs[key] = df.copy()
                dfs[key]["_key"] = df[param_cols].astype(str).agg("|".join, axis=1)

            # Pairwise correlations
            model_names = list(dfs.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    m1, m2 = model_names[i], model_names[j]
                    d1 = dfs[m1].set_index("_key")
                    d2 = dfs[m2].set_index("_key")
                    shared = d1.index.intersection(d2.index)
                    if len(shared) < 3:
                        continue

                    rho, p = stats.spearmanr(
                        d1.loc[shared, "accuracy"],
                        d2.loc[shared, "accuracy"],
                    )
                    rows.append({
                        "feature": feat_name,
                        "task": task,
                        "model_1": MODEL_LABELS[m1],
                        "model_2": MODEL_LABELS[m2],
                        "spearman_rho": round(float(rho), 4),
                        "p_value": float(p),
                        "n_combos": len(shared),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "model_rank_correlations.csv", index=False)
    return df


# -------------------------------------------------------------------------
# 4. Fig 9 — Model independence scatter
# -------------------------------------------------------------------------


def fig9_model_independence():
    """Scatter: LightGBM accuracy vs XGBoost/RF for the most sensitive features."""
    print("\nFig 9: Model independence scatter...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    # Show the 3 most sensitive features (by binary acc range)
    highlight_features = ["psd", "hurst", "fuzzen"]
    task_colors = {"binary": "#1f77b4", "ternary": "#ff7f0e", "five": "#2ca02c"}

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=False, sharey=False)

    for col, feat_name in enumerate(highlight_features):
        for row_idx, (alt_model, suffix, alt_label) in enumerate([
            ("xgboost", "_xgboost", "XGBoost"),
            ("random_forest", "_rf", "RF"),
        ]):
            ax = axes[row_idx, col]

            for task in TASKS:
                lgbm_path = OUTPUT_DIR / f"classification_{feat_name}_{task}.csv"
                alt_path = OUTPUT_DIR / f"classification_{feat_name}_{task}{suffix}.csv"
                if not lgbm_path.exists() or not alt_path.exists():
                    continue

                df_lgbm = pd.read_csv(lgbm_path)
                df_alt = pd.read_csv(alt_path)

                # Match by param columns
                metric_cols = {"accuracy", "f1_macro", "f1_weighted", "std_accuracy"}
                param_cols = [c for c in df_lgbm.columns if c not in metric_cols]

                df_lgbm["_key"] = df_lgbm[param_cols].astype(str).agg("|".join, axis=1)
                df_alt["_key"] = df_alt[param_cols].astype(str).agg("|".join, axis=1)

                merged = df_lgbm[["_key", "accuracy"]].merge(
                    df_alt[["_key", "accuracy"]], on="_key", suffixes=("_lgbm", "_alt")
                )

                ax.scatter(
                    merged["accuracy_lgbm"], merged["accuracy_alt"],
                    c=task_colors[task], alpha=0.6, s=20, label=task.capitalize(),
                    edgecolors="none",
                )

            # 45-degree line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")

            if row_idx == 0:
                ax.set_title(feat_name.upper(), fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{alt_label} Accuracy")
            if row_idx == 1:
                ax.set_xlabel("LightGBM Accuracy")
            if col == 2 and row_idx == 0:
                ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Model Independence: Do Hyperparameter Rankings Hold Across Classifiers?",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig9_model_independence.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved: fig9_model_independence.png / .pdf")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main():
    t_start = time.perf_counter()

    # 1. Run XGBoost + RF classification
    run_all_models()

    # 2. Comparison summary
    print(f"\n{'=' * 70}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    summary = build_comparison()
    pd.set_option("display.width", 130)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string(index=False))

    # 3. Rank correlations
    print(f"\n{'=' * 70}")
    print("  SPEARMAN RANK CORRELATIONS (model pairs)")
    print(f"{'=' * 70}")
    rank_corr = compute_rank_correlations()
    print(rank_corr.to_string(index=False))

    # Overall average rho
    mean_rho = rank_corr["spearman_rho"].mean()
    print(f"\n  Overall mean Spearman rho: {mean_rho:.4f}")

    # 4. Figure
    fig9_model_independence()

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE -- {elapsed / 60:.1f} min total")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
