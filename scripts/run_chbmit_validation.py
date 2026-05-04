"""
Cross-dataset validation: run 6 feature families on CHB-MIT and compare
sensitivity rankings with Bonn results.

Features tested (4 sensitive + 2 stable):
    PSD, Hurst, FuzzyEn, SampEn, DWT, Hjorth

Two CV strategies:
    1. 5-fold StratifiedKFold (comparable to Bonn)
    2. Leave-One-Patient-Out (clinically meaningful)

Usage:
    python scripts/run_chbmit_validation.py
"""

import os
import sys
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from src.chbmit_loader import load_chbmit, FS_CHBMIT
from src.classification import _create_model, classify_single_combination
from src.features.wavelet_features import compute_dwt_features
from src.features.spectral_features import compute_psd_features
from src.features.temporal_features import compute_hjorth_features, compute_hurst
from src.features.entropy_features import (
    compute_sampen, compute_fuzzen, compute_apen, compute_permen, compute_lzc,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "chbmit" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameter_grid.yaml"
SEED = 42

# Feature grids to test (read from YAML) — all 9 families
FEATURES_TO_TEST = [
    "psd_welch", "hurst", "fuzzy_entropy", "sample_entropy",
    "approximate_entropy", "permutation_entropy", "lempel_ziv_complexity",
    "dwt", "hjorth",
]

FEATURE_SHORT = {
    "psd_welch": "psd", "hurst": "hurst", "fuzzy_entropy": "fuzzen",
    "sample_entropy": "sampen", "approximate_entropy": "apen",
    "permutation_entropy": "permen", "lempel_ziv_complexity": "lzc",
    "dwt": "dwt", "hjorth": "hjorth",
}


# ---------------------------------------------------------------------------
# Feature computation for a single segment
# ---------------------------------------------------------------------------


def compute_features_for_combo(signal, feature_key, params, fs=FS_CHBMIT):
    """Compute feature(s) for a single segment with given params.

    Returns a 1-D numpy array of feature values.
    """
    if feature_key == "psd_welch":
        feat_dict = compute_psd_features(
            signal, window=params["window"], nperseg=params["nperseg"],
            overlap_ratio=params["overlap_ratio"], fs=fs,
        )
    elif feature_key == "dwt":
        feat_dict = compute_dwt_features(
            signal, wavelet=params["wavelet"], level=params["level"]
        )
    elif feature_key == "hjorth":
        feat_dict = compute_hjorth_features(
            signal, window_size_sec=params["window_size_sec"], fs=fs,
        )
    elif feature_key == "hurst":
        val = compute_hurst(
            signal, method=params["method"], min_window=params["min_window"]
        )
        return np.array([val])
    elif feature_key == "sample_entropy":
        val = compute_sampen(
            signal, m=params["m"], r_multiplier=params["r_multiplier"]
        )
        return np.array([val])
    elif feature_key == "fuzzy_entropy":
        val = compute_fuzzen(
            signal, m=params["m"], r_multiplier=params["r_multiplier"],
            n=params["n"],
        )
        return np.array([val])
    elif feature_key == "approximate_entropy":
        val = compute_apen(
            signal, m=params["m"], r_multiplier=params["r_multiplier"]
        )
        return np.array([val])
    elif feature_key == "permutation_entropy":
        val = compute_permen(
            signal, order=params["order"], delay=params["delay"]
        )
        return np.array([val])
    elif feature_key == "lempel_ziv_complexity":
        val = compute_lzc(
            signal, threshold=str(params["threshold"]),
            multiscale=bool(params["multiscale"]),
        )
        return np.array([val])
    else:
        raise ValueError(f"Unknown feature: {feature_key}")

    return np.array(list(feat_dict.values()))


# ---------------------------------------------------------------------------
# Run classification for one feature x one combo x one CV strategy
# ---------------------------------------------------------------------------


def classify_combo(
    signals, y, combo_params, feature_key, model, cv_folds, fs=FS_CHBMIT,
):
    """Build feature matrix for one param combo, then classify."""
    from sklearn.base import clone
    from tqdm import tqdm

    # Build feature matrix
    X_list = []
    for i in range(len(signals)):
        feats = compute_features_for_combo(signals[i], feature_key, combo_params, fs)
        X_list.append(feats)
    X = np.vstack(X_list)

    # Handle NaN
    col_medians = np.nanmedian(X, axis=0)
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        X[mask, col] = col_medians[col] if np.isfinite(col_medians[col]) else 0.0

    # Cross-validation
    fold_acc = []
    for train_idx, test_idx in cv_folds:
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[test_idx])
        acc = float(np.mean(y_pred == y[test_idx]))
        fold_acc.append(acc)

    return {
        "accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_total = time.perf_counter()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # ---- 1. Load CHB-MIT data ----
    print("=" * 60)
    print("  STEP 1: Loading CHB-MIT data")
    print("=" * 60)
    X_all, y_all, pid_all = load_chbmit(data_dir=str(DATA_DIR), seed=SEED)

    # ---- 2. Prepare CV folds ----
    # 5-fold stratified
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_5fold = list(skf.split(X_all, y_all))

    # Leave-one-patient-out
    logo = LeaveOneGroupOut()
    cv_lopo = list(logo.split(X_all, y_all, groups=pid_all))

    print(f"\n  CV: 5-fold stratified ({len(cv_5fold)} folds)")
    print(f"  CV: LOPO ({len(cv_lopo)} folds)")

    # ---- 3. Model ----
    model = _create_model("lightgbm", config, n_classes=2)

    # ---- 4. Run all feature x combo classifications ----
    print(f"\n{'=' * 60}")
    print("  STEP 2: Feature extraction + classification")
    print(f"{'=' * 60}")

    from itertools import product
    from tqdm import tqdm

    all_results = []

    for feature_key in FEATURES_TO_TEST:
        short = FEATURE_SHORT[feature_key]
        grid = config[feature_key]

        # Build combo list
        param_keys = list(grid.keys())
        param_values = [grid[k] for k in param_keys]
        combos = list(product(*param_values))
        n_combos = len(combos)

        # Checkpoint
        ckpt_path = OUTPUT_DIR / f"chbmit_classification_{short}.csv"
        done_keys = set()
        existing_rows = []
        if ckpt_path.exists():
            existing_df = pd.read_csv(ckpt_path)
            if not existing_df.empty:
                existing_rows = existing_df.to_dict("records")
                for _, row in existing_df.iterrows():
                    key = tuple(str(row[k]) for k in param_keys if k in row)
                    done_keys.add(key)

        print(f"\n  {short.upper()} ({n_combos} combos)", end="")
        if done_keys:
            print(f" — {len(done_keys)} checkpointed", end="")
        print()

        new_rows = []
        for combo in tqdm(combos, desc=f"  {short}"):
            combo_key = tuple(str(v) for v in combo)
            if combo_key in done_keys:
                continue

            params = dict(zip(param_keys, combo))

            # 5-fold
            r5 = classify_combo(X_all, y_all, params, feature_key, model, cv_5fold)
            # LOPO
            rlopo = classify_combo(X_all, y_all, params, feature_key, model, cv_lopo)

            row = {
                **params,
                "acc_5fold": round(r5["accuracy"], 4),
                "std_5fold": round(r5["std_accuracy"], 4),
                "acc_lopo": round(rlopo["accuracy"], 4),
                "std_lopo": round(rlopo["std_accuracy"], 4),
            }
            new_rows.append(row)

            # Checkpoint after each combo
            all_rows_now = existing_rows + new_rows
            pd.DataFrame(all_rows_now).to_csv(ckpt_path, index=False)

        # Final save
        final_rows = existing_rows + new_rows
        df = pd.DataFrame(final_rows)
        df.to_csv(ckpt_path, index=False)

        if not df.empty:
            for cv_type in ["5fold", "lopo"]:
                col = f"acc_{cv_type}"
                best = df[col].max()
                worst = df[col].min()
                rng = best - worst
                all_results.append({
                    "feature": short,
                    "cv": cv_type,
                    "n_combos": len(df),
                    "best_acc": round(best, 4),
                    "worst_acc": round(worst, 4),
                    "mean_acc": round(df[col].mean(), 4),
                    "acc_range": round(rng, 4),
                })
                print(f"    {cv_type}: best={best:.4f} worst={worst:.4f} range={rng:.4f}")

    # ---- 5. Summary + Bonn comparison ----
    print(f"\n{'=' * 60}")
    print("  STEP 3: Sensitivity comparison (Bonn vs CHB-MIT)")
    print(f"{'=' * 60}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_DIR / "chbmit_sensitivity_summary.csv", index=False)

    # Compare with Bonn binary ranking
    bonn_scores = pd.read_csv(OUTPUT_DIR / "sensitivity_scores.csv")
    bonn_binary = bonn_scores[bonn_scores["task"] == "binary"].copy()
    bonn_binary = bonn_binary[bonn_binary["feature"].isin(FEATURE_SHORT.values())]
    bonn_binary = bonn_binary.set_index("feature")["acc_range"]

    chbmit_5fold = summary_df[summary_df["cv"] == "5fold"].set_index("feature")["acc_range"]

    # Align features
    shared = sorted(set(bonn_binary.index) & set(chbmit_5fold.index))
    if len(shared) >= 3:
        bonn_vals = [bonn_binary[f] for f in shared]
        chbmit_vals = [chbmit_5fold[f] for f in shared]

        rho, p = stats.spearmanr(bonn_vals, chbmit_vals)
        print(f"\n  Spearman rho (sensitivity ranking): {rho:.4f}  (p={p:.4f})")
        print(f"\n  Feature ranking comparison:")
        print(f"  {'Feature':<12} {'Bonn range':>12} {'CHB-MIT range':>14} {'Bonn rank':>10} {'CHB-MIT rank':>12}")
        print(f"  {'-'*62}")

        bonn_rank = pd.Series(bonn_vals, index=shared).rank(ascending=False)
        chbmit_rank = pd.Series(chbmit_vals, index=shared).rank(ascending=False)
        for f in shared:
            print(
                f"  {f:<12} {bonn_binary[f]:>12.4f} {chbmit_5fold[f]:>14.4f}"
                f" {int(bonn_rank[f]):>10} {int(chbmit_rank[f]):>12}"
            )

        # Save comparison
        comp_df = pd.DataFrame({
            "feature": shared,
            "bonn_range": bonn_vals,
            "chbmit_range": chbmit_vals,
            "bonn_rank": [int(bonn_rank[f]) for f in shared],
            "chbmit_rank": [int(chbmit_rank[f]) for f in shared],
        })
        comp_df.to_csv(OUTPUT_DIR / "cross_dataset_ranking_comparison.csv", index=False)
    else:
        print("  Not enough shared features to compute ranking correlation")

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 60}")
    print(f"  DONE — {elapsed / 60:.1f} min total")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
