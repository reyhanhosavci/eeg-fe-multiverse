"""
Classification pipeline for the Feature Extraction Multiverse project.

Reads feature CSVs produced by the feature extraction modules, runs stratified
k-fold cross-validation for every hyperparameter combination, and saves
per-combination metrics (accuracy, F1-macro, F1-weighted).

Supports two CSV layouts automatically:
    - Scalar features (entropy, Hurst): one row per combo, no 'feature_name' col.
    - Multi-feature (DWT, PSD, Hjorth): multiple rows per combo with 'feature_name'.

Checkpoint-enabled: completed combinations are skipped on restart.
"""

import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Task → segment index / label mapping
# ---------------------------------------------------------------------------

# Segment layout: A(0-99), B(100-199), C(200-299), D(300-399), E(400-499)
_SET_RANGES = {
    "A": (0, 100), "B": (100, 200), "C": (200, 300),
    "D": (300, 400), "E": (400, 500),
}


def get_task_indices(task: str) -> tuple[np.ndarray, np.ndarray]:
    """Return segment indices and labels for a classification task.

    Args:
        task: One of 'binary', 'ternary', 'five'.

    Returns:
        (indices, y) where indices selects seg columns and y holds class labels.
    """
    task = task.lower()
    if task == "binary":
        # A+B (0) vs E (1)
        idx = list(range(0, 200)) + list(range(400, 500))
        y = [0] * 200 + [1] * 100
    elif task == "ternary":
        # A+B (0) vs C+D (1) vs E (2)
        idx = list(range(500))
        y = [0] * 200 + [1] * 200 + [2] * 100
    elif task == "five":
        # A(0) B(1) C(2) D(3) E(4)
        idx = list(range(500))
        y = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100
    else:
        raise ValueError(f"Unknown task '{task}'. Use 'binary', 'ternary', 'five'.")
    return np.array(idx), np.array(y)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def iter_combinations(csv_path: str | Path) -> list[tuple[dict, np.ndarray]]:
    """Read a feature CSV and yield (params_dict, X_matrix) per combination.

    For scalar CSVs (no 'feature_name' column), each row is one combination
    producing X of shape (n_segments, 1).

    For multi-feature CSVs (with 'feature_name'), rows sharing the same
    parameter values are grouped and stacked into X of shape
    (n_segments, n_sub_features).

    Args:
        csv_path: Path to a feature CSV.

    Returns:
        List of (params_dict, X) tuples.  X columns correspond to seg_000 …
        seg_N-1 (transposed so rows = segments).
    """
    df = pd.read_csv(csv_path)
    seg_cols = sorted([c for c in df.columns if c.startswith("seg_")])
    other_cols = [c for c in df.columns if not c.startswith("seg_")]

    has_feature_name = "feature_name" in other_cols
    param_cols = [c for c in other_cols if c != "feature_name"]

    results = []

    if not has_feature_name:
        # Scalar: one row = one combo
        for _, row in df.iterrows():
            params = {c: row[c] for c in param_cols}
            X = row[seg_cols].values.astype(float).reshape(-1, 1)  # (n_seg, 1)
            results.append((params, X))
    else:
        # Multi-feature: group by param columns
        grouped = df.groupby(param_cols, sort=False)
        for keys, group in grouped:
            if isinstance(keys, tuple):
                params = dict(zip(param_cols, keys))
            else:
                params = {param_cols[0]: keys}
            # Each row in group is a sub-feature; stack → (n_sub_feat, n_seg)
            feat_matrix = group[seg_cols].values.astype(float)
            X = feat_matrix.T  # (n_seg, n_sub_features)
            results.append((params, X))

    return results


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _create_model(model_type: str, config: dict, n_classes: int):
    """Create a classifier instance from config.

    Args:
        model_type: One of 'lightgbm', 'xgboost', 'random_forest'.
        config: Full YAML config dict.
        n_classes: Number of classes (used for is_unbalance flag).

    Returns:
        Scikit-learn compatible classifier.
    """
    if model_type == "lightgbm":
        params = dict(config.get("lightgbm", {}))
        params["is_unbalance"] = True
        return lgb.LGBMClassifier(**params)

    elif model_type == "xgboost":
        params = dict(config.get("xgboost", {}))
        return XGBClassifier(**params)

    elif model_type == "random_forest":
        params = dict(config.get("random_forest", {}))
        return RandomForestClassifier(**params)

    else:
        raise ValueError(f"Unknown model_type '{model_type}'")


# ---------------------------------------------------------------------------
# Single combination classifier
# ---------------------------------------------------------------------------


def classify_single_combination(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    model,
) -> dict:
    """Run stratified k-fold CV for one hyperparameter combination.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        cv_folds: Pre-computed list of (train_idx, test_idx) tuples.
        model: Scikit-learn compatible classifier (cloned per fold).

    Returns:
        Dict with keys: accuracy, f1_macro, f1_weighted, std_accuracy,
        fold_accuracies.
    """
    from sklearn.base import clone

    fold_acc, fold_f1m, fold_f1w = [], [], []

    for train_idx, test_idx in cv_folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle NaN: replace with column median from train set
        col_medians = np.nanmedian(X_train, axis=0)
        for col in range(X_train.shape[1]):
            mask_train = np.isnan(X_train[:, col])
            mask_test = np.isnan(X_test[:, col])
            X_train[mask_train, col] = col_medians[col]
            X_test[mask_test, col] = col_medians[col]

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        fold_acc.append(accuracy_score(y_test, y_pred))
        fold_f1m.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        fold_f1w.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": float(np.mean(fold_acc)),
        "f1_macro": float(np.mean(fold_f1m)),
        "f1_weighted": float(np.mean(fold_f1w)),
        "std_accuracy": float(np.std(fold_acc)),
        "fold_accuracies": fold_acc,
    }


# ---------------------------------------------------------------------------
# All combinations for one CSV × one task
# ---------------------------------------------------------------------------


def classify_all_combinations(
    feature_csv_path: str | Path,
    task: str,
    model_type: str = "lightgbm",
    config_path: str | Path = "configs/hyperparameter_grid.yaml",
    checkpoint_path: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Classify every parameter combination in a feature CSV for a given task.

    Args:
        feature_csv_path: Path to the feature CSV.
        task: 'binary', 'ternary', or 'five'.
        model_type: 'lightgbm', 'xgboost', or 'random_forest'.
        config_path: Path to YAML config with model parameters.
        checkpoint_path: Optional path for incremental saves.
        seed: Random seed for CV splits.

    Returns:
        DataFrame with param columns + accuracy, f1_macro, f1_weighted, std_accuracy.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Task indices and labels
    seg_indices, y_all = get_task_indices(task)

    # CV folds (same for all combinations within this task)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_folds = list(skf.split(np.zeros(len(y_all)), y_all))

    # Model template
    n_classes = len(np.unique(y_all))
    model = _create_model(model_type, config, n_classes)

    # Load all combinations from CSV
    combos = iter_combinations(feature_csv_path)

    # Load checkpoint
    existing_rows = []
    done_keys: set = set()
    if checkpoint_path and checkpoint_path.exists():
        existing_df = pd.read_csv(checkpoint_path)
        if not existing_df.empty:
            existing_rows = existing_df.to_dict("records")
            # Build done keys from param columns
            metric_cols = {"accuracy", "f1_macro", "f1_weighted", "std_accuracy"}
            param_cols_ck = [
                c for c in existing_df.columns if c not in metric_cols
            ]
            for _, row in existing_df.iterrows():
                key = tuple(str(row[c]) for c in param_cols_ck)
                done_keys.add(key)
            print(
                f"  Checkpoint: {len(done_keys)}/{len(combos)} "
                f"combinations already done"
            )

    new_rows = []
    for params, X_full in tqdm(combos, desc=f"Classify {task}"):
        # Check checkpoint
        param_key = tuple(str(v) for v in params.values())
        if param_key in done_keys:
            continue

        # Select task segments
        X = X_full[seg_indices].copy()

        # Classify
        metrics = classify_single_combination(X, y_all, cv_folds, model)

        row = {
            **params,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "std_accuracy": metrics["std_accuracy"],
        }
        new_rows.append(row)

        # Incremental checkpoint
        if checkpoint_path:
            all_rows = existing_rows + new_rows
            pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)

    # Final DataFrame
    all_rows = existing_rows + new_rows
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Full classification across all features × tasks
# ---------------------------------------------------------------------------

# Feature CSV filenames → human-readable names
_FEATURE_CSV_MAP = {
    "entropy_features_sampen.csv": "sampen",
    "entropy_features_apen.csv": "apen",
    "entropy_features_permen.csv": "permen",
    "entropy_features_fuzzen.csv": "fuzzen",
    "entropy_features_lzc.csv": "lzc",
    "dwt_features.csv": "dwt",
    "psd_features.csv": "psd",
    "hjorth_features.csv": "hjorth",
    "hurst_features.csv": "hurst",
}


def run_full_classification(
    feature_dir: str | Path = "results/tables",
    output_dir: str | Path = "results/tables",
    tasks: list[str] | None = None,
    model_type: str = "lightgbm",
    config_path: str | Path = "configs/hyperparameter_grid.yaml",
    seed: int = 42,
) -> pd.DataFrame:
    """Run classification for all feature CSVs × all tasks.

    Produces one result CSV per (feature, task) pair and a timing summary.

    Args:
        feature_dir: Directory containing feature CSVs.
        output_dir: Directory for classification result CSVs.
        tasks: List of tasks to run. Defaults to ['binary', 'ternary', 'five'].
        model_type: Classifier type.
        config_path: Path to YAML config.
        seed: Random seed.

    Returns:
        Summary DataFrame with one row per (feature, task) combination.
    """
    if tasks is None:
        tasks = ["binary", "ternary", "five"]

    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timings = []

    for csv_name, feat_name in _FEATURE_CSV_MAP.items():
        csv_path = feature_dir / csv_name
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue

        for task in tasks:
            out_csv = output_dir / f"classification_{feat_name}_{task}.csv"

            print(f"\n{'=' * 60}")
            print(f"  {feat_name} × {task}  (model={model_type})")
            print(f"  Checkpoint: {out_csv}")
            print(f"{'=' * 60}")

            t_start = time.perf_counter()

            df = classify_all_combinations(
                feature_csv_path=csv_path,
                task=task,
                model_type=model_type,
                config_path=config_path,
                checkpoint_path=out_csv,
                seed=seed,
            )

            # Final save
            df.to_csv(out_csv, index=False)
            elapsed = time.perf_counter() - t_start

            if not df.empty:
                best_acc = df["accuracy"].max()
                mean_acc = df["accuracy"].mean()
                acc_range = df["accuracy"].max() - df["accuracy"].min()
            else:
                best_acc = mean_acc = acc_range = np.nan

            print(
                f"  {len(df)} combinations | "
                f"best={best_acc:.4f} mean={mean_acc:.4f} range={acc_range:.4f} | "
                f"{elapsed:.1f}s"
            )

            timings.append({
                "feature": feat_name,
                "task": task,
                "model": model_type,
                "n_combinations": len(df),
                "best_accuracy": round(best_acc, 4),
                "mean_accuracy": round(mean_acc, 4),
                "accuracy_range": round(acc_range, 4),
                "elapsed_sec": round(elapsed, 2),
            })

    # Save timing summary
    timing_df = pd.DataFrame(timings)
    timing_path = output_dir / "classification_times.csv"
    timing_df.to_csv(timing_path, index=False)
    print(f"\nTiming summary: {timing_path}")

    return timing_df
