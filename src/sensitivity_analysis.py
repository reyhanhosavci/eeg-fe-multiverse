"""
Sensitivity analysis for the Feature Extraction Multiverse project.

Quantifies how much classification accuracy depends on feature-extraction
hyperparameters via four complementary lenses:

    1. Sensitivity scores   — range, std, CV for each feature × task
    2. Main effects         — marginal-mean analysis + η² per parameter
    3. Interactions         — practical interaction score + optional 2-way ANOVA
    4. Task comparison      — Spearman ρ across tasks, optimal-match rate
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Check optional dependency once at import time
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols as sm_ols
    from statsmodels.stats.anova import anova_lm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Feature family → its hyperparameter column names
PARAM_KEYS: dict[str, list[str]] = {
    "sampen": ["m", "r_multiplier"],
    "apen": ["m", "r_multiplier"],
    "permen": ["order", "delay"],
    "fuzzen": ["m", "r_multiplier", "n"],
    "lzc": ["threshold", "multiscale"],
    "dwt": ["wavelet", "level"],
    "psd": ["window", "nperseg", "overlap_ratio"],
    "hjorth": ["window_size_sec"],
    "hurst": ["method", "min_window"],
}


def _get_param_cols(df: pd.DataFrame, feature: str) -> list[str]:
    """Return the hyperparameter columns present in *df* for *feature*.

    Falls back to inferring from column names if the feature is unknown.
    """
    if feature in PARAM_KEYS:
        return [c for c in PARAM_KEYS[feature] if c in df.columns]
    # Fallback: everything that isn't a metric or metadata column
    skip = {
        "feature_family", "task", "accuracy", "f1_macro",
        "f1_weighted", "std_accuracy",
    }
    return [c for c in df.columns if c not in skip]


def _subset(master: pd.DataFrame, feature: str, task: str) -> pd.DataFrame:
    """Filter master DataFrame to one (feature, task) slice."""
    return master[
        (master["feature_family"] == feature) & (master["task"] == task)
    ].copy()


# =========================================================================
# 1. Sensitivity scores
# =========================================================================


def compute_sensitivity_scores(
    master_csv: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute accuracy sensitivity statistics for every feature × task pair.

    Metrics per pair:
        acc_min, acc_max, acc_mean, acc_range, acc_std, cv_pct

    Args:
        master_csv: Path to classification_master.csv.
        output_path: If set, save result CSV here.

    Returns:
        DataFrame with one row per (feature, task).
    """
    master = pd.read_csv(master_csv)
    rows = []

    for feature in sorted(master["feature_family"].unique()):
        for task in ["binary", "ternary", "five"]:
            sub = _subset(master, feature, task)
            if sub.empty:
                continue
            acc = sub["accuracy"].values
            mean_val = float(np.mean(acc))
            std_val = float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0
            cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan

            rows.append({
                "feature": feature,
                "task": task,
                "n_combos": len(acc),
                "acc_min": round(float(np.min(acc)), 4),
                "acc_max": round(float(np.max(acc)), 4),
                "acc_mean": round(mean_val, 4),
                "acc_range": round(float(np.max(acc) - np.min(acc)), 4),
                "acc_std": round(std_val, 4),
                "cv_pct": round(cv, 2),
            })

    df = pd.DataFrame(rows)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


# =========================================================================
# 2. Main effects — marginal means + eta-squared
# =========================================================================


def compute_main_effects(
    master_csv: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute the independent effect of each hyperparameter on accuracy.

    For every (feature, task, parameter):
        - **marginal_mean** per level: average accuracy across all other params
        - **marginal_range**: max - min of marginal means (practical importance)
        - **eta_squared**: from one-way ANOVA (variance explained)

    Args:
        master_csv: Path to classification_master.csv.
        output_path: If set, save result CSV here.

    Returns:
        DataFrame with one row per (feature, task, parameter).
    """
    master = pd.read_csv(master_csv)
    rows = []

    for feature in sorted(master["feature_family"].unique()):
        param_cols = _get_param_cols(master, feature)
        for task in ["binary", "ternary", "five"]:
            sub = _subset(master, feature, task)
            if sub.empty:
                continue

            for param in param_cols:
                # Marginal means: group by this param, average accuracy
                marginals = sub.groupby(param)["accuracy"].mean()
                marg_range = float(marginals.max() - marginals.min())

                # One-way ANOVA: groups defined by levels of this param
                groups = [
                    g["accuracy"].values
                    for _, g in sub.groupby(param)
                ]
                # Need ≥ 2 groups with ≥ 1 observation each
                groups = [g for g in groups if len(g) > 0]

                # ANOVA requires ≥ 2 groups with ≥ 2 observations each.
                # Single-parameter features (e.g. Hjorth) have 1 obs per
                # group, making SS_within = 0 and eta² trivially 1.0.
                groups_valid = [g for g in groups if len(g) >= 2]
                min_group_size = min((len(g) for g in groups), default=0)

                if len(groups) >= 2 and min_group_size >= 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    grand_mean = sub["accuracy"].mean()
                    ss_between = sum(
                        len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
                    )
                    ss_total = float(np.sum((sub["accuracy"] - grand_mean) ** 2))
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
                else:
                    # Not enough within-group variance for meaningful ANOVA
                    f_stat, p_val, eta_sq = np.nan, np.nan, np.nan

                # Best and worst levels
                best_level = marginals.idxmax()
                worst_level = marginals.idxmin()

                rows.append({
                    "feature": feature,
                    "task": task,
                    "parameter": param,
                    "n_levels": len(marginals),
                    "marginal_range": round(marg_range, 4),
                    "eta_squared": round(float(eta_sq), 4),
                    "f_statistic": round(float(f_stat), 4) if np.isfinite(f_stat) else np.nan,
                    "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                    "best_level": best_level,
                    "worst_level": worst_level,
                })

    df = pd.DataFrame(rows)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


# =========================================================================
# 3. Interactions — practical score + optional two-way ANOVA
# =========================================================================


def compute_interactions(
    master_csv: str | Path,
    output_path: str | Path | None = None,
    force_anova: bool = False,
) -> pd.DataFrame:
    """Analyse pairwise parameter interactions for each feature × task.

    Two complementary methods:

    **Practical interaction score** (always computed):
        For each cell (level_A, level_B) of a param pair, the *expected*
        accuracy under additivity is::

            expected = grand_mean + (row_mean - grand_mean) + (col_mean - grand_mean)

        The interaction score is the RMSE of (observed - expected) across cells.
        A score near 0 means no interaction; larger values indicate synergy
        or antagonism between parameters.

    **Two-way ANOVA** (requires statsmodels):
        Proper Type-II ANOVA giving interaction F-statistic, p-value, and
        partial eta-squared.  Runs only when statsmodels is installed (or
        *force_anova* is True, which will raise ImportError if missing).

    Args:
        master_csv: Path to classification_master.csv.
        output_path: If set, save result CSV here.
        force_anova: If True, raise an error when statsmodels is missing
            instead of silently skipping the ANOVA columns.

    Returns:
        DataFrame with one row per (feature, task, param_pair).
    """
    if force_anova and not HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is required for two-way ANOVA. "
            "Install it with: pip install statsmodels"
        )

    master = pd.read_csv(master_csv)
    rows = []

    for feature in sorted(master["feature_family"].unique()):
        param_cols = _get_param_cols(master, feature)
        if len(param_cols) < 2:
            continue  # need ≥ 2 params for interaction

        for task in ["binary", "ternary", "five"]:
            sub = _subset(master, feature, task)
            if sub.empty:
                continue

            for p1, p2 in combinations(param_cols, 2):
                row = _interaction_score(sub, p1, p2, feature, task)

                # Optional: two-way ANOVA via statsmodels
                if HAS_STATSMODELS:
                    anova_row = _two_way_anova(sub, p1, p2)
                    row.update(anova_row)

                rows.append(row)

    df = pd.DataFrame(rows)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def _interaction_score(
    df: pd.DataFrame, p1: str, p2: str, feature: str, task: str
) -> dict:
    """Compute the practical interaction score for a parameter pair.

    The score is the RMSE of the residuals after removing main effects
    (additive model).  It is interpretable in the same units as accuracy.
    """
    # Cell means
    cell_means = df.groupby([p1, p2])["accuracy"].mean()

    grand_mean = df["accuracy"].mean()
    row_means = df.groupby(p1)["accuracy"].mean()
    col_means = df.groupby(p2)["accuracy"].mean()

    residuals = []
    for (r, c), obs in cell_means.items():
        expected = grand_mean + (row_means[r] - grand_mean) + (col_means[c] - grand_mean)
        residuals.append(obs - expected)

    residuals = np.array(residuals)
    interaction_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    interaction_max = float(np.max(np.abs(residuals)))

    return {
        "feature": feature,
        "task": task,
        "param_pair": f"{p1} x {p2}",
        "param_1": p1,
        "param_2": p2,
        "interaction_rmse": round(interaction_rmse, 4),
        "interaction_max_abs": round(interaction_max, 4),
        "n_cells": len(cell_means),
    }


def _two_way_anova(df: pd.DataFrame, p1: str, p2: str) -> dict:
    """Run a Type-II two-way ANOVA and return interaction statistics.

    Returns an empty dict (gracefully) if the model cannot be fitted
    (e.g. insufficient data, singular design matrix).
    """
    result: dict = {}
    try:
        # statsmodels requires string factor names without special chars
        tmp = df[[p1, p2, "accuracy"]].copy()
        tmp[p1] = tmp[p1].astype(str)
        tmp[p2] = tmp[p2].astype(str)
        tmp.columns = ["A", "B", "accuracy"]

        model = sm_ols("accuracy ~ C(A) * C(B)", data=tmp).fit()
        anova_table = anova_lm(model, typ=2)

        # Interaction row is labelled 'C(A):C(B)'
        interaction_label = "C(A):C(B)"
        if interaction_label in anova_table.index:
            row = anova_table.loc[interaction_label]
            ss_interaction = float(row["sum_sq"])
            ss_total = float(anova_table["sum_sq"].sum())
            partial_eta = ss_interaction / ss_total if ss_total > 0 else 0.0

            result["anova_F"] = round(float(row["F"]), 4)
            result["anova_p"] = float(row["PR(>F)"])
            result["anova_partial_eta_sq"] = round(partial_eta, 4)
            result["anova_significant"] = result["anova_p"] < 0.05
    except Exception as e:
        warnings.warn(f"Two-way ANOVA failed ({p1} x {p2}): {e}")

    return result


# =========================================================================
# 4. Task comparison — rank correlation + optimal-match rate
# =========================================================================


def compare_tasks(
    master_csv: str | Path,
    output_path: str | Path | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compare parameter rankings across classification tasks.

    For each feature family and each pair of tasks:
        - **Spearman rho**: rank correlation of combo accuracies
        - **optimal_match_rate**: fraction of the top-N combos in task_1 that
          also appear in the top-N of task_2.

    Combos are matched by their parameter values (not by position).

    Args:
        master_csv: Path to classification_master.csv.
        output_path: If set, save result CSV here.
        top_n: How many top combos to compare for the match rate.

    Returns:
        DataFrame with one row per (feature, task_1, task_2).
    """
    master = pd.read_csv(master_csv)
    tasks = ["binary", "ternary", "five"]
    rows = []

    for feature in sorted(master["feature_family"].unique()):
        param_cols = _get_param_cols(master, feature)

        # Collect per-task dataframes keyed by a stringified param tuple
        task_data: dict[str, pd.DataFrame] = {}
        for task in tasks:
            sub = _subset(master, feature, task)
            if sub.empty:
                continue
            # Create a hashable combo key for joining
            sub = sub.copy()
            sub["_combo_key"] = sub[param_cols].astype(str).agg("|".join, axis=1)
            task_data[task] = sub

        # Pairwise task comparison
        for i, t1 in enumerate(tasks):
            for t2 in tasks[i + 1 :]:
                if t1 not in task_data or t2 not in task_data:
                    continue

                d1 = task_data[t1].set_index("_combo_key")
                d2 = task_data[t2].set_index("_combo_key")

                # Align on shared combos
                shared = d1.index.intersection(d2.index)
                if len(shared) < 3:
                    continue

                acc1 = d1.loc[shared, "accuracy"]
                acc2 = d2.loc[shared, "accuracy"]

                # Spearman rank correlation
                rho, p_val = stats.spearmanr(acc1, acc2)

                # Optimal match: top-N overlap
                effective_n = min(top_n, len(shared))
                top1 = set(acc1.nlargest(effective_n).index)
                top2 = set(acc2.nlargest(effective_n).index)
                match_rate = len(top1 & top2) / effective_n

                # Does the best combo from t1 also rank well in t2?
                best_combo_t1 = acc1.idxmax()
                rank_in_t2 = int((acc2.rank(ascending=False))[best_combo_t1])

                # Flag unreliable correlations from small samples.
                # With n <= 10, Spearman has very low statistical power
                # and extreme rho values (including +/-1.0) can arise
                # from rank ties or trivial orderings.
                MIN_N_RELIABLE = 10
                reliable = len(shared) >= MIN_N_RELIABLE

                rows.append({
                    "feature": feature,
                    "task_1": t1,
                    "task_2": t2,
                    "n_shared_combos": len(shared),
                    "spearman_rho": round(float(rho), 4),
                    "spearman_p": float(p_val),
                    "top_n": effective_n,
                    "optimal_match_rate": round(match_rate, 2),
                    "best_t1_rank_in_t2": rank_in_t2,
                    "reliable": reliable,
                })

    df = pd.DataFrame(rows)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


# =========================================================================
# Master runner
# =========================================================================


def run_full_sensitivity_analysis(
    master_csv: str | Path = "results/tables/classification_master.csv",
    output_dir: str | Path = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Run all four sensitivity analyses and save results.

    Args:
        master_csv: Path to the merged classification results.
        output_dir: Directory for output CSVs.

    Returns:
        Dict mapping analysis name to its result DataFrame.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    print("1/4  Sensitivity scores...")
    results["sensitivity"] = compute_sensitivity_scores(
        master_csv, out / "sensitivity_scores.csv"
    )
    print(f"     {len(results['sensitivity'])} rows saved.")

    print("2/4  Main effects...")
    results["main_effects"] = compute_main_effects(
        master_csv, out / "main_effects.csv"
    )
    print(f"     {len(results['main_effects'])} rows saved.")

    print("3/4  Interactions...")
    results["interactions"] = compute_interactions(
        master_csv, out / "interactions.csv"
    )
    anova_status = "with ANOVA" if HAS_STATSMODELS else "without ANOVA (install statsmodels)"
    print(f"     {len(results['interactions'])} rows saved ({anova_status}).")

    print("4/4  Task comparison...")
    results["task_comparison"] = compare_tasks(
        master_csv, out / "task_comparison.csv"
    )
    print(f"     {len(results['task_comparison'])} rows saved.")

    return results
