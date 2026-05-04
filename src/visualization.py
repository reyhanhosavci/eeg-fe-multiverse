"""
Paper figure generation for the Feature Extraction Multiverse project.

Produces 8 publication-ready figures (300 DPI, PDF + PNG) under
results/figures/.  All figures share a consistent colour scheme:

    - Tasks:    binary = blue,  ternary = orange,  five = green
    - Heatmaps: RdYlGn  (red = low accuracy, green = high accuracy)

Usage:
    python -c "from src.visualization import generate_all_figures; generate_all_figures()"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =========================================================================
# Global style
# =========================================================================

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

TASK_COLORS = {"binary": "#1f77b4", "ternary": "#ff7f0e", "five": "#2ca02c"}
TASK_ORDER = ["binary", "ternary", "five"]
HEATMAP_CMAP = "RdYlGn"

FIG_DIR = Path("results/figures")


def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure as both PNG (300 DPI) and PDF."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


def _load_master() -> pd.DataFrame:
    """Load the classification master CSV."""
    return pd.read_csv("results/tables/classification_master.csv")


def _load_classification_csv(feature: str, task: str) -> pd.DataFrame:
    """Load a single classification result CSV."""
    path = Path(f"results/tables/classification_{feature}_{task}.csv")
    return pd.read_csv(path)


# =========================================================================
# Fig 1 — Sensitivity overview (grouped horizontal bar chart)
# =========================================================================


def fig1_sensitivity_overview() -> None:
    """Grouped bar chart: accuracy range per feature, 3 tasks side by side."""
    print("Fig 1: Sensitivity overview...")
    scores = pd.read_csv("results/tables/sensitivity_scores.csv")

    # Sort features by binary range descending
    feat_order = (
        scores[scores["task"] == "binary"]
        .sort_values("acc_range", ascending=True)["feature"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_height = 0.25
    y_base = np.arange(len(feat_order))

    for i, task in enumerate(TASK_ORDER):
        sub = scores[scores["task"] == task].set_index("feature")
        vals = [sub.loc[f, "acc_range"] * 100 if f in sub.index else 0
                for f in feat_order]
        ax.barh(
            y_base + i * bar_height, vals, bar_height,
            label=task.capitalize(), color=TASK_COLORS[task], edgecolor="white",
        )

    ax.set_yticks(y_base + bar_height)
    ax.set_yticklabels([f.upper() for f in feat_order])
    ax.set_xlabel("Accuracy Range (percentage points)")
    ax.set_title("Feature Extraction Sensitivity: How Much Do Hyperparameters Matter?")
    ax.legend(title="Task", loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    _save_fig(fig, "fig1_sensitivity_overview")


# =========================================================================
# Fig 2–5 — Heatmaps (shared helper)
# =========================================================================


def _heatmap_triplet(
    feature: str,
    row_param: str,
    col_param: str,
    fig_name: str,
    title: str,
    figsize: tuple = (11, 3.8),
    marginalise_over: str | None = None,
) -> None:
    """Draw 3 side-by-side heatmaps (binary / ternary / five).

    Args:
        feature: Feature family name (e.g. 'psd').
        row_param: Parameter for heatmap rows (y-axis).
        col_param: Parameter for heatmap columns (x-axis).
        fig_name: Output file stem.
        title: Suptitle text.
        figsize: Figure dimensions.
        marginalise_over: If set, average accuracy over this parameter
            before pivoting (for 3-param features).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, task in zip(axes, TASK_ORDER):
        df = _load_classification_csv(feature, task)

        if marginalise_over and marginalise_over in df.columns:
            df = df.groupby([row_param, col_param], as_index=False)["accuracy"].mean()

        pivot = df.pivot_table(
            index=row_param, columns=col_param, values="accuracy", aggfunc="mean",
        )

        vmin = float(pivot.min().min())
        vmax = float(pivot.max().max())

        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
            vmin=vmin, vmax=vmax, ax=ax, cbar_kws={"shrink": 0.8},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(f"{task.capitalize()}\n({vmin:.3f} – {vmax:.3f})", fontsize=11)

        # Only show y-label on leftmost subplot
        if ax != axes[0]:
            ax.set_ylabel("")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, fig_name)


def fig2_psd_heatmap() -> None:
    """PSD: window x nperseg heatmap (averaged over overlap_ratio)."""
    print("Fig 2: PSD heatmap...")
    _heatmap_triplet(
        feature="psd",
        row_param="window",
        col_param="nperseg",
        fig_name="fig2_psd_heatmap",
        title="PSD (Welch): Window Type x Segment Length",
        marginalise_over="overlap_ratio",
    )


def fig3_sampen_heatmap() -> None:
    """SampEn: m x r_multiplier heatmap."""
    print("Fig 3: SampEn heatmap...")
    _heatmap_triplet(
        feature="sampen",
        row_param="m",
        col_param="r_multiplier",
        fig_name="fig3_sampen_heatmap",
        title="Sample Entropy: Embedding Dimension (m) x Tolerance (r)",
    )


def fig4_permen_heatmap() -> None:
    """PermEn: order x delay heatmap."""
    print("Fig 4: PermEn heatmap...")
    _heatmap_triplet(
        feature="permen",
        row_param="order",
        col_param="delay",
        fig_name="fig4_permen_heatmap",
        title="Permutation Entropy: Order x Delay",
    )


def fig5_dwt_heatmap() -> None:
    """DWT: wavelet x level heatmap."""
    print("Fig 5: DWT heatmap...")
    _heatmap_triplet(
        feature="dwt",
        row_param="wavelet",
        col_param="level",
        fig_name="fig5_dwt_heatmap",
        title="DWT: Wavelet Family x Decomposition Level",
    )


# =========================================================================
# Fig 6 — Main effects eta-squared
# =========================================================================


def fig6_main_effects() -> None:
    """Horizontal bar chart of eta-squared for each feature's dominant parameter.

    Single-parameter features (e.g. Hjorth) have undefined eta-squared
    (1 observation per group) and are excluded from the ANOVA chart.
    A footnote is added to explain their omission.
    """
    print("Fig 6: Main effects...")
    effects = pd.read_csv("results/tables/main_effects.csv")

    # Drop rows where eta_squared is NaN (single-param features)
    effects_valid = effects.dropna(subset=["eta_squared"])

    # For each (feature, task): pick the parameter with highest eta_squared
    idx = effects_valid.groupby(["feature", "task"])["eta_squared"].idxmax()
    dominant = effects_valid.loc[idx].copy()

    # Focus on binary task for clarity, sorted
    dom_binary = dominant[dominant["task"] == "binary"].sort_values(
        "eta_squared", ascending=True
    )

    # Identify excluded features for footnote
    all_features = set(effects["feature"].unique())
    included_features = set(dom_binary["feature"].unique())
    excluded = all_features - included_features

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = [
        f"{row['feature'].upper()}\n({row['parameter']})"
        for _, row in dom_binary.iterrows()
    ]
    colors = [
        "#d62728" if row["eta_squared"] > 0.5 else
        "#ff7f0e" if row["eta_squared"] > 0.25 else "#2ca02c"
        for _, row in dom_binary.iterrows()
    ]

    bars = ax.barh(labels, dom_binary["eta_squared"], color=colors, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, dom_binary["eta_squared"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    ax.set_xlabel("Eta-squared (variance explained)")
    ax.set_title("Main Effect: Dominant Hyperparameter per Feature (Binary Task)")
    ax.set_xlim(0, 1.1)
    ax.axvline(0.14, color="gray", linestyle="--", alpha=0.5, label="Large effect (0.14)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    if excluded:
        note = (
            f"* {', '.join(f.upper() for f in sorted(excluded))}: "
            f"single-parameter (1 obs/group), eta-squared undefined; "
            f"see accuracy range instead."
        )
        fig.text(0.12, -0.02, note, fontsize=7, style="italic", color="gray")

    _save_fig(fig, "fig6_main_effects")


def fig6_main_effects_all_tasks() -> None:
    """Three-panel main effects chart: binary, ternary, five side by side.

    Each panel shows the dominant hyperparameter's eta-squared per feature
    for that task.  Single-parameter features (NaN eta-squared) are excluded.
    """
    print("Fig 6 (all tasks): Main effects...")
    effects = pd.read_csv("results/tables/main_effects.csv")
    effects_valid = effects.dropna(subset=["eta_squared"])

    idx = effects_valid.groupby(["feature", "task"])["eta_squared"].idxmax()
    dominant = effects_valid.loc[idx].copy()

    # Consistent feature order across panels (sorted by binary eta-sq)
    dom_binary = dominant[dominant["task"] == "binary"].sort_values("eta_squared")
    feat_order = dom_binary["feature"].tolist()

    # Excluded features
    all_features = set(effects["feature"].unique())
    excluded = all_features - set(feat_order)

    task_titles = {"binary": "Binary", "ternary": "Ternary", "five": "Five-class"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)

    for ax, task in zip(axes, ["binary", "ternary", "five"]):
        dom = dominant[dominant["task"] == task].copy()
        # Align to feat_order (some features may be missing for a task)
        dom = dom.set_index("feature").reindex(feat_order).reset_index()

        labels = [
            f"{row['feature'].upper()}\n({row['parameter']})"
            if pd.notna(row.get("parameter")) else row["feature"].upper()
            for _, row in dom.iterrows()
        ]
        vals = dom["eta_squared"].fillna(0).values

        colors = [
            "#d62728" if v > 0.5 else "#ff7f0e" if v > 0.25 else "#2ca02c"
            for v in vals
        ]

        bars = ax.barh(labels, vals, color=colors, edgecolor="white")

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8,
                )

        ax.set_xlim(0, 1.1)
        ax.set_title(task_titles[task], fontsize=12)
        ax.axvline(0.14, color="gray", linestyle="--", alpha=0.4)
        ax.grid(axis="x", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("")
        if ax == axes[1]:
            ax.set_xlabel("Eta-squared (variance explained)")

    fig.suptitle(
        "Main Effect: Dominant Hyperparameter per Feature Across Tasks",
        fontsize=13, y=1.02,
    )

    if excluded:
        note = (
            f"* {', '.join(f.upper() for f in sorted(excluded))}: "
            f"single-parameter, eta-squared undefined."
        )
        fig.text(0.08, -0.02, note, fontsize=7, style="italic", color="gray")

    fig.tight_layout()
    _save_fig(fig, "fig6_main_effects_all_tasks")


def fig6_main_effects_grouped() -> None:
    """Grouped bar chart: for each feature, show the binary-dominant
    parameter's eta-squared across all 3 tasks side by side.

    The dominant parameter is chosen from the binary task and kept
    fixed so rows are consistent across panels.
    """
    print("Fig 6 (grouped): Main effects...")
    effects = pd.read_csv("results/tables/main_effects.csv")
    effects_valid = effects.dropna(subset=["eta_squared"])

    # Pick each feature's dominant parameter from binary task
    binary = effects_valid[effects_valid["task"] == "binary"]
    idx = binary.groupby("feature")["eta_squared"].idxmax()
    dominant_params = binary.loc[idx][["feature", "parameter"]].set_index("feature")

    # Sort features by binary eta-squared descending
    binary_sorted = binary.loc[idx].sort_values("eta_squared", ascending=True)
    feat_order = binary_sorted["feature"].tolist()

    # For each feature + task, look up eta-squared of the SAME parameter
    task_order = ["binary", "ternary", "five"]
    task_colors = {"binary": "#1f77b4", "ternary": "#ff7f0e", "five": "#2ca02c"}

    data = {t: [] for t in task_order}
    for feat in feat_order:
        param = dominant_params.loc[feat, "parameter"]
        for task in task_order:
            row = effects_valid[
                (effects_valid["feature"] == feat)
                & (effects_valid["task"] == task)
                & (effects_valid["parameter"] == param)
            ]
            val = float(row["eta_squared"].values[0]) if len(row) > 0 else 0.0
            data[task].append(val)

    # Excluded features
    all_features = set(effects["feature"].unique())
    excluded = all_features - set(feat_order)

    # Build labels
    labels = [
        f"{feat.upper()} ({dominant_params.loc[feat, 'parameter']})"
        for feat in feat_order
    ]

    y_pos = np.arange(len(feat_order))
    bar_h = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, task in enumerate(task_order):
        bars = ax.barh(
            y_pos + i * bar_h, data[task], bar_h,
            label=task.capitalize(), color=task_colors[task], edgecolor="white",
        )
        for bar, v in zip(bars, data[task]):
            if v > 0.05:
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=7,
                )

    ax.set_yticks(y_pos + bar_h)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Eta-squared (variance explained)")
    ax.set_title("Main Effect of Dominant Hyperparameter Across Tasks")
    ax.set_xlim(0, 1.15)
    ax.axvline(0.14, color="gray", linestyle="--", alpha=0.4, label="Large effect (0.14)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    if excluded:
        note = (
            f"* {', '.join(f.upper() for f in sorted(excluded))}: "
            f"single-parameter, eta-squared undefined."
        )
        fig.text(0.10, -0.02, note, fontsize=7, style="italic", color="gray")

    fig.tight_layout()
    _save_fig(fig, "fig6_main_effects_all_tasks_group")


# =========================================================================
# Fig 7 — Task comparison (Spearman rho heatmap)
# =========================================================================


def fig7_task_comparison() -> None:
    """Heatmap of Spearman rho across task pairs for each feature.

    Cells based on fewer than 10 parameter combinations are marked with
    a dagger symbol to warn that the correlation is unreliable.
    """
    print("Fig 7: Task comparison...")
    tc = pd.read_csv("results/tables/task_comparison.csv")
    tc["pair"] = tc["task_1"] + " vs " + tc["task_2"]

    pair_order = ["binary vs ternary", "binary vs five", "ternary vs five"]
    feat_order = sorted(tc["feature"].unique())

    pivot_rho = tc.pivot_table(index="feature", columns="pair", values="spearman_rho")
    pivot_rho = pivot_rho.reindex(index=feat_order, columns=pair_order)

    pivot_n = tc.pivot_table(index="feature", columns="pair", values="n_shared_combos")
    pivot_n = pivot_n.reindex(index=feat_order, columns=pair_order)

    # Build annotation matrix: rho value + dagger for small n
    MIN_N = 10
    annot = pivot_rho.copy().astype(str)
    for feat in feat_order:
        for pair in pair_order:
            rho_val = pivot_rho.loc[feat, pair] if feat in pivot_rho.index else np.nan
            n_val = pivot_n.loc[feat, pair] if feat in pivot_n.index else 0
            if pd.isna(rho_val):
                annot.loc[feat, pair] = ""
            elif n_val < MIN_N:
                annot.loc[feat, pair] = f"{rho_val:.2f}*"
            else:
                annot.loc[feat, pair] = f"{rho_val:.2f}"

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.heatmap(
        pivot_rho, annot=annot, fmt="", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Spearman rho"},
    )
    ax.set_title("Do Optimal Parameters Transfer Across Tasks?")
    ax.set_ylabel("")
    ax.set_yticklabels([f.upper() for f in feat_order], rotation=0)

    fig.text(
        0.12, -0.01,
        "* n < 10 combinations -- correlation may be unreliable due to low statistical power.",
        fontsize=7, style="italic", color="gray",
    )

    _save_fig(fig, "fig7_task_comparison")


# =========================================================================
# Fig 8 — FuzzyEn interaction heatmap (m x n)
# =========================================================================


def fig8_fuzzen_interaction() -> None:
    """FuzzyEn m x n: observed accuracy + residual (interaction effect)."""
    print("Fig 8: FuzzyEn interaction...")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    for j, task in enumerate(TASK_ORDER):
        df = _load_classification_csv("fuzzen", task)
        # Average over r_multiplier
        avg = df.groupby(["m", "n"], as_index=False)["accuracy"].mean()
        pivot_obs = avg.pivot_table(index="m", columns="n", values="accuracy")

        # Compute expected under additivity
        grand = pivot_obs.values.mean()
        row_effects = pivot_obs.mean(axis=1) - grand
        col_effects = pivot_obs.mean(axis=0) - grand
        expected = pd.DataFrame(
            grand + np.add.outer(row_effects.values, col_effects.values),
            index=pivot_obs.index, columns=pivot_obs.columns,
        )
        residual = pivot_obs - expected

        # Top row: observed accuracy
        ax_obs = axes[0, j]
        vmin_o = float(pivot_obs.min().min())
        vmax_o = float(pivot_obs.max().max())
        sns.heatmap(
            pivot_obs, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
            vmin=vmin_o, vmax=vmax_o, ax=ax_obs,
            linewidths=0.5, linecolor="white",
        )
        ax_obs.set_title(f"{task.capitalize()} — Observed\n({vmin_o:.3f} – {vmax_o:.3f})")
        if j > 0:
            ax_obs.set_ylabel("")

        # Bottom row: interaction residual
        ax_res = axes[1, j]
        abs_max = max(abs(residual.min().min()), abs(residual.max().max()))
        sns.heatmap(
            residual, annot=True, fmt="+.3f", cmap="RdBu_r",
            center=0, vmin=-abs_max, vmax=abs_max, ax=ax_res,
            linewidths=0.5, linecolor="white",
        )
        ax_res.set_title(f"{task.capitalize()} — Interaction Residual")
        if j > 0:
            ax_res.set_ylabel("")

    fig.suptitle(
        "FuzzyEn: m x n Interaction (averaged over r_multiplier)", fontsize=13, y=1.01
    )
    fig.tight_layout()
    _save_fig(fig, "fig8_fuzzen_interaction")


# =========================================================================
# Master runner
# =========================================================================


def generate_all_figures() -> None:
    """Generate all 8 paper figures."""
    print(f"Output directory: {FIG_DIR.resolve()}\n")

    fig1_sensitivity_overview()
    fig2_psd_heatmap()
    fig3_sampen_heatmap()
    fig4_permen_heatmap()
    fig5_dwt_heatmap()
    fig6_main_effects()
    fig7_task_comparison()
    fig8_fuzzen_interaction()

    n_files = len(list(FIG_DIR.glob("*")))
    print(f"\nDone — {n_files} files in {FIG_DIR}/")
