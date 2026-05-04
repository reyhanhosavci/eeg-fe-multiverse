"""
Fig 10 — Time-Domain Features: Hurst Exponent and Hjorth Parameters.

Top row: Hurst heatmaps (method × min_window) for binary, ternary, five-class.
Bottom row: Hjorth grouped bar charts (window_size_sec) for binary, ternary, five-class.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Global style (consistent with visualization.py) ──────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

TASK_ORDER = ["binary", "ternary", "five"]
TASK_COLORS = {"binary": "steelblue", "ternary": "darkorange", "five": "forestgreen"}
HEATMAP_CMAP = "RdYlGn"

TABLE_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")


def _load(feature: str, task: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / f"classification_{feature}_{task}.csv")


def generate_fig10() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ── Top row: Hurst heatmaps ──────────────────────────────────────────
    for col, task in enumerate(TASK_ORDER):
        ax = axes[0, col]
        df = _load("hurst", task)
        pivot = df.pivot_table(
            index="method", columns="min_window", values="accuracy", aggfunc="mean",
        )
        # Enforce row/col order
        row_order = ["rescaled_range", "dfa", "wavelet"]
        col_order = sorted(pivot.columns)
        pivot = pivot.reindex(index=row_order, columns=col_order)

        vmin = float(pivot.min().min())
        vmax = float(pivot.max().max())

        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=HEATMAP_CMAP,
            vmin=vmin, vmax=vmax, ax=ax, cbar_kws={"shrink": 0.8},
            linewidths=0.5, linecolor="white", annot_kws={"fontsize": 11},
        )
        ax.set_title(f"{task.capitalize()}\n({vmin:.3f} \u2013 {vmax:.3f})", fontsize=11)
        ax.set_xlabel("min_window", fontsize=12)
        if col == 0:
            ax.set_ylabel("method", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=10)

    # ── Bottom row: Hjorth bar charts ────────────────────────────────────
    for col, task in enumerate(TASK_ORDER):
        ax = axes[1, col]
        df = _load("hjorth", task)
        df = df.sort_values("window_size_sec", key=lambda s: s.replace("full", "999").astype(float))

        x_labels = [str(v) for v in df["window_size_sec"]]
        accuracies = df["accuracy"].values
        x_pos = np.arange(len(x_labels))

        bars = ax.bar(
            x_pos, accuracies, color=TASK_COLORS[task],
            edgecolor="white", width=0.65,
        )

        # Annotate bars
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9, rotation=0,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel("window_size (s)", fontsize=12)

        ymin = accuracies.min() - 0.01
        ymax = accuracies.max() + 0.015  # extra room for annotations
        ax.set_ylim(ymin, ymax)

        ax.set_title(f"{task.capitalize()}", fontsize=11)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Time-Domain Features: Hurst Exponent and Hjorth Parameters",
        fontsize=14, fontweight="bold", y=0.97,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save ─────────────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    name = "fig10_timedomain_heatmap"
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")

    # Copy PNG to project root for paper convenience
    shutil.copy2(FIG_DIR / f"{name}.png", Path(f"{name}.png"))
    print(f"  Copied: {name}.png -> project root")


if __name__ == "__main__":
    generate_fig10()
