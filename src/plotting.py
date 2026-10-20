"""
Plotting helpers for the notebook.

"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#b5b3ac"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": SECONDARY_INK,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "font.size": 11,
        "font.family": "sans-serif",
    }
)

# Resolve relative to the repo root (this file's parent's parent) rather
# than the caller's working directory, so results always land in the same
# place whether this is run as a script from the repo root or imported
# from a notebook running with cwd=notebooks/.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _save(fig, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def plot_forest(groups: dict[str, dict], highlight: str | None = None, title: str = "") -> Path:
    """Horizontal forest plot: point estimate + 95% CI per group.
    `groups` maps a label -> the dict returned by evaluate.observed_ate.
    """
    labels = list(groups.keys())
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(labels) + 1))
    y_pos = np.arange(len(labels))[::-1]

    for y, label in zip(y_pos, labels):
        g = groups[label]
        color = ORANGE if label == highlight else BLUE
        ax.errorbar(
            g["ate"], y,
            xerr=[[g["ate"] - g["ci_lo"]], [g["ci_hi"] - g["ate"]]],
            fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4, markersize=8,
        )
        ax.annotate(
            f"{g['ate']:.1f}  (n={g['n']})",
            xy=(g["ate"], y), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9.5, color=SECONDARY_INK,
        )

    ax.axvline(0, color=MUTED, linewidth=1.2, linestyle="--", zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_ylim(y_pos.min() - 0.6, y_pos.max() + 0.75)
    ax.set_xlabel("Treatment effect on tumor burden (percentage points; negative = shrinkage)")
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=16)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "ate_forest_plot.png")


def plot_calibration(cate_hat: np.ndarray, true_cate: np.ndarray, true_responder: np.ndarray) -> Path:
    """Estimated vs. true CATE, colored by the (otherwise hidden) true
    responder label -- only plottable because this is simulated data."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, mask, color in [
        ("True responder", true_responder, ORANGE),
        ("True non-responder", ~true_responder, BLUE),
    ]:
        ax.scatter(
            true_cate[mask], cate_hat[mask],
            s=22, alpha=0.65, color=color, edgecolors="white", linewidths=0.4, label=label,
        )

    lo = min(true_cate.min(), cate_hat.min()) - 3
    hi = max(true_cate.max(), cate_hat.max()) + 3
    ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=1.2, zorder=0, label="Perfect calibration")

    ax.set_xlabel("True individual treatment effect (simulation ground truth)")
    ax.set_ylabel("Estimated treatment effect (cross-fitted T-learner)")
    ax.set_title("Estimated vs. true treatment effect, per patient", loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.tight_layout()
    return _save(fig, "cate_calibration.png")


def plot_leaf_summary(leaf_summary: pd.DataFrame, benefit_threshold: float) -> Path:
    """Horizontal bar chart of the policy tree's leaves, sorted by mean
    estimated benefit, with the discovered (responder) leaves highlighted."""
    df = leaf_summary.sort_values("mean_cate_hat", ascending=True).reset_index(drop=True)
    colors = [ORANGE if v <= benefit_threshold else MUTED for v in df["mean_cate_hat"]]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 1.2))
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df["mean_cate_hat"], color=colors, height=0.6)

    # All bars extend leftward from 0 in this application (every leaf is a
    # cost, i.e. tumor shrinkage), so put every size annotation in a single
    # readable column just to the right of the zero line rather than at each
    # bar's tip -- at the tip it would land on top of the long rule labels
    # on the left for the largest-magnitude bars.
    label_x = max(0.0, df["mean_cate_hat"].max()) + 0.4
    for y, (_, row) in zip(y_pos, df.iterrows()):
        ax.annotate(
            f"n={row['n_patients']} ({row['pct_of_population']:.1%})",
            xy=(label_x, y), va="center", ha="left", fontsize=9, color=SECONDARY_INK,
        )

    ax.axvline(benefit_threshold, color=INK, linewidth=1, linestyle=":", zorder=0)
    ax.annotate(
        f"benefit threshold ({benefit_threshold:g})", xy=(benefit_threshold, y_pos.max() + 0.35),
        xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=SECONDARY_INK,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["rule"], fontsize=8.5)
    ax.set_ylim(y_pos.min() - 0.6, y_pos.max() + 0.75)
    # Bars are anchored ("sticky") to the x=0 baseline, so ax.margins()
    # will not add whitespace past 0 no matter what -- reserve room for the
    # size-annotation column explicitly instead.
    xmin, _ = ax.get_xlim()
    ax.set_xlim(xmin * 1.08, abs(xmin) * 0.22)
    ax.set_xlabel("Mean estimated treatment effect in leaf (percentage points)")
    ax.set_title("Policy tree leaves, ranked by estimated benefit", loc="left", fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "policy_tree_leaves.png")


def plot_policy_tree(tree, feature_names: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        tree, feature_names=feature_names, filled=True, rounded=True,
        fontsize=9, ax=ax, precision=1,
    )
    ax.set_title("Policy tree: predicting estimated treatment effect from baseline covariates",
                 loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "policy_tree_diagram.png")


def plot_cate_distribution(cate_hat: np.ndarray, benefit_threshold: float) -> Path:
    """Histogram of the estimated per-patient treatment effect across the
    whole trial -- shows most patients clustered near the modest
    background effect, with a small tail of large responders."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cate_hat, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(benefit_threshold, color=ORANGE, linewidth=1.5, linestyle="--")
    ax.annotate(
        f"benefit threshold ({benefit_threshold:g})", xy=(benefit_threshold, ax.get_ylim()[1] * 0.95),
        xytext=(6, 0), textcoords="offset points", fontsize=9, color=ORANGE, fontweight="bold",
    )
    ax.set_xlabel("Estimated individual treatment effect (percentage points)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Distribution of estimated treatment effect across the trial", loc="left", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    return _save(fig, "cate_distribution.png")
