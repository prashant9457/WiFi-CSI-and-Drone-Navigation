"""
bvp_plotting.py
===============
Plotting utilities for BVP domain-invariance experiment evaluations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def _dark_ax(ax):
    ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: str,
) -> None:
    """Dark-themed confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)),
                           facecolor="#12121f")
    _dark_ax(ax)
    im   = ax.imshow(cm, cmap="magma", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", color="#94a3b8", fontsize=9)
    ax.set_yticklabels(class_names, color="#94a3b8", fontsize=9)
    ax.set_xlabel("Predicted", color="#94a3b8", fontsize=11, labelpad=8)
    ax.set_ylabel("True",      color="#94a3b8", fontsize=11, labelpad=8)
    ax.set_title(title, color="#c4b5fd", fontsize=12, pad=12)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if cm[i, j] < thresh else "#12121f")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved -> {out_path}")
    plt.close(fig)

def plot_invariance_summary(
    results: list[dict],
    out_path: str,
) -> None:
    """
    Horizontal bar chart comparing achieved accuracy vs random-chance baseline
    for all classification tasks.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#12121f")
    _dark_ax(ax)

    labels   = [r["label"]            for r in results]
    achieved = [r["acc"] * 100        for r in results]
    baseline = [r["random"] * 100     for r in results]
    colors   = [r["color"]            for r in results]

    y_pos = np.arange(len(results))
    bar_h = 0.35

    # Achieved accuracy bars
    bars = ax.barh(y_pos + bar_h / 2, achieved, height=bar_h,
                   color=colors, alpha=0.85, label="Model accuracy (%)")
    # Random-chance baseline bars
    ax.barh(y_pos - bar_h / 2, baseline, height=bar_h,
            color="#334155", alpha=0.85, label="Random baseline (%)")

    # Value labels
    for bar, val in zip(bars, achieved):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Accuracy (%)", color="#94a3b8", fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_title(
        "BVP Domain-Invariance & Temporal Model Evaluation\n"
        "Preserving temporal trajectories resolves spatial shortcut learning",
        color="#c4b5fd", fontsize=12, pad=14,
    )

    legend = ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9,
                       loc="lower right")
    legend.get_frame().set_edgecolor("#334155")

    # Annotate what each result means dynamically from the result dictionaries
    for i, r in enumerate(results):
        if "ann" in r:
            line1, line2 = r["ann"]
            col = r["color"]
            ax.text(105, y_pos[i] + bar_h / 2,
                    f"{line1}\n{line2}", va="center", ha="right",
                    color=col, fontsize=7.5, style="italic",
                    bbox=dict(facecolor="#1e1e2e", edgecolor="#334155",
                              boxstyle="round,pad=0.3", alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved -> {out_path}")
    plt.close(fig)




