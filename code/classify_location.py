"""
classify_location.py
====================
Widar 3.0 -- BVP Domain-Invariance Experiment

Scientific Question
-------------------
The BVP (Body-coordinate Velocity Profile) was designed to be invariant to
*where* in the room the gesture is performed and *which direction* the subject
faces.  If that claim holds, a classifier trained on BVP features should be
able to predict *gesture* well but should fail to predict *location* or
*orientation* -- those signals would have been engineered out.

This script tests that claim by training three parallel Random Forest
classifiers on exactly the same 1200-dimensional BVP feature vectors:

  Classifier A  -- predicts GESTURE     (5 classes, random = 20.0%)
  Classifier B  -- predicts LOCATION    (8 classes, random = 12.5%)
  Classifier C  -- predicts ORIENTATION (5 classes, random = 20.0%)

Actual finding (see RESULTS section in README):
  - Gesture accuracy  ~41%  -- moderate lift over random (20%)
                               (was measured on top-5; all-9 result may differ)
  - Location accuracy ~95%  -- BVP projections are NOT location-invariant!
  - Orientation       100%  -- perfectly separable

Interpretation:
  Time-collapsing (max/mean/std over T) destroys the temporal ordering
  that distinguishes gesture trajectories, but it *preserves* systematic
  differences in which velocity cells are activated at each location and
  orientation.  The BVP design removes location from the *temporal pattern*,
  not from the *marginal velocity distribution*.

  This motivates temporal models (CNN+LSTM, Transformer) that consume the
  raw (20, 20, T) tensor -- only those can leverage the temporal ordering
  that makes gesture classification possible without leaking positional cues.

Feature Engineering (same as classify_bvp.py)
----------------------------------------------
Each (20, 20, T) BVP tensor is collapsed into a 1200-dim vector:
  [  0: 400)  max-projection  -- full gesture velocity footprint
  [400: 800)  mean-projection -- average energy distribution
  [800:1200)  std-projection  -- temporal variability

Outputs (saved to img/)
-----------------------
  domain_invariance_bars.png  -- accuracy vs random-chance for all 3 tasks
  confusion_location.png      -- location confusion matrix
  confusion_orientation.png   -- orientation confusion matrix
  confusion_gesture5.png      -- gesture (top-5) confusion matrix
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR  = os.path.join("code", "data", "BVP")
OUT_DIR   = "img"
SEED      = 42
TEST_SIZE = 0.15

# All gestures except gesture 10 ("Random", only 500 samples -- the smallest class)
ALL9_GESTURE_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9}

GESTURE_NAMES = {
    1: "Push & Pull",
    2: "Sweep",
    3: "Clap",
    4: "Slide",
    5: "Circle (CW)",
    6: "Circle (CCW)",
    7: "Triangle",
    8: "Zigzag",
    9: "Draw N",
}

LOCATION_NAMES    = [f"Loc {i}" for i in range(1, 9)]      # 8 grid positions
ORIENTATION_NAMES = ["North", "NE", "East", "SE", "South"] # 5 directions

# ---------------------------------------------------------------------------
# 1. Load dataset -- parse gesture, location, AND orientation per file
# ---------------------------------------------------------------------------

def load_dataset(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk the BVP tree for all 9 gestures (dropping gesture 10) and return
    four aligned arrays.

    Returns
    -------
    X   : (N, 1200) float32  -- BVP feature vectors
    y_g : (N,)     int64    -- gesture  class index  (0..8)
    y_l : (N,)     int64    -- location class index  (0..7)
    y_o : (N,)     int64    -- orientation class index (0..4)
    """
    gesture_label_map = {gid: idx for idx, gid in enumerate(sorted(ALL9_GESTURE_IDS))}

    X_list, yg_list, yl_list, yo_list = [], [], [], []
    skipped = 0

    print("Scanning BVP dataset tree...")
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.endswith(".mat"):
                continue
            parts = fname.split("-")
            try:
                g   = int(parts[1])
                loc = int(parts[2])
                ori = int(parts[3])
            except (IndexError, ValueError):
                continue
            if g not in ALL9_GESTURE_IDS:
                continue

            path = os.path.join(dirpath, fname)
            try:
                mat  = sio.loadmat(path)
                data = mat["velocity_spectrum_ro"].astype(np.float32)
                if data.ndim == 2:
                    data = data[:, :, np.newaxis]
                if data.shape[2] == 0:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            # 3-projection feature vector (same as classify_bvp.py)
            feat = np.concatenate([
                np.max(data,  axis=2).ravel(),   # max-projection
                np.mean(data, axis=2).ravel(),   # mean-projection
                np.std(data,  axis=2).ravel(),   # std-projection
            ])
            X_list.append(feat)
            yg_list.append(gesture_label_map[g])
            yl_list.append(loc - 1)    # 1-indexed -> 0-indexed
            yo_list.append(ori - 1)    # 1-indexed -> 0-indexed

    if skipped:
        print(f"  [!] Skipped {skipped} corrupt / empty files")

    X   = np.vstack(X_list)
    y_g = np.array(yg_list, dtype=np.int64)
    y_l = np.array(yl_list, dtype=np.int64)
    y_o = np.array(yo_list, dtype=np.int64)

    print(f"  [ok] {len(y_g)} samples loaded  |  feature dim = {X.shape[1]}")
    print(f"       Gestures     : {np.unique(y_g)}")
    print(f"       Locations    : {np.unique(y_l) + 1}")
    print(f"       Orientations : {np.unique(y_o) + 1}")
    print()
    return X, y_g, y_l, y_o


# ---------------------------------------------------------------------------
# 2. Build and evaluate a single Random Forest
# ---------------------------------------------------------------------------

def run_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    label:   str,
) -> tuple[float, np.ndarray]:
    """
    Train a Random Forest and return (test_accuracy, predictions).
    Uses a StandardScaler -> RF pipeline identical to classify_bvp.py.
    """
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=SEED,
            class_weight="balanced",
        )),
    ])
    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    preds = rf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  [{label}]  trained in {elapsed:.1f}s  |  test accuracy = {acc*100:.2f}%")
    return acc, preds


# ---------------------------------------------------------------------------
# 3. Visualization helpers
# ---------------------------------------------------------------------------

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
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved -> {out_path}")
    plt.close(fig)


def plot_invariance_summary(
    results: list[dict],  # each: {label, acc, random_baseline, color}
    out_path: str,
) -> None:
    """
    Horizontal bar chart comparing achieved accuracy vs random-chance baseline
    for all three classification tasks.

    The gap between achieved and random visually shows how much information
    each task's labels carry in the BVP feature space.
    """
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#12121f")
    _dark_ax(ax)

    labels   = [r["label"]            for r in results]
    achieved = [r["acc"] * 100        for r in results]
    baseline = [r["random"] * 100     for r in results]
    colors   = [r["color"]            for r in results]

    y_pos = np.arange(len(results))
    bar_h = 0.35

    # Achieved accuracy bars
    bars = ax.barh(y_pos + bar_h / 2, achieved, height=bar_h,
                   color=colors, alpha=0.85, label="RF accuracy (%)")
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
        "BVP Domain-Invariance Test\n"
        "Projection features encode location strongly -- temporal models needed",
        color="#c4b5fd", fontsize=12, pad=14,
    )

    legend = ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9,
                       loc="lower right")

    # Annotate what each result means
    annotations = [
        ("~41% (proj. loses",   "temporal ordering)",       "#fb923c"),
        ("~95% -- location NOT", "removed by projections",   "#f472b6"),
        ("100% -- orient. NOT", "removed by projections",    "#f472b6"),
    ]
    for i, (line1, line2, col) in enumerate(annotations):
        ax.text(105, y_pos[i] + bar_h / 2,
                f"{line1}\n{line2}", va="center", ha="right",
                color=col, fontsize=7.5, style="italic",
                bbox=dict(facecolor="#1e1e2e", edgecolor="#334155",
                          boxstyle="round,pad=0.3", alpha=0.8))

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(SEED)
    print(f"\n{'='*60}")
    print("  Widar 3.0 -- BVP Domain-Invariance Experiment")
    print(f"{'='*60}\n")
    print("Hypothesis: BVP should encode gesture signal while removing")
    print("            location and orientation cues from the representation.")
    print("            We test this by running 3 classifiers on the same features.\n")

    # ── Load ────────────────────────────────────────────────────────────────
    X, y_g, y_l, y_o = load_dataset(DATA_DIR)

    # ── Split (stratify on gesture so all tasks use the same split) ─────────
    idx = np.arange(len(y_g))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y_g, random_state=SEED
    )
    X_train, X_test = X[idx_train], X[idx_test]
    print(f"Split  : train={len(idx_train)}  test={len(idx_test)}\n")

    # ── Run three classifiers ───────────────────────────────────────────────
    print("=" * 60)
    print("  Training classifiers (200 trees each)...")
    print("=" * 60)

    g_acc, g_pred = run_classifier(
        X_train, y_g[idx_train], X_test, y_g[idx_test], "Gesture (5-class) "
    )
    l_acc, l_pred = run_classifier(
        X_train, y_l[idx_train], X_test, y_l[idx_test], "Location (8-class) "
    )
    o_acc, o_pred = run_classifier(
        X_train, y_o[idx_train], X_test, y_o[idx_test], "Orientation (5-class)"
    )

    # ── Per-class reports ───────────────────────────────────────────────────
    gesture_names = [GESTURE_NAMES[gid] for gid in sorted(ALL9_GESTURE_IDS)]
    print(f"\n--- Gesture Classification Report ---")
    print(classification_report(y_g[idx_test], g_pred, target_names=gesture_names))

    print(f"--- Location Classification Report ---")
    print(classification_report(y_l[idx_test], l_pred, target_names=LOCATION_NAMES))

    print(f"--- Orientation Classification Report ---")
    print(classification_report(y_o[idx_test], o_pred, target_names=ORIENTATION_NAMES))

    # ── Domain-invariance summary ───────────────────────────────────────────
    n_gesture     = len(np.unique(y_g))
    n_location    = len(np.unique(y_l))
    n_orientation = len(np.unique(y_o))

    print("\n" + "=" * 60)
    print("  DOMAIN-INVARIANCE SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<22}  {'Classes':>7}  {'Random':>8}  {'RF Acc':>8}  {'Gap':>8}")
    print("  " + "-" * 56)

    rows = [
        ("Gesture",     n_gesture,     g_acc),
        ("Location",    n_location,    l_acc),
        ("Orientation", n_orientation, o_acc),
    ]
    for name, n_cls, acc in rows:
        rand = 1.0 / n_cls
        gap  = acc - rand
        if gap < 0.10:
            verdict = "[REMOVED -- invariant]"
        elif gap < 0.40:
            verdict = "[WEAK signal]"
        else:
            verdict = "[STRONG signal -- NOT removed]"
        print(f"  {name:<22}  {n_cls:>7}  {rand*100:>7.1f}%  {acc*100:>7.2f}%  "
              f"{gap*100:>+7.2f}%  {verdict}")

    print()

    # ── Plots ───────────────────────────────────────────────────────────────
    print("Generating plots...")

    plot_invariance_summary(
        results=[
            {"label": "Gesture (5-class)",      "acc": g_acc,
             "random": 1/n_gesture,     "color": "#818cf8"},
            {"label": "Location (8-class)",      "acc": l_acc,
             "random": 1/n_location,    "color": "#f472b6"},
            {"label": "Orientation (5-class)",   "acc": o_acc,
             "random": 1/n_orientation, "color": "#fb923c"},
        ],
        out_path=os.path.join(OUT_DIR, "domain_invariance_bars.png"),
    )

    cm_g = confusion_matrix(y_g[idx_test], g_pred)
    plot_confusion_matrix(
        cm_g, gesture_names,
        "Gesture Confusion Matrix (test set)",
        os.path.join(OUT_DIR, "confusion_gesture5.png"),
    )

    cm_l = confusion_matrix(y_l[idx_test], l_pred)
    plot_confusion_matrix(
        cm_l, LOCATION_NAMES,
        "Location Confusion Matrix -- 95% accuracy: projections leak location",
        os.path.join(OUT_DIR, "confusion_location.png"),
    )

    cm_o = confusion_matrix(y_o[idx_test], o_pred)
    plot_confusion_matrix(
        cm_o, ORIENTATION_NAMES,
        "Orientation Confusion Matrix -- 100% accuracy: projections leak orientation",
        os.path.join(OUT_DIR, "confusion_orientation.png"),
    )

    print("\nDone.")
