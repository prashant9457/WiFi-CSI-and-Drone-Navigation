"""
classify_bvp.py
===============
Widar 3.0 — Top-5 Gesture Classifier (BVP)

Top 5 gestures selected by sample count
----------------------------------------
  1  Push & Pull        ~6 547
  2  Sweep              ~6 424
  3  Clap               ~6 421
  4  Slide              ~6 300
  5  Draw Circle (CW)   ~6 175
  ─────────────────────────────
  Total                ~31 867

Feature Engineering
-------------------
Each .mat file holds a (20, 20, T) BVP tensor with variable T.  We collapse
the time axis into three fixed-size projections that capture complementary
aspects of the gesture's velocity footprint:

  - max-projection   (20x20) -- every velocity cell ever activated
  - mean-projection  (20x20) -- average energy distribution across all frames
  - std-projection   (20x20) -- how much the velocity pattern fluctuates in time

Stacking these three channels gives a (3, 20, 20) feature map → 1 200 floats.
This fixed-size representation lets any standard classifier run without padding
or recurrent architectures.

Classifier
----------
  Primary : Random Forest (200 trees, sklearn) — fast, no GPU needed
  Optional: CNN via PyTorch — activated automatically if `torch` is importable

Outputs (saved to img/)
-----------------------
  confusion_matrix.png   — heatmap of test-set predictions vs ground truth
  feature_importance.png — top-40 Random Forest feature importance bars
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Custom modules
from bvp_loader import load_dataset as load_bvp_dataset
from bvp_plotting import plot_confusion_matrix, plot_feature_importance
from bvp_cnn import run_cnn as _try_torch_cnn

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join("code", "data", "BVP")
OUT_DIR  = "img"

# Top-5 gestures (by sample count)
TOP5 = {
    1: "Push & Pull",
    2: "Sweep",
    3: "Clap",
    4: "Slide",
    5: "Draw Circle (CW)",
}

# Map gesture IDs → consecutive class indices 0..4
LABEL_MAP   = {gid: idx for idx, gid in enumerate(sorted(TOP5))}
CLASS_NAMES = [TOP5[gid] for gid in sorted(TOP5)]

SEED        = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15   # of remaining after test split

# ---------------------------------------------------------------------------
# 1. Data loading & feature extraction
# ---------------------------------------------------------------------------

def load_dataset(root: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk the BVP directory tree and load all samples for the top-5 gestures.
    Uses caching to speed up future runs.

    Returns
    -------
    X : np.ndarray  shape (N, 1200)   feature vectors
    y : np.ndarray  shape (N,)        class indices 0..4
    """
    X, y, _, _ = load_bvp_dataset(
        root,
        gesture_ids=set(TOP5.keys()),
        cache_filename="classify_bvp_cache.npz",
    )
    # Print per-class counts
    for idx, name in enumerate(CLASS_NAMES):
        print(f"     [{idx}] {name:<22} {(y == idx).sum():>5} samples")
    print()
    return X, y



# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(SEED)
    print(f"\n{'='*55}")
    print("  Widar 3.0 — Top-5 Gesture Classifier (BVP)")
    print(f"{'='*55}\n")

    # ── Load ──────────────────────────────────────────────────────────────
    X, y = load_dataset(DATA_DIR)

    # ── Train / val / test split (stratified) ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        stratify=y_train, random_state=SEED,
    )
    print(f"Split  : train={len(y_train)}  val={len(y_val)}  test={len(y_test)}\n")

    # ── MLP Classifier ────────────────────────────────────────────────────
    print("=" * 55)
    print("  Multi-Layer Perceptron (MLP) Classifier")
    print("=" * 55)

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        )),
    ])

    t0 = time.time()
    print("  Training MLP... (using early stopping to prevent overfitting)")
    mlp.fit(X_train, y_train)
    print(f"  Trained in {time.time() - t0:.1f}s\n")

    val_acc  = accuracy_score(y_val,  mlp.predict(X_val))
    test_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"  Val  accuracy : {val_acc  * 100:.2f}%")
    print(f"  Test accuracy : {test_acc * 100:.2f}%\n")

    y_pred = mlp.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # ── Plots ─────────────────────────────────────────────────────────────
    print("Generating plots...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        "Confusion Matrix -- Test Set",
        os.path.join(OUT_DIR, "confusion_matrix.png")
    )

    # Compute connection weights heuristic for feature importance:
    # average weight magnitude of the first layer connections
    coefs = mlp.named_steps["clf"].coefs_[0]
    importances = np.mean(np.abs(coefs), axis=1)
    plot_feature_importance(importances, top_k=40,
                            out_path=os.path.join(OUT_DIR, "feature_importance.png"))

    # ── Optional CNN ──────────────────────────────────────────────────────
    _try_torch_cnn(X_train, y_train, X_test, y_test)

    print("\nDone.")
