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
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# Custom modules
from bvp_loader import load_dataset as load_bvp_dataset
from bvp_plotting import plot_confusion_matrix
from bvp_cnn import run_cnn as _try_torch_cnn

# Common package imports
from common.constants import (
    DATA_DIR,
    OUT_DIR,
    TOP5,
    TOP5_LABEL_MAP as LABEL_MAP,
    TOP5_CLASS_NAMES as CLASS_NAMES,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
)
from common.models import get_mlp_pipeline

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

    mlp = get_mlp_pipeline(SEED)

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



    # ── Optional CNN ──────────────────────────────────────────────────────
    _try_torch_cnn(X_train, y_train, X_test, y_test)

    print("\nDone.")
