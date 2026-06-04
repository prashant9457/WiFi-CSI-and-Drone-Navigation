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
from bvp_plotting import _dark_ax, plot_confusion_matrix

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
# 2. Visualization helpers
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importances: np.ndarray, top_k: int, out_path: str
) -> None:
    """
    Bar chart of the top-k most important feature dimensions.

    The 1200 features are laid out as:
      [  0: 400)  max-projection  cells  (row-major 20×20)
      [400: 800)  mean-projection cells
      [800:1200)  std-projection  cells
    """
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_imp = importances[top_idx]

    def _channel_label(i: int) -> str:
        if i < 400:
            r, c = divmod(i, 20)
            return f"max [{r},{c}]"
        elif i < 800:
            r, c = divmod(i - 400, 20)
            return f"mean[{r},{c}]"
        else:
            r, c = divmod(i - 800, 20)
            return f"std [{r},{c}]"

    labels = [_channel_label(i) for i in top_idx]
    colors = [
        "#818cf8" if i < 400 else "#f472b6" if i < 800 else "#34d399"
        for i in top_idx
    ]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#12121f")
    _dark_ax(ax)
    ax.barh(range(top_k)[::-1], top_imp, color=colors, alpha=0.85)
    ax.set_yticks(range(top_k)[::-1])
    ax.set_yticklabels(labels, fontsize=8, color="#94a3b8")
    ax.set_xlabel("Importance", color="#94a3b8", fontsize=10)
    ax.set_title(
        f"MLP Connection Weights — Top {top_k} Feature Importances\n"
        "■ max-projection  ■ mean-projection  ■ std-projection",
        color="#c4b5fd", fontsize=11, pad=10,
    )

    # Legend patches
    from matplotlib.patches import Patch
    legend = ax.legend(
        handles=[
            Patch(color="#818cf8", label="max-projection"),
            Patch(color="#f472b6", label="mean-projection"),
            Patch(color="#34d399", label="std-projection"),
        ],
        facecolor="#1e1e2e", labelcolor="white", fontsize=9,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved -> {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. PyTorch CNN  (optional — only if torch is available)
# ---------------------------------------------------------------------------

def _try_torch_cnn(
    X: np.ndarray, y: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> None:
    """
    Train a lightweight CNN if PyTorch is importable.

    Input shape per sample: (3, 20, 20)
      channel 0 — max-projection
      channel 1 — mean-projection
      channel 2 — std-projection
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ModuleNotFoundError:
        print("PyTorch not found — skipping CNN.  Install with:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n")
        return

    print("\n" + "=" * 55)
    print("  PyTorch detected -- training BVP-CNN...")
    print("=" * 55)

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS     = 30
    BATCH_SIZE = 64
    LR         = 1e-3
    print(f"  Device : {DEVICE}\n")

    # Reshape flat 1200-dim vectors back to (3, 20, 20)
    X_tr  = X.reshape(-1, 3, 20, 20).astype(np.float32)
    X_te  = X_test.reshape(-1, 3, 20, 20).astype(np.float32)

    # Normalise per channel
    for c in range(3):
        mu = X_tr[:, c].mean(); sd = X_tr[:, c].std() + 1e-8
        X_tr[:, c] = (X_tr[:, c] - mu) / sd
        X_te[:, c] = (X_te[:, c] - mu) / sd

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y.astype(np.int64)))
    te_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_test.astype(np.int64)))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE)

    class BVP_CNN(nn.Module):
        """
        Small CNN:  (B,3,20,20) → conv×3 → GlobalAvgPool → Linear(5)
        ~47 k parameters.
        """
        def __init__(self, nc=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2),                          # → (B,64,10,10)
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1),                  # → (B,128,1,1)
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(128, nc),
            )
        def forward(self, x): return self.net(x)

    model = BVP_CNN().to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Train Acc':>9}")
    print("  " + "-" * 30)

    tr_losses, tr_accs = [], []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            ep_loss    += loss.item() * len(yb)
            ep_correct += (out.argmax(1) == yb).sum().item()
            ep_total   += len(yb)
        sched.step()
        tr_losses.append(ep_loss / ep_total)
        tr_accs.append(ep_correct / ep_total)
        print(f"  {ep:>5}  {tr_losses[-1]:>8.4f}  {tr_accs[-1]*100:>8.2f}%")

    elapsed = time.time() - t0
    print(f"\n  Training : {elapsed:.1f}s")

    # Test evaluation
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            all_p.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            all_l.extend(yb.numpy())
    cnn_acc = accuracy_score(all_l, all_p)
    print(f"  CNN Test accuracy : {cnn_acc*100:.2f}%\n")
    print(classification_report(all_l, all_p, target_names=CLASS_NAMES))

    # Training curve
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#12121f")
    _dark_ax(ax)
    epochs = range(1, EPOCHS + 1)
    ax.plot(epochs, [l for l in tr_losses], color="#818cf8", label="Train loss")
    ax2 = ax.twinx()
    ax2.plot(epochs, [a * 100 for a in tr_accs], color="#f472b6",
             linestyle="--", label="Train acc (%)")
    ax2.set_ylabel("Accuracy (%)", color="#f472b6", fontsize=10)
    ax2.tick_params(colors="#f472b6")
    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", color="#818cf8", fontsize=10)
    ax.set_title("BVP-CNN Training Curve", color="#c4b5fd", fontsize=12, pad=10)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, facecolor="#1e1e2e", labelcolor="white")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cnn_training_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved -> {out}")
    plt.close(fig)

    # CNN confusion matrix
    cm_cnn = confusion_matrix(all_l, all_p)
    plot_confusion_matrix(
        cm_cnn, CLASS_NAMES,
        "CNN Confusion Matrix -- Test Set",
        os.path.join(OUT_DIR, "confusion_matrix_cnn.png")
    )


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
