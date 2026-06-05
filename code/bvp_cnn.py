"""
bvp_cnn.py
==========
Optional CNN-based gesture classification using PyTorch.
Can be run directly:
    python code/bvp_cnn.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from bvp_loader import load_dataset as load_bvp_dataset
from bvp_plotting import _dark_ax, plot_confusion_matrix

from common.constants import (
    DATA_DIR,
    OUT_DIR,
    SEED,
    TOP5,
    TOP5_CLASS_NAMES as CLASS_NAMES,
)

def run_cnn(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Train and evaluate the PyTorch CNN classifier."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ModuleNotFoundError:
        print("\nPyTorch not found -- skipping CNN. Install with:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n")
        return

    print("\n" + "=" * 55)
    print("  PyTorch BVP-CNN Classifier")
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
        def __init__(self, nc=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1),
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

if __name__ == "__main__":
    X, y, _, _ = load_bvp_dataset(
        DATA_DIR,
        gesture_ids=set(TOP5.keys()),
        cache_filename="classify_bvp_cache.npz",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    run_cnn(X_train, y_train, X_test, y_test)
