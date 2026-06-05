"""
classify_bvp_lstm.py
====================
Widar 3.0 — Gesture Classification using a Temporal LSTM Network.

This script tests the scientific hypothesis:
"Gesture information primarily exists in the temporal evolution of the BVP sequence
and is lost when the time dimension is collapsed."

Why LSTMs?
----------
Long Short-Term Memory (LSTM) networks are a type of Recurrent Neural Network (RNN)
specifically designed to process sequential data. Unlike standard feedforward networks,
LSTMs maintain an internal cell state that acts as a memory buffer. This allows them to
capture long-term dependencies and patterns over time. Since human gestures (like Sweep
vs. Push & Pull) consist of specific velocity profiles occurring in a distinct temporal
sequence, LSTMs are ideally suited to model this time-varying trajectory.

Why Projections Lose Information:
---------------------------------
Collapsing a 3D BVP tensor (20x20xT) into static 2D max, mean, and std projections
discards the order in which velocities occurred. For example, if you move your hand
left then right, or right then left, their static "max projections" look identical,
but their temporal trajectories are opposite. The LSTM processes the sequence frame
by frame, preserving this temporal ordering.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom modules
from bvp_loader import load_sequence_dataset
from bvp_plotting import plot_confusion_matrix, _dark_ax

# Common package imports
from common.constants import (
    DATA_DIR,
    OUT_DIR,
    SEED,
    TOP5,
    TOP5_CLASS_NAMES as CLASS_NAMES,
)
from common.models import (
    BVPSequenceDataset,
    collate_fn,
    BVPLSTMClassifier,
)
from common.training import train_lstm

# Configuration
MODEL_PATH = "lstm_best.pth"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5  # Early stopping patience
MLP_ACCURACY = 45.75  # Stored baseline MLP result for comparison

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n" + "=" * 55)
    print("  Widar 3.0 — LSTM Sequence Classifier")
    print("=" * 55 + "\n")

    # ── Load Sequence Data ────────────────────────────────────────────────
    sequences, labels = load_sequence_dataset(DATA_DIR)

    # ── Stratified Splits ─────────────────────────────────────────────────
    # Split indices to ensure alignment of variable-length lists
    indices = np.arange(len(sequences))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=SEED
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15 / (1.0 - 0.15), stratify=labels[train_idx], random_state=SEED
    )

    train_seqs = [sequences[i] for i in train_idx]
    train_lbls = labels[train_idx]

    val_seqs = [sequences[i] for i in val_idx]
    val_lbls = labels[val_idx]

    test_seqs = [sequences[i] for i in test_idx]
    test_lbls = labels[test_idx]

    print(f"Data Splits: train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)}\n")

    # Create PyTorch DataLoaders
    train_dataset = BVPSequenceDataset(train_seqs, train_lbls)
    val_dataset = BVPSequenceDataset(val_seqs, val_lbls)
    test_dataset = BVPSequenceDataset(test_seqs, test_lbls)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BVPLSTMClassifier(num_classes=len(TOP5)).to(device)

    print(f"Using Device: {device}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Training with Early Stopping ──────────────────────────────────────
    t0 = time.time()
    tr_losses, val_losses, val_accs = train_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE,
        checkpoint_path=MODEL_PATH,
        verbose=True,
    )
    # Convert validation accuracies from percentages (returned by train_lstm) to fractions for plotting
    val_accs = [acc / 100.0 for acc in val_accs]
    print(f"Training completed in {time.time() - t0:.1f}s\n")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#12121f")
    _dark_ax(ax)
    epochs_range = range(1, len(tr_losses) + 1)
    ax.plot(epochs_range, tr_losses, color="#818cf8", label="Train Loss")
    ax.plot(epochs_range, val_losses, color="#34d399", label="Val Loss")
    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", color="#94a3b8", fontsize=10)
    
    ax2 = ax.twinx()
    ax2.plot(epochs_range, [a * 100 for a in val_accs], color="#f472b6", linestyle="--", label="Val Acc (%)")
    ax2.set_ylabel("Accuracy (%)", color="#f472b6", fontsize=10)
    ax2.tick_params(colors="#f472b6")
    
    ax.set_title("BVP-LSTM Training Curve", color="#c4b5fd", fontsize=12, pad=10)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, facecolor="#1e1e2e", labelcolor="white")
    
    plt.tight_layout()
    curve_out = os.path.join(OUT_DIR, "lstm_training_curve.png")
    plt.savefig(curve_out, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved -> {curve_out}")
    plt.close(fig)

    # ── Evaluation ────────────────────────────────────────────────────────
    print("Evaluating best model checkpoint on test set...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_preds = []
    test_targets = []
    with torch.no_grad():
        for xb, lengths, yb in test_loader:
            xb, lengths = xb.to(device), lengths.to(device)
            logits = model(xb, lengths)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_targets.extend(yb.numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_acc_percentage = test_acc * 100

    print(f"\nLSTM Test Accuracy: {test_acc_percentage:.2f}%\n")
    print(classification_report(test_targets, test_preds, target_names=CLASS_NAMES))

    # Save Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        "LSTM Confusion Matrix -- Test Set",
        os.path.join(OUT_DIR, "confusion_matrix_lstm.png")
    )

    # ── Comparison Section ────────────────────────────────────────────────
    diff = test_acc_percentage - MLP_ACCURACY

    print("\n" + "=" * 55)
    print("  Model Comparison")
    print("=" * 55)
    print(f"MLP Accuracy  : {MLP_ACCURACY:.2f}%")
    print(f"LSTM Accuracy : {test_acc_percentage:.2f}%")
    print(f"Difference    : {diff:+.2f}%")
    print("-" * 55)

    print("Interpretation:")
    if diff > 0:
        print(
            "The LSTM outperforms the MLP, confirming our scientific hypothesis:\n"
            "Preserving the temporal ordering of BVP features is essential for\n"
            "accurately recognizing and separating physical human gesture dynamics."
        )
    else:
        print(
            "The LSTM did not outperform the MLP. Possible causes include:\n"
            "1. Insufficient tuning (e.g., learning rate, hidden sizes, optimization steps).\n"
            "2. Limited model capacity (1 layer of 128 hidden units may be insufficient).\n"
            "3. Location leakage dominates: the spatial distribution of velocities remains\n"
            "   a strong feature, and the temporal model struggles to overcome this shortcut."
        )
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
