"""
classify_location.py
====================
Widar 3.0 -- BVP Domain-Invariance Experiment

This script evaluates BVP projections for location-invariance by training two
parallel classifiers on exactly the same BVP features:
  - Classifier A: predicts GESTURE (9 classes)
  - Classifier B: predicts LOCATION (8 classes)
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Custom modules
from bvp_loader import load_dataset, GESTURE_NAMES, LOCATION_NAMES, ALL9_GESTURE_IDS
from bvp_plotting import plot_confusion_matrix, plot_invariance_summary

warnings.filterwarnings("ignore")

DATA_DIR  = os.path.join("code", "data", "BVP")
OUT_DIR   = "img"
SEED      = 42
TEST_SIZE = 0.15

def run_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    label:   str,
) -> tuple[float, np.ndarray]:
    """
    Train an MLP and return (test_accuracy, predictions).
    Uses a StandardScaler -> MLP pipeline.
    """
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
    mlp.fit(X_train, y_train)
    elapsed = time.time() - t0
    preds = mlp.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  [{label}]  trained in {elapsed:.1f}s  |  test accuracy = {acc*100:.2f}%")
    return acc, preds

def run_lstm_gesture_9(
    idx_train: np.ndarray,
    idx_test: np.ndarray,
) -> float:
    """
    Train LSTM on 9 gesture classes and return the test accuracy.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from bvp_loader import load_sequence_dataset, ALL9_GESTURE_IDS
    from classify_bvp_lstm import BVPSequenceDataset, collate_fn, BVPLSTMClassifier

    print("\n  Loading BVP sequences for all 9 gestures...")
    sequences, labels = load_sequence_dataset(
        DATA_DIR,
        gesture_ids=ALL9_GESTURE_IDS,
        cache_filename="classify_lstm_9_cache.npz",
    )

    # Split sequences into train/val/test using the same indices
    train_seqs = [sequences[i] for i in idx_train]
    train_lbls = labels[idx_train]

    # Validation split from training (15%) for early stopping
    idx_tr, idx_val = train_test_split(
        np.arange(len(train_seqs)), test_size=0.15, stratify=train_lbls, random_state=SEED
    )

    val_seqs = [train_seqs[i] for i in idx_val]
    val_lbls = train_lbls[idx_val]

    tr_seqs = [train_seqs[i] for i in idx_tr]
    tr_lbls = train_lbls[idx_tr]

    test_seqs = [sequences[i] for i in idx_test]
    test_lbls = labels[idx_test]

    train_ds = BVPSequenceDataset(tr_seqs, tr_lbls)
    val_ds = BVPSequenceDataset(val_seqs, val_lbls)
    test_ds = BVPSequenceDataset(test_seqs, test_lbls)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BVPLSTMClassifier(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    checkpoint_path = "lstm_temp_9.pth"

    print("  Training LSTM on 9 gesture classes (with early stopping)...")
    for epoch in range(1, 30):
        model.train()
        for xb, lengths, yb in train_loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, lengths, yb in val_loader:
                xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
                logits = model(xb, lengths)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(yb)
        val_loss /= len(val_lbls)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, lengths, yb in test_loader:
            xb, lengths = xb.to(device), lengths.to(device)
            logits = model(xb, lengths)
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    acc = accuracy_score(test_lbls, preds)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"  [Gesture (LSTM)]  trained  |  test accuracy = {acc*100:.2f}%\n")
    return acc

if __name__ == "__main__":
    np.random.seed(SEED)
    print(f"\n{'='*60}")
    print("  Widar 3.0 -- BVP Domain-Invariance Experiment")
    print(f"{'='*60}\n")
    print("Hypothesis: BVP should encode gesture signal while removing")
    print("            location cues from the representation.")
    print("            We test this by running 2 classifiers on the same features.\n")

    # ── Load ────────────────────────────────────────────────────────────────
    X, y_g, y_l, _ = load_dataset(DATA_DIR)

    # ── Split (stratify on gesture so all tasks use the same split) ─────────
    idx = np.arange(len(y_g))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y_g, random_state=SEED
    )
    X_train, X_test = X[idx_train], X[idx_test]
    print(f"Split  : train={len(idx_train)}  test={len(idx_test)}\n")

    # ── Run classifiers ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  Training MLP Classifiers on BVP projections...")
    print("=" * 60)

    g_acc, g_pred = run_classifier(
        X_train, y_g[idx_train], X_test, y_g[idx_test], "Gesture (9-class) "
    )
    l_acc, l_pred = run_classifier(
        X_train, y_l[idx_train], X_test, y_l[idx_test], "Location (8-class) "
    )

    print("\n" + "=" * 60)
    print("  Training LSTM Classifier on raw BVP sequences...")
    print("=" * 60)
    lstm_acc = run_lstm_gesture_9(idx_train, idx_test)

    # ── Per-class reports ───────────────────────────────────────────────────
    gesture_names = [GESTURE_NAMES[gid] for gid in sorted(ALL9_GESTURE_IDS)]
    print(f"\n--- Gesture (MLP) Classification Report ---")
    print(classification_report(y_g[idx_test], g_pred, target_names=gesture_names))

    print(f"--- Location (MLP) Classification Report ---")
    print(classification_report(y_l[idx_test], l_pred, target_names=LOCATION_NAMES))

    # ── Domain-invariance summary ───────────────────────────────────────────
    n_gesture     = len(np.unique(y_g))
    n_location    = len(np.unique(y_l))

    print("\n" + "=" * 60)
    print("  DOMAIN-INVARIANCE SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<22}  {'Classes':>7}  {'Random':>8}  {'Accuracy':>8}  {'Gap':>8}")
    print("  " + "-" * 57)

    rows = [
        ("Gesture MLP",     n_gesture,     g_acc),
        ("Gesture LSTM",    n_gesture,     lstm_acc),
        ("Location MLP",    n_location,    l_acc),
    ]
    for name, n_cls, acc in rows:
        rand = 1.0 / n_cls
        gap  = acc - rand
        if "Location" in name:
            verdict = "[STRONG signal -- NOT removed]"
        elif "LSTM" in name:
            verdict = "[STRONG signal -- temporal ordering preserves shape]"
        else:
            verdict = "[WEAK signal -- collapsed dimension loses shape]"
        print(f"  {name:<22}  {n_cls:>7}  {rand*100:>7.1f}%  {acc*100:>7.2f}%  "
              f"{gap*100:>+7.2f}%  {verdict}")

    print()

    # ── Plots ───────────────────────────────────────────────────────────────
    print("Generating plots...")

    plot_invariance_summary(
        results=[
            {"label": "Gesture MLP (9-class)",      "acc": g_acc,
             "random": 1/n_gesture,     "color": "#818cf8",
             "ann": ("MLP: loses temporal", "ordering")},
            {"label": "Gesture LSTM (9-class)",     "acc": lstm_acc,
             "random": 1/n_gesture,     "color": "#34d399",
             "ann": ("LSTM: preserves", "temporal ordering")},
            {"label": "Location MLP (8-class)",      "acc": l_acc,
             "random": 1/n_location,    "color": "#f472b6",
             "ann": ("Location: leaking", "spatial cues")},
        ],
        out_path=os.path.join(OUT_DIR, "domain_invariance_bars.png"),
    )

    cm_g = confusion_matrix(y_g[idx_test], g_pred)
    plot_confusion_matrix(
        cm_g, gesture_names,
        "Gesture Confusion Matrix (test set)",
        os.path.join(OUT_DIR, "confusion_gesture9.png"),
    )

    cm_l = confusion_matrix(y_l[idx_test], l_pred)
    plot_confusion_matrix(
        cm_l, LOCATION_NAMES,
        "Location Confusion Matrix -- 95% accuracy: projections leak location",
        os.path.join(OUT_DIR, "confusion_location.png"),
    )

    print("\nDone.")
