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
from bvp_plotting import plot_confusion_matrix

# Configuration
DATA_DIR = os.path.join("code", "data", "BVP")
OUT_DIR = "img"
MODEL_PATH = "lstm_best.pth"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5  # Early stopping patience

TOP5 = {
    1: "Push & Pull",
    2: "Sweep",
    3: "Clap",
    4: "Slide",
    5: "Draw Circle (CW)",
}
CLASS_NAMES = [TOP5[gid] for gid in sorted(TOP5)]
MLP_ACCURACY = 45.75  # Stored baseline MLP result for comparison

# ---------------------------------------------------------------------------
# 1. Custom Dataset & Collate Function (Handling Variable Length Sequences)
# ---------------------------------------------------------------------------

class BVPSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset that loads variable-length BVP sequences.
    Each sample is a float32 tensor of shape (T, 400).
    """
    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray):
        # Convert list of NumPy arrays to PyTorch tensors
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Why Padding is Required:
    ------------------------
    PyTorch DataLoader batches tensors together into a single multi-dimensional tensor.
    However, neural networks require all samples in a batch to have the same dimensions.
    Since gestures have variable time lengths (T), we use `pad_sequence` to fill shorter
    sequences with zeros (padding) up to the maximum sequence length in the current batch.
    
    We also track the actual lengths of the sequences before padding so that the LSTM
    can extract the correct final state instead of reading the padded zeros.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Pad sequences to (batch_size, max_T_in_batch, 400)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    
    return padded_seqs, lengths, labels

# ---------------------------------------------------------------------------
# 2. LSTM Neural Network Architecture
# ---------------------------------------------------------------------------

class BVPLSTMClassifier(nn.Module):
    """
    Compact LSTM Classifier for 400-dimensional BVP time series.
    """
    def __init__(self, input_size: int = 400, hidden_size: int = 128,
                 num_layers: int = 1, num_classes: int = 5):
        super().__init__()
        # Compact LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, max_T_in_batch, 400)
        # outputs shape: (batch_size, max_T_in_batch, hidden_size)
        outputs, (hn, cn) = self.lstm(x)
        
        # Why the Final Hidden State is Used:
        # -----------------------------------
        # The outputs tensor contains the hidden states of the LSTM at *every* timestep.
        # For sequence-level classification, we only care about the representation of
        # the sequence *after* the entire gesture has been seen.
        # We index into the outputs tensor at `lengths - 1` to extract the hidden state
        # at the last actual frame before padding zeros began.
        batch_size = x.size(0)
        final_states = outputs[torch.arange(batch_size), lengths - 1]
        
        # Pass final hidden state through classification head
        logits = self.fc(final_states)
        return logits

# ---------------------------------------------------------------------------
# 3. Main Training & Evaluation Loop
# ---------------------------------------------------------------------------

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

    # Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BVPLSTMClassifier(num_classes=len(TOP5)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Using Device: {device}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Training with Early Stopping ──────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"  {'Epoch':>5}  {'Tr Loss':>8}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("  " + "-" * 38)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_total = 0
        for xb, lengths, yb in train_loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(yb)
            train_total += len(yb)
            
        epoch_train_loss = train_loss / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, lengths, yb in val_loader:
                xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
                logits = model(xb, lengths)
                loss = criterion(logits, yb)
                
                val_loss += loss.item() * len(yb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += len(yb)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"  {epoch:>5d}  {epoch_train_loss:>8.4f}  {epoch_val_loss:>8.4f}  {epoch_val_acc*100:>6.2f}%")

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)  # Save best checkpoint
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    print(f"Training completed in {time.time() - t0:.1f}s\n")

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
