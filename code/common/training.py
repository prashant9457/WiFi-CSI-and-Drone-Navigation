"""
code/common/training.py
=======================
Reusable training utilities and loops for BVP sequence neural network models.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    checkpoint_path: str = "lstm_checkpoint.pth",
    verbose: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """
    Train a PyTorch LSTM model on BVP sequence batches with automatic early stopping.
    
    This training utility runs backpropagation on the training set, measures cross-entropy
    loss and classification accuracy on the validation set, and stops training early when
    the validation loss stops decreasing for `patience` consecutive epochs.

    Args:
        model: PyTorch BVPLSTMClassifier model instance.
        train_loader: DataLoader wrapping the training split.
        val_loader: DataLoader wrapping the validation split.
        device: CPU or CUDA torch device context.
        epochs: Maximum number of epochs to train.
        lr: Learning rate parameter for the Adam optimizer.
        patience: Epoch tolerance count for validation loss degradation (early stopping).
        checkpoint_path: File system path where the best weights will be stored.
        verbose: If True, prints epoch stats to stdout.

    Returns:
        tr_losses: List of average training loss per epoch.
        val_losses: List of average validation loss per epoch.
        val_accs: List of average validation accuracy (%) per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    tr_losses, val_losses, val_accs = [], [], []

    if verbose:
        print(f"  {'Epoch':>5}  {'Tr Loss':>8}  {'Val Loss':>8}  {'Val Acc':>7}")
        print("  " + "-" * 38)

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
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
        tr_losses.append(epoch_train_loss)

        # 2. Validation Phase
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
        epoch_val_acc = (val_correct / val_total) * 100
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        if verbose:
            print(f"  {epoch:>5}  {epoch_train_loss:>8.4f}  {epoch_val_loss:>8.4f}  {epoch_val_acc:>6.1f}%")

        # 3. Checkpoint & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\n  [Early Stopping] Triggered after {epoch} epochs.")
                break

    return tr_losses, val_losses, val_accs
