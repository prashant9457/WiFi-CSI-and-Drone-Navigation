"""
code/common/models.py
=====================
Defines deep learning and classical architectures for BVP data processing.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Dataset & Collate Function (For Variable Length BVP Sequences)
# ---------------------------------------------------------------------------

class BVPSequenceDataset(Dataset):
    """
    A PyTorch Dataset class that stores gesture trajectories as sequence matrices.
    
    Each item is a sequence tensor of shape (T, 400), representing a 20x20 velocity
    profile flattened over T timesteps.
    """
    def __init__(self, sequences: list, labels):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of NumPy arrays of shape (T_i, 400).
            labels: Array of ground-truth integer labels.
        """
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve the sequence and label at the specified index."""
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for PyTorch DataLoader to handle variable sequence lengths.
    
    Pads shorter sequences with zeros to make all samples in a batch uniform in shape,
    and returns their original unpadded lengths to mask padding states in recurrent layers.

    Args:
        batch: List of tuples (sequence, label) from BVPSequenceDataset.
        
    Returns:
        padded_seqs: Tensor of shape (batch_size, max_T, 400).
        lengths: 1D Tensor of original length values.
        labels: 1D Tensor of labels.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Pad sequences to (batch_size, max_T_in_batch, 400)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    
    return padded_seqs, lengths, labels


# ---------------------------------------------------------------------------
# 2. LSTM Classifier Architecture
# ---------------------------------------------------------------------------

class BVPLSTMClassifier(nn.Module):
    """
    An LSTM-based Recurrent Neural Network for processing flattened BVP frames.
    
    Accepts sequence batches, captures spatial-velocity changes over time using 
    recurrent weights, and uses the hidden representation from the last active
    timestep (determined by the sequence length) to predict gesture categories.
    """
    def __init__(self, input_size: int = 400, hidden_size: int = 128,
                 num_layers: int = 1, num_classes: int = 5):
        """
        Initialize the LSTM classifier network.

        Args:
            input_size: Number of features per frame (20x20 = 400).
            hidden_size: Hidden dimension size of LSTM cell.
            num_layers: Number of recurrent layers.
            num_classes: Classification output count.
        """
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
        """
        Forward pass of the LSTM classifier.

        Args:
            x: Input sequence batch of shape (batch_size, max_T, 400).
            lengths: Tensor containing original lengths of each sequence in the batch.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # Outputs shape: (batch_size, max_T_in_batch, hidden_size)
        outputs, _ = self.lstm(x)
        
        # We index into the outputs tensor at `lengths - 1` to extract the hidden state
        # at the last actual frame before padding zeros began.
        batch_size = x.size(0)
        final_states = outputs[torch.arange(batch_size), lengths - 1]
        
        # Pass final hidden state through classification head
        logits = self.fc(final_states)
        return logits


# ---------------------------------------------------------------------------
# 3. MLP Baseline Model Builder
# ---------------------------------------------------------------------------

def get_mlp_pipeline(seed: int = 42) -> Pipeline:
    """
    Construct a machine learning pipeline composed of a standard scaler and an MLP.
    
    Standardizes features to zero-mean and unit variance, then feeds the collapsed
    1200-dimensional BVP projections (max/mean/std) to a Multi-Layer Perceptron.

    Args:
        seed: Random state seed for reproducibility.

    Returns:
        StandardScaler + MLPClassifier Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )),
    ])
