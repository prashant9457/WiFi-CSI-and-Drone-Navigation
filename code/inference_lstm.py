"""
inference_lstm.py
===================
Run inference on a single BVP (.mat) file using the trained LSTM model.

Usage:
------
python code/inference_lstm.py --file code/data/BVP/20181109-VS/user1-1-1-1-1.mat
"""

import os
import argparse
import numpy as np
import scipy.io as sio
import torch

from bvp_loader import load_single_bvp
from common.models import BVPLSTMClassifier
from common.constants import TOP5_CLASS_NAMES

# Mapping of class index (0..4) to Gesture Names for Top-5
TOP5_NAMES = {idx: name for idx, name in enumerate(TOP5_CLASS_NAMES)}

def predict_single_file(file_path: str, model_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find input BVP file: {file_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model weights file: {model_path}. Please train the model first.")

    # 1. Load raw BVP tensor of shape (20, 20, T)
    print(f"Loading BVP file: {file_path}...")
    bvp = load_single_bvp(file_path)
    T = bvp.shape[2]
    
    # 2. Preprocess BVP sequence: transpose to (T, 20, 20) and flatten to (T, 400)
    bvp_t = np.transpose(bvp, (2, 0, 1))
    sequence = bvp_t.reshape(T, 400)
    
    # 3. Convert to batch format for PyTorch: shape (1, T, 400)
    seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    length = torch.tensor([T], dtype=torch.long)

    # 4. Load the BVPLSTMClassifier (using Top-5 output classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BVPLSTMClassifier(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 5. Run forward pass
    seq_tensor, length = seq_tensor.to(device), length.to(device)
    with torch.no_grad():
        logits = model(seq_tensor, length)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = logits.argmax(dim=1).item()

    # 6. Print predictions
    print("\n" + "=" * 45)
    print("  LSTM Inference Results")
    print("=" * 45)
    print(f"File         : {os.path.basename(file_path)}")
    print(f"Sequence len : {T} frames")
    print(f"Predicted    : {TOP5_NAMES[predicted_idx]} ({probabilities[predicted_idx]*100:.2f}% confidence)")
    print("-" * 45)
    print("Probabilities:")
    for idx, name in TOP5_NAMES.items():
        mark = "-->" if idx == predicted_idx else "   "
        print(f" {mark} {name:<18} : {probabilities[idx]*100:>6.2f}%")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the trained LSTM model on a single BVP .mat file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the BVP .mat file.")
    parser.add_argument("--model", type=str, default="lstm_best.pth", help="Path to the trained model checkpoint (.pth).")
    
    args = parser.parse_args()
    predict_single_file(args.file, args.model)
