"""
code/common/constants.py
========================
Centralized constants for the Widar 3.0 BVP gesture classification and domain-invariance experiments.
"""

import os

# General settings
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Gesture IDs and Names
ALL9_GESTURE_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9}

GESTURE_NAMES = {
    1: "Push & Pull",
    2: "Sweep",
    3: "Clap",
    4: "Slide",
    5: "Circle (CW)",
    6: "Circle (CCW)",
    7: "Triangle",
    8: "Zigzag",
    9: "Draw N",
    10: "Random",
}

# Grid locations and receiver orientations
LOCATION_NAMES = [f"Loc {i}" for i in range(1, 9)]      # 8 positions
ORIENTATION_NAMES = ["South", "SE", "East", "NE", "North"] # 5 orientations

# Top-5 Gestures (by highest sample count)
TOP5 = {
    1: "Push & Pull",
    2: "Sweep",
    3: "Clap",
    4: "Slide",
    5: "Draw Circle (CW)",
}

TOP5_CLASS_NAMES = [TOP5[gid] for gid in sorted(TOP5)]
TOP5_LABEL_MAP = {gid: idx for idx, gid in enumerate(sorted(TOP5))}

# Directory configuration
DATA_DIR = os.path.join("code", "data", "BVP")
OUT_DIR = "img"
