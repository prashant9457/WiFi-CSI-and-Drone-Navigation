"""
bvp_loader.py
=============
Dataset loading, parsing, and caching logic for Widar 3.0 BVP (Body-coordinate Velocity Profile).
"""

import os
import numpy as np
import scipy.io as sio

# Configurations
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

LOCATION_NAMES = [f"Loc {i}" for i in range(1, 9)]      # 8 grid positions
ORIENTATION_NAMES = ["South", "SE", "East", "NE", "North"] # 5 directions

def extract_features(bvp: np.ndarray) -> np.ndarray:
    """
    Collapse the (20, 20, T) BVP tensor into a flat feature vector.

    Three projections are computed along the time axis:
      · max-projection  — the gesture's full velocity footprint
      · mean-projection — consistent energy distribution
      · std-projection  — temporal variability / motion dynamics

    Result: (3 × 20 × 20,) = 1 200-dimensional vector.
    """
    max_p  = np.max(bvp,  axis=2)   # (20, 20)
    mean_p = np.mean(bvp, axis=2)   # (20, 20)
    std_p  = np.std(bvp,  axis=2)   # (20, 20)
    return np.concatenate([max_p.ravel(), mean_p.ravel(), std_p.ravel()])

def load_single_bvp(path: str) -> np.ndarray:
    """
    Load the velocity spectrum tensor from a single .mat file.
    Restores the time axis if squeezed.
    """
    mat  = sio.loadmat(path)
    data = mat["velocity_spectrum_ro"].astype(np.float32)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]   # restore missing time axis
    if data.shape[2] == 0:
        raise ValueError(f"Empty tensor (T=0): {path}")
    return data

def find_sample(root: str, gesture_id: int, repetition: int) -> str:
    """
    Walk the BVP directory tree and return the path of the first .mat file
    that matches the requested gesture index and repetition number.
    """
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.endswith(".mat"):
                continue
            parts = fname.split("-")
            try:
                g = int(parts[1])
                r = int(parts[4])
            except (IndexError, ValueError):
                continue
            if g == gesture_id and r == repetition:
                return os.path.join(dirpath, fname)

    raise FileNotFoundError(
        f"No sample found for gesture={gesture_id}, rep={repetition} under {root}"
    )

def load_dataset(
    root: str,
    gesture_ids: set[int] | None = None,
    cache_filename: str = "classify_location_cache.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk the BVP tree for the specified gesture IDs and return four aligned arrays.
    Uses caching to speed up future runs.

    Returns
    -------
    X   : (N, 1200) float32  -- BVP feature vectors
    y_g : (N,)     int64    -- gesture class index (0..len(gesture_ids)-1)
    y_l : (N,)     int64    -- location class index (0..7)
    y_o : (N,)     int64    -- orientation class index (0..4)
    """
    if gesture_ids is None:
        gesture_ids = ALL9_GESTURE_IDS

    cache_path = os.path.join(root, cache_filename)
    if os.path.exists(cache_path):
        print(f"Loading dataset from cache: {cache_path}")
        try:
            cache = np.load(cache_path)
            X = cache["X"]
            y_g = cache["y_g"]
            y_l = cache["y_l"]
            y_o = cache["y_o"]
            print(f"  [ok] {len(y_g)} samples loaded from cache  |  feature dim = {X.shape[1]}")
            print(f"       Gestures     : {np.unique(y_g)}")
            print(f"       Locations    : {np.unique(y_l) + 1}")
            print(f"       Orientations : {np.unique(y_o) + 1}")
            print()
            return X, y_g, y_l, y_o
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-scanning raw files...")

    gesture_label_map = {gid: idx for idx, gid in enumerate(sorted(gesture_ids))}

    X_list, yg_list, yl_list, yo_list = [], [], [], []
    skipped = 0

    print("Scanning BVP dataset tree...")
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.endswith(".mat"):
                continue
            parts = fname.split("-")
            try:
                g   = int(parts[1])
                loc = int(parts[2])
                ori = int(parts[3])
            except (IndexError, ValueError):
                continue
            if g not in gesture_ids:
                continue

            path = os.path.join(dirpath, fname)
            try:
                data = load_single_bvp(path)
            except Exception:
                skipped += 1
                continue

            feat = extract_features(data)
            X_list.append(feat)
            yg_list.append(gesture_label_map[g])
            yl_list.append(loc - 1)    # 1-indexed -> 0-indexed
            yo_list.append(ori - 1)    # 1-indexed -> 0-indexed

    if skipped:
        print(f"  [!] Skipped {skipped} corrupt / empty files")

    X   = np.vstack(X_list)
    y_g = np.array(yg_list, dtype=np.int64)
    y_l = np.array(yl_list, dtype=np.int64)
    y_o = np.array(yo_list, dtype=np.int64)

    try:
        np.savez_compressed(cache_path, X=X, y_g=y_g, y_l=y_l, y_o=y_o)
        print(f"Saved dataset cache to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    print(f"  [ok] {len(y_g)} samples loaded  |  feature dim = {X.shape[1]}")
    print(f"       Gestures     : {np.unique(y_g)}")
    print(f"       Locations    : {np.unique(y_l) + 1}")
    print(f"       Orientations : {np.unique(y_o) + 1}")
    print()
    return X, y_g, y_l, y_o
