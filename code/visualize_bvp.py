"""
visualize_bvp.py
================
Widar 3.0 — BVP Visualization

Workflow intuition
------------------
A BVP (Body-coordinate Velocity Profile) encodes *where* in velocity space
a human body is moving at each instant.  Think of it as a 2-D radar snapshot:
the X-axis is radial velocity (toward / away from the receiver array) and the
Y-axis is tangential velocity (left / right sweep).  Each gesture traces a
unique trajectory through this 20×20 velocity grid over time.

This script:
  1. find_sample  – locates one .mat file for a chosen gesture / repetition
  2. load_bvp     – reads the tensor  (20 × 20 × T)  from that file
  3. plot_projections  – collapses the time axis three ways so we can read the
                         gesture's "fingerprint" in a single static image:
                           · middle frame   → what the gesture looks like mid-way
                           · max-projection → every velocity cell that was ever
                                              active (the gesture's full footprint)
                           · mean-projection→ the average energy distribution
  4. animate_bvp  – plays back every frame as a GIF so we can watch the
                    velocity blob move through the grid in real time

Usage (from repo root):
    python code/visualize_bvp.py
"""

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join("code", "data", "BVP")
OUT_DIR  = "img"

# Widar 3.0 gesture index → human-readable name
GESTURE_NAMES = {
    1: "Push & Pull",
    2: "Sweep",
    3: "Clap",
    4: "Slide",
    5: "Draw Circle (CW)",
    6: "Draw Circle (CCW)",
    7: "Draw Triangle",
    8: "Draw Zigzag",
    9: "Draw N",
    10: "Random",
}

# The 20×20 grid covers velocities from -2 m/s to +2 m/s on both axes
VEL_EXTENT = [-2, 2, -2, 2]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_sample(root: str, gesture_id: int, repetition: int) -> str:
    """
    Walk the BVP directory tree and return the path of the first .mat file
    that matches the requested gesture index and repetition number.

    File names follow the convention:
        user{U}-{gesture}-{location}-{orientation}-{repetition}-...mat
    so gesture is at index 1 and repetition at index 4 after splitting on '-'.
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


def load_bvp(path: str) -> np.ndarray:
    """
    Load the velocity spectrum tensor from a .mat file.

    The variable 'velocity_spectrum_ro' is always shaped (20, 20, T) where T
    is the number of 100ms time frames captured during the gesture.  A handful
    of files in the dataset were saved with T squeezed out (shape 20×20); we
    restore that axis so the rest of the pipeline never has to special-case it.
    """
    mat  = sio.loadmat(path)
    data = mat["velocity_spectrum_ro"]

    if data.ndim == 2:
        data = data[:, :, np.newaxis]   # restore missing time axis

    if data.shape[2] == 0:
        raise ValueError(f"Empty tensor (T=0): {path}")

    return data


# ---------------------------------------------------------------------------
# Visualization helpers  (shared dark-theme style)
# ---------------------------------------------------------------------------

def _apply_dark_style(ax, cbar):
    """Apply a consistent dark-background style to an axes + colorbar pair."""
    ax.set_facecolor("#12121f")
    ax.set_xlabel("Vx  (m/s)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Vy  (m/s)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    cbar.ax.tick_params(colors="#94a3b8", labelsize=7)
    cbar.outline.set_edgecolor("#334155")


# ---------------------------------------------------------------------------
# Plot 1 — Static 3-panel projection figure
# ---------------------------------------------------------------------------

def plot_projections(bvp: np.ndarray, title: str, out_path: str) -> None:
    """
    Render three complementary views of the gesture's velocity footprint.

    Each view collapses the time axis differently:
      · Middle frame    – a snapshot at the midpoint of the gesture
      · Max-projection  – pixel-wise maximum over all frames; highlights every
                          velocity region the body passed through
      · Mean-projection – pixel-wise average; shows where energy was
                          concentrated most consistently
    """
    T = bvp.shape[2]

    panels = {
        "Middle Frame":            bvp[:, :, T // 2],
        "Max Projection  (time)":  np.max(bvp,  axis=2),
        "Mean Projection (time)":  np.mean(bvp, axis=2),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#12121f")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.02)

    for ax, (label, data) in zip(axes, panels.items()):
        im   = ax.imshow(data, cmap="magma", extent=VEL_EXTENT,
                         origin="lower", aspect="equal")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, color="#c4b5fd", fontsize=11, pad=10)
        _apply_dark_style(ax, cbar)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#12121f")
    print(f"  Saved  ->  {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Temporal animation
# ---------------------------------------------------------------------------

def animate_bvp(bvp: np.ndarray, gesture_name: str,
                out_path: str, fps: int = 10) -> None:
    """
    Animate the BVP frame-by-frame and save as a looping GIF.

    The colour scale is fixed to the 99th percentile of the entire tensor so
    no single bright frame dominates and the motion pattern stays readable
    throughout.  At 10 fps each frame represents ~100ms of real movement.
    """
    T    = bvp.shape[2]
    vmax = float(np.percentile(bvp, 99))

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#12121f")
    im    = ax.imshow(bvp[:, :, 0], cmap="magma", extent=VEL_EXTENT,
                      origin="lower", aspect="equal", vmin=0, vmax=vmax)
    cbar  = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    title = ax.set_title("", color="white", fontsize=11, fontweight="bold", pad=10)
    _apply_dark_style(ax, cbar)

    def _update(t: int):
        im.set_array(bvp[:, :, t])
        title.set_text(
            f"{gesture_name}  |  frame {t + 1:02d}/{T}  ({t / fps:.1f} s)"
        )
        return im, title

    ani = animation.FuncAnimation(
        fig, _update, frames=T, interval=1000 // fps, blit=True
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, writer="pillow", fps=fps)
    print(f"  Saved  ->  {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    GESTURE_ID = 1   # 1–10  (see GESTURE_NAMES above)
    REPETITION = 1   # 1–20

    print(f"\nGesture : {GESTURE_NAMES[GESTURE_ID]}  (id={GESTURE_ID}, rep={REPETITION})")

    path = find_sample(DATA_DIR, gesture_id=GESTURE_ID, repetition=REPETITION)
    print(f"File    : {path}")

    bvp  = load_bvp(path)
    name = GESTURE_NAMES[GESTURE_ID]
    print(f"Shape   : {bvp.shape}  ->  20x20 velocity grid, T={bvp.shape[2]} frames\n")

    print("[1/2] Static projections ...")
    plot_projections(
        bvp,
        title    = f"BVP Projections  –  {name}",
        out_path = os.path.join(OUT_DIR, "bvp_projections.png"),
    )

    print("[2/2] Temporal animation ...")
    animate_bvp(
        bvp,
        gesture_name = name,
        out_path     = os.path.join(OUT_DIR, "bvp_animation.gif"),
        fps          = 10,
    )

    print("\nDone.")
