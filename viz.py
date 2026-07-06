"""
viz.py
------
Visualization helpers used across every phase (generate / train / evaluate /
generalize) so you can sanity-check things as you go.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_sample_sequences(frames: np.ndarray, positions: np.ndarray,
                           n_sequences: int = 3, n_frames: int = 8,
                           save_path: str = "samples.png"):
    """Show a small grid of (sequence x frame) snapshots so you can eyeball
    whether the random walk looks right (movement frequency, step size)."""
    n_sequences = min(n_sequences, frames.shape[0])
    n_frames = min(n_frames, frames.shape[1])
    step = max(1, frames.shape[1] // n_frames)

    fig, axes = plt.subplots(n_sequences, n_frames, figsize=(n_frames * 1.3, n_sequences * 1.3))
    if n_sequences == 1:
        axes = axes[None, :]
    for s in range(n_sequences):
        for j in range(n_frames):
            t = j * step
            ax = axes[s, j]
            ax.imshow(frames[s, t], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if s == 0:
                ax.set_title(f"t={t}", fontsize=8)
        axes[s, 0].set_ylabel(f"seq {s}", fontsize=8)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_trajectory(positions: np.ndarray, d: int, save_path: str = "trajectory.png"):
    """Plot the (row, col) path of a single sequence over time."""
    fig, ax = plt.subplots(figsize=(4, 4))
    rows, cols = positions[:, 0], positions[:, 1]
    ax.plot(cols, rows, "-o", markersize=2, linewidth=0.8, alpha=0.7)
    ax.scatter([cols[0]], [rows[0]], color="green", s=60, label="start", zorder=5)
    ax.scatter([cols[-1]], [rows[-1]], color="red", s=60, label="end", zorder=5)
    ax.set_xlim(-0.5, d - 0.5); ax.set_ylim(d - 0.5, -0.5)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("dot trajectory")
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    """history: dict of metric_name -> list of values (per epoch/step)."""
    n = len(history)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_prediction_vs_truth(true_pos: np.ndarray, pred_pos: np.ndarray, d: int,
                              t_obs: int, save_path: str = "prediction.png"):
    """Overlay true vs predicted future trajectory on the grid.
    true_pos / pred_pos: (t_future, 2) arrays of (row, col).
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(true_pos[:, 1], true_pos[:, 0], "-o", color="tab:green",
            markersize=3, label="ground truth", alpha=0.8)
    ax.plot(pred_pos[:, 1], pred_pos[:, 0], "--x", color="tab:red",
            markersize=4, label="predicted", alpha=0.8)
    ax.set_xlim(-0.5, d - 0.5); ax.set_ylim(d - 0.5, -0.5)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"future prediction (observed {t_obs} frames)")
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_sweep(x_values, y_values, xlabel: str, ylabel: str,
                title: str = "", save_path: str = "sweep.png", y_err=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    if y_err is not None:
        ax.errorbar(x_values, y_values, yerr=y_err, marker="o", capsize=3)
    else:
        ax.plot(x_values, y_values, marker="o")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_p_estimation_scatter(true_p: np.ndarray, pred_p: np.ndarray,
                               save_path: str = "p_estimation.png"):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(true_p, pred_p, alpha=0.5, s=15)
    lo, hi = min(true_p.min(), pred_p.min()), max(true_p.max(), pred_p.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="ideal")
    ax.set_xlabel("true p"); ax.set_ylabel("estimated p")
    ax.legend(fontsize=8)
    ax.set_title("p estimation")
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
