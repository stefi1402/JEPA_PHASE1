"""
dataset.py
----------
Simulates a single dot performing a lazy random walk on a d x d grid.

Dynamics (per the project spec):
    - Grid is d x d, rendered as a +1 / -1 image (+1 where the dot is).
    - At each timestep t:
        * with probability p, the dot moves; with probability 1-p it stays.
        * if it moves, the direction is picked uniformly at random from
          {up, down, left, right}.
        * the move has a step size k (number of cells). k can be fixed
          (same for the whole sequence) or randomized per-move.
    - The walk starts from a uniformly random position on the grid.
    - Movement is clipped at the grid boundary (hitting a wall = the dot
      stops at the edge for that step, it does not wrap around).

Important design note
----------------------
If you want a model to *learn to estimate p (or k)* from a sequence, or to
study how performance depends on p/k, then p (or k) must actually VARY
across sequences in the dataset. If every sequence uses the same fixed p,
there is nothing to estimate -- the network would just learn a constant.

To support both use cases this module can generate:
    * a "fixed" dataset: every sequence uses the same p and k
      (use this for Phase 1 -- pure trajectory prediction).
    * a "range" dataset: p (and/or k) is sampled per-sequence from a
      given range (use this for the p/k estimation and generalization
      studies).
"""

from __future__ import annotations

import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

# 4 directions: up, down, left, right
DIRS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])


@dataclass
class SequenceBatch:
    """Container for a generated dataset."""
    frames: np.ndarray       # (N, T, d, d) float32, values in {-1, +1}
    positions: np.ndarray    # (N, T, 2) int, (row, col) of the dot at each frame
    p_values: np.ndarray     # (N,) float32, the p used for each sequence
    k_values: np.ndarray     # (N,) float32, the (base) step size used for each sequence
    d: int = field(default=10)


def simulate_sequence(
    d: int,
    p: float,
    k: int,
    seq_len: int,
    step_size_random: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one sequence of length seq_len on a d x d grid.

    Returns
    -------
    frames : (seq_len, d, d) float32 array, +1 at the dot's cell, -1 elsewhere
    positions : (seq_len, 2) int array of (row, col) positions
    """
    pos = np.array([rng.integers(0, d), rng.integers(0, d)])
    positions = np.zeros((seq_len, 2), dtype=np.int64)
    frames = np.full((seq_len, d, d), -1.0, dtype=np.float32)

    for t in range(seq_len):
        positions[t] = pos
        frames[t, pos[0], pos[1]] = 1.0

        if rng.random() < p:
            direction = DIRS[rng.integers(0, 4)]
            step = rng.integers(1, k + 1) if step_size_random else k
            new_pos = pos + direction * step
            new_pos = np.clip(new_pos, 0, d - 1)
            pos = new_pos

    return frames, positions


def generate_dataset(
    n_sequences: int,
    d: int,
    seq_len: int,
    p: float = 0.2,
    k: int = 1,
    p_range: Optional[Tuple[float, float]] = None,
    k_range: Optional[Tuple[int, int]] = None,
    step_size_random: bool = False,
    seed: Optional[int] = None,
) -> SequenceBatch:
    """Generate a dataset of n_sequences random-walk sequences.

    If p_range is given, each sequence draws its own p ~ Uniform(p_range).
    Otherwise every sequence uses the fixed `p`. Same logic for k_range/k.

    Implementation note
    --------------------
    All N sequences are simulated together, vectorized across the batch
    dimension with numpy. Only the seq_len (time) axis remains a Python
    loop, since each step depends on the previous one; every other axis
    (sequences, and frame construction) is done with array ops. This
    turns an O(n_sequences * seq_len) pure-Python loop into an
    O(seq_len) loop of vectorized O(n_sequences) numpy operations, which
    is dramatically faster for the batch sizes used here (thousands of
    sequences).
    """
    rng = np.random.default_rng(seed)
    N = n_sequences

    # Per-sequence p / k, drawn once for the whole batch.
    if p_range is not None:
        all_p = rng.uniform(p_range[0], p_range[1], size=N).astype(np.float32)
    else:
        all_p = np.full(N, p, dtype=np.float32)

    if k_range is not None:
        all_k = rng.integers(k_range[0], k_range[1] + 1, size=N).astype(np.float32)
    else:
        all_k = np.full(N, k, dtype=np.float32)
    k_int = all_k.astype(np.int64)

    # Pre-draw all the per-step randomness for the whole batch at once.
    move_mask = rng.random((seq_len, N)) < all_p[None, :]          # (T, N)
    dir_idx = rng.integers(0, 4, size=(seq_len, N))                # (T, N)
    if step_size_random:
        # step ~ Uniform{1, ..., k_i} per sequence, per timestep.
        step = np.empty((seq_len, N), dtype=np.int64)
        low = np.ones(N, dtype=np.int64)
        high = k_int + 1
        for t in range(seq_len):
            step[t] = rng.integers(low, high)
    else:
        step = np.broadcast_to(k_int, (seq_len, N)).astype(np.int64)

    all_positions = np.empty((N, seq_len, 2), dtype=np.int64)
    all_frames = np.full((N, seq_len, d, d), -1.0, dtype=np.float32)
    seq_idx = np.arange(N)

    pos = np.stack([rng.integers(0, d, size=N), rng.integers(0, d, size=N)], axis=1)  # (N, 2)
    for t in range(seq_len):
        all_positions[:, t] = pos
        all_frames[seq_idx, t, pos[:, 0], pos[:, 1]] = 1.0

        delta = DIRS[dir_idx[t]] * step[t][:, None]                # (N, 2)
        moved = np.where(move_mask[t][:, None], pos + delta, pos)
        pos = np.clip(moved, 0, d - 1)

    return SequenceBatch(frames=all_frames, positions=all_positions,
                          p_values=all_p, k_values=all_k, d=d)


def save_dataset(batch: SequenceBatch, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        frames=batch.frames,
        positions=batch.positions,
        p_values=batch.p_values,
        k_values=batch.k_values,
        d=batch.d,
    )


def load_dataset(path: str) -> SequenceBatch:
    data = np.load(path)
    return SequenceBatch(
        frames=data["frames"],
        positions=data["positions"],
        p_values=data["p_values"],
        k_values=data["k_values"],
        d=int(data["d"]),
    )
