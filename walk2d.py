"""
walk2d.py
=========
Generates the 2D generalization of the paper's "random walk on a circle".

Paper recap (Shi & Cao, "Towards Understanding Transformers in Learning
Random Walks", NeurIPS 2025):
    - 1D walk on K nodes on a circle. At every step, move +1 (clockwise)
      w.p. p, or -1 (counter-clockwise) w.p. 1-p. Markov / memoryless.
    - A 1-layer transformer trained from zero-init provably learns the
      Bayes-optimal predictor when 0<p<1, by (a) attention collapsing onto
      the "direct parent" token (the immediately preceding state) and
      (b) the value matrix converging to the true transition matrix.
    - When p in {0,1} (deterministic walk) zero-init GD gets stuck forever,
      because the token AVERAGE is uninformative and can't break symmetry.

Our generalization:
    - State space = GxG grid (default 10x10 = 100 "nodes"), instead of a
      circle of K nodes.
    - Each frame is a black/white image: -1 everywhere except +1 at the
      walker's current position (a "one-hot" 2D state, exactly like a
      one-hot node index on the circle).
    - Dynamics are now LAZY: w.p. (1-p) the walker STAYS (this is the new
      "p" -- probability of moving at all, distinct from the paper's
      clockwise/counter-clockwise split). w.p. p it takes a step of random
      magnitude/direction inside a local neighborhood of radius k
      (Chebyshev/Moore neighborhood), i.e. it can move to any of the
      up to 8*k boundary cells of the k-radius box (edges are clipped,
      i.e. reflecting/clamped boundary, so the effective moves near a
      wall are a subset).
    - "Step size is also random": when curriculum stage requires it, the
      actual radius used for a move is drawn uniformly from {1,...,k}
      instead of being fixed at k. Early in training we keep the step size
      FIXED at k (matching the paper's clean "distance-1" moves), and only
      later randomize it -- this is the "keep the same step size in the
      beginning" curriculum you asked for.

This module is data-generation only; see model.py / train.py for the
transformer and analyze.py / experiments.py for the diagnostics.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class WalkConfig:
    grid: int = 10          # G, grid is GxG, N = G*G nodes (100 by default)
    p: float = 0.2          # probability of MOVING at a given step (1-p = stay/"latency")
    k: int = 1              # max step radius (local region size)
    directions: int = 4      # 4: only up/down/left/right (von Neumann). 8: also diagonals (Moore).
    step_mode: str = "fixed"  # "fixed": step radius always = k
                              # "random": step radius ~ Uniform{1,...,k}
    boundary: str = "clip"    # "clip": clamp to grid edges (reflecting-ish)
                              # "wrap": toroidal wraparound
    seed: int | None = None


_CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def _neighbors_at_radius(pos, radius, grid, boundary, rng, directions=4):
    """Sample a new cell exactly `radius` steps away from `pos`, respecting the
    grid boundary policy.

    directions=4 (default): pick one of the 4 cardinal directions
    (up/down/left/right) and move `radius` cells straight in that direction
    -- this is "the walker can walk in 4 directions" with a variable step
    size (k).
    directions=8: pick a uniformly random cell on the boundary ring of the
    radius-`radius` Chebyshev box (includes diagonals), kept for comparison
    with the earlier Moore-neighborhood version.
    """
    r0, c0 = pos
    if directions == 4:
        dr, dc = _CARDINAL[rng.integers(0, 4)]
        dr, dc = dr * radius, dc * radius
    else:
        cells = []
        for dr_ in range(-radius, radius + 1):
            for dc_ in range(-radius, radius + 1):
                if max(abs(dr_), abs(dc_)) != radius:
                    continue  # only the outer ring, i.e. "a step of exactly this size"
                cells.append((dr_, dc_))
        dr, dc = cells[rng.integers(0, len(cells))]
    r, c = r0 + dr, c0 + dc
    if boundary == "wrap":
        r %= grid
        c %= grid
    else:  # clip
        r = min(max(r, 0), grid - 1)
        c = min(max(c, 0), grid - 1)
    return (r, c)


def simulate_walk(T, cfg: WalkConfig, start=None):
    """Simulate one trajectory of length T (T frames).

    Returns dict with:
      positions:  (T,2) int array of (row,col) at each frame
      moved:      (T,) bool array, True if the walker moved at that step
                  (moved[0] is always False -- frame 0 is the initial state)
      step_size:  (T,) int array, the radius used when it moved (0 if stayed)
      frames:     (T, G, G) float32 array of -1/+1 images
    """
    rng = np.random.default_rng(cfg.seed)
    G = cfg.grid
    if start is None:
        pos = (int(rng.integers(0, G)), int(rng.integers(0, G)))
    else:
        pos = start

    positions = np.zeros((T, 2), dtype=np.int64)
    moved = np.zeros(T, dtype=bool)
    step_size = np.zeros(T, dtype=np.int64)
    positions[0] = pos

    for t in range(1, T):
        do_move = rng.random() < cfg.p
        if do_move:
            if cfg.step_mode == "fixed":
                radius = cfg.k
            else:  # "random": uniform over {1,...,k}
                radius = int(rng.integers(1, cfg.k + 1))
            pos = _neighbors_at_radius(pos, radius, G, cfg.boundary, rng, directions=cfg.directions)
            moved[t] = True
            step_size[t] = radius
        positions[t] = pos

    frames = -np.ones((T, G, G), dtype=np.float32)
    frames[np.arange(T), positions[:, 0], positions[:, 1]] = 1.0
    return {
        "positions": positions,
        "moved": moved,
        "step_size": step_size,
        "frames": frames,
        "cfg": cfg,
    }


def batch_simulate(n_seq, T, cfg: WalkConfig, base_seed=0):
    """Simulate a batch of n_seq independent trajectories (different seeds)."""
    frames = np.zeros((n_seq, T, cfg.grid, cfg.grid), dtype=np.float32)
    positions = np.zeros((n_seq, T, 2), dtype=np.int64)
    moved = np.zeros((n_seq, T), dtype=bool)
    step_size = np.zeros((n_seq, T), dtype=np.int64)
    for i in range(n_seq):
        cfg_i = WalkConfig(**{**cfg.__dict__, "seed": base_seed + i})
        out = simulate_walk(T, cfg_i)
        frames[i] = out["frames"]
        positions[i] = out["positions"]
        moved[i] = out["moved"]
        step_size[i] = out["step_size"]
    return {"frames": frames, "positions": positions, "moved": moved, "step_size": step_size}


def flat_index(positions, grid):
    """(...,2) row,col -> (...,) flattened index in [0, grid*grid)."""
    return positions[..., 0] * grid + positions[..., 1]


if __name__ == "__main__":
    cfg = WalkConfig(grid=10, p=0.2, k=1, step_mode="fixed", seed=0)
    out = simulate_walk(20, cfg)
    print("positions:\n", out["positions"])
    print("moved:", out["moved"])
    print("frame 1:\n", out["frames"][1])
