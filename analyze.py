"""
analyze.py
==========
Diagnostics that mirror the paper's interpretability story:

  Paper (1-layer, circle, 0<p<1):
    - attention collapses onto the "direct parent" (previous) token
    - value matrix ~ true transition matrix Π
    - model reaches Bayes-optimal accuracy max(p,1-p)

  Here we ask the analogous questions on the 2D grid:
    1. p_hat: does the model's predicted "stay probability" match true p?
       (Bayes optimal for our lazy walk: P(stay)=1-p, and P(move to any of
        the 8k boundary cells) = p / (#cells at radius used).)
    2. k_hat: does the predicted distribution's support match the true
       neighborhood radius k? We measure the expected Chebyshev distance
       between predicted next-location and current location, weighted by
       predicted probability, and compare to the true expected step size.
    3. attention probe: how much attention mass (layer 0) does the
       frame-boundary token of frame t put on frame t-1 as a block
       (vs frame t-2, t-3, ... further back)? In the paper's Markov
       process, all the necessary information is in the immediately
       preceding state -- we check whether the transformer discovers the
       same "Markov" structure here (attend to the previous frame block
       and roughly ignore older ones).
    4. Out-of-distribution generalization in p: train at one p, evaluate
       (without retraining) at a different, unseen p.
"""
import numpy as np
import torch
import torch.nn.functional as F

from walk2d import WalkConfig, batch_simulate, flat_index
from train import make_batch, coord_loss


@torch.no_grad()
def estimate_p_and_k(model, grid, T, p_true, k_true, step_mode="fixed", directions=4,
                      n_seq=64, device="cpu"):
    frames, next_loc, moved, step_size = make_batch(n_seq, T, p_true, k_true, grid, step_mode,
                                                      directions=directions, base_seed=999)
    frames = frames.to(device)
    next_loc = next_loc.to(device)
    out = model(frames, need_coord=True, need_pixel=False)
    logits = out["coord_logits"][:, :-1, :]     # (B,T-1,N) predictions after frames 0..T-2
    probs = F.softmax(logits, dim=-1)

    cur_loc = next_loc[:, :-1]                   # (B,T-1) current (frame t) location
    B, Tm1, N = probs.shape
    rows = torch.arange(N, device=device) // grid
    cols = torch.arange(N, device=device) % grid

    cur_r = rows[cur_loc]  # (B,T-1)
    cur_c = cols[cur_loc]

    # P(stay) predicted := probability mass model puts on the SAME location
    same_idx = cur_loc  # index equal to current location = "stay"
    p_stay_hat = probs.gather(-1, same_idx.unsqueeze(-1)).squeeze(-1).mean().item()
    p_move_hat = 1 - p_stay_hat  # model's implied estimate of p

    # expected Chebyshev distance of predicted next location from current
    all_r = rows.view(1, 1, N).expand(B, Tm1, N)
    all_c = cols.view(1, 1, N).expand(B, Tm1, N)
    dr = (all_r - cur_r.unsqueeze(-1)).abs()
    dc = (all_c - cur_c.unsqueeze(-1)).abs()
    cheby = torch.maximum(dr, dc).float()
    exp_dist_hat = (probs * cheby).sum(-1).mean().item()

    # ground-truth expected step size, for reference
    # E[step] = p * E[radius | moved]; for step_mode=fixed, E[radius|moved]=k
    # for step_mode=random, E[radius|moved] = (k+1)/2
    if step_mode == "fixed":
        exp_dist_true = p_true * k_true
    else:
        exp_dist_true = p_true * (k_true + 1) / 2.0

    loss, acc = coord_loss(out, next_loc)
    return {
        "p_true": p_true, "p_move_hat": p_move_hat,
        "k_true": k_true, "exp_step_dist_true": exp_dist_true, "exp_step_dist_hat": exp_dist_hat,
        "coord_loss": loss.item(), "coord_acc": acc,
    }


@torch.no_grad()
def attention_probe(model, grid, T, p, k, step_mode="fixed", directions=4, n_seq=16, device="cpu"):
    """Returns, for the frame-boundary tokens, the average attention mass
    placed on each *previous frame block* (lag = 1, 2, 3, ... frames back),
    summed over the 100 pixel tokens of that block. lag=0 means "within the
    current frame" (attending to its own earlier pixels)."""
    frames, next_loc, moved, step_size = make_batch(n_seq, T, p, k, grid, step_mode, directions=directions, base_seed=1234)
    frames = frames.to(device)
    attn = model.get_attention(frames, layer=0)   # (B, L, L)
    N = grid * grid
    L = attn.shape[1]
    boundary_idx = torch.arange(N - 1, L, N)
    B = attn.shape[0]

    max_lag = min(6, L // N - 1)
    lag_mass = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)
    for bi, qpos in enumerate(boundary_idx.tolist()):
        frame_of_query = qpos // N
        for lag in range(0, max_lag + 1):
            f = frame_of_query - lag
            if f < 0:
                continue
            lo, hi = f * N, f * N + N
            hi = min(hi, qpos + 1)  # causal: can't see beyond query
            if lo >= hi:
                continue
            mass = attn[:, qpos, lo:hi].sum(-1).mean().item()
            lag_mass[lag] += mass
            counts[lag] += 1
    lag_mass = lag_mass / np.maximum(counts, 1)
    return lag_mass  # lag_mass[0] = mass on current frame (incl. self), lag_mass[1] = prev frame, ...


@torch.no_grad()
def ood_generalization(model, grid, T, p_train, p_test_list, k, step_mode="fixed", directions=4,
                        n_seq=64, device="cpu"):
    results = []
    for p_test in p_test_list:
        frames, next_loc, moved, step_size = make_batch(n_seq, T, p_test, k, grid, step_mode,
                                                          directions=directions, base_seed=555)
        frames = frames.to(device)
        next_loc = next_loc.to(device)
        out = model(frames, need_coord=True, need_pixel=False)
        loss, acc = coord_loss(out, next_loc)
        results.append({"p_train": p_train, "p_test": p_test,
                         "coord_loss": loss.item(), "coord_acc": acc})
    return results
