"""
multi_walkers.py
=================
Extension: "put multiple Ps, see what happens."

We place M independent walkers on the same GxG grid simultaneously, each
with its OWN probability p_m of moving (and, optionally, its own k_m).
Every walker occupies a +1 pixel (background -1), so a frame can contain
up to M lit pixels (fewer if two walkers land on the same cell).

Two variants are supported:

  variant="labeled":
    We keep the M walkers' identities and train M separate coordinate
    heads (one per walker), each predicting that specific walker's next
    location. This is the "easy" version -- the model is TOLD which lit
    pixel belongs to which walker (via M separate targets), so it can in
    principle learn a separate p_m/k_m per walker (i.e. per-head
    transition rule), similar to running M independent copies of the
    1-walker task.

  variant="unlabeled":
    The frame only shows an unordered SET of +1 pixels -- there is no way
    to tell, from the image alone, which lit pixel came from which
    walker. We only ask the model to predict the NEXT SET of lit
    locations (as a multi-hot target / set-prediction loss), never "which
    walker is which." This is the interesting case: if two walkers have
    different p_m (e.g. one nearly always moves, one almost always
    stays), the walker identity is recoverable in principle from the
    trajectory (a moving dot vs a nearly-static dot), so the model *can*
    learn to route each dot through its own transition rule by tracking
    "what happened to the pixel that was here" rather than by any
    external label. But this is much harder for the model to discover,
    and is the closest 2D analogue to the paper's discussion of
    "unbreakable symmetry" style failures: if the walkers are made
    perfectly identical (same p, same k, indistinguishable trajectories),
    the task becomes genuinely UNIDENTIFIABLE (predicting an unordered
    set of positions rather than which-pixel-is-which), just as the
    paper's zero-init deterministic walk was undiagnosable by the
    "uninformative token average" -- here it's the walkers' own symmetry,
    not the initialization, that removes the signal needed to
    disambiguate.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from walk2d import WalkConfig, simulate_walk, flat_index
from model import CausalTransformer


def simulate_multi_walk(T, grid, p_list, k_list, boundary="clip", seed=0):
    """Simulate M independent walkers on the same grid. Returns:
      frames:     (T, grid, grid) with multiple +1 pixels (one per walker,
                  overwritten if they collide)
      positions:  (M, T, 2)  per-walker positions
    """
    M = len(p_list)
    positions = np.zeros((M, T), dtype=np.int64)
    all_pos = np.zeros((M, T, 2), dtype=np.int64)
    frames = -np.ones((T, grid, grid), dtype=np.float32)
    for m in range(M):
        cfg = WalkConfig(grid=grid, p=p_list[m], k=k_list[m], step_mode="fixed",
                          boundary=boundary, seed=seed * 1000 + m)
        out = simulate_walk(T, cfg)
        all_pos[m] = out["positions"]
    for t in range(T):
        for m in range(M):
            r, c = all_pos[m, t]
            frames[t, r, c] = 1.0
    return frames, all_pos


def batch_multi(n_seq, T, grid, p_list, k_list, base_seed=0):
    M = len(p_list)
    frames = np.zeros((n_seq, T, grid, grid), dtype=np.float32)
    all_pos = np.zeros((n_seq, M, T, 2), dtype=np.int64)
    for i in range(n_seq):
        f, pos = simulate_multi_walk(T, grid, p_list, k_list, seed=base_seed + i)
        frames[i] = f
        all_pos[i] = pos
    return frames, all_pos


class MultiWalkerModel(CausalTransformer):
    """Adds M per-walker coordinate heads on top of the base transformer,
    for the 'labeled' variant."""
    def __init__(self, n_walkers, **kwargs):
        super().__init__(**kwargs)
        self.n_walkers = n_walkers
        self.walker_heads = nn.ModuleList([nn.Linear(self.d_model, self.N)
                                            for _ in range(n_walkers)])

    def forward_multi(self, pixel_vals):
        x, T = self.embed(pixel_vals)
        L = x.shape[1]
        device = x.device
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        h = self.encoder(x, mask=causal_mask, is_causal=True)
        h = self.ln_f(h)
        N = self.N
        boundary_idx = torch.arange(N - 1, L, N, device=device)
        h_bd = h[:, boundary_idx, :]   # (B, T, d)
        logits_per_walker = [head(h_bd) for head in self.walker_heads]  # M x (B,T,N)
        return logits_per_walker


def train_labeled(p_list, k_list, grid=10, T=20, steps=200, d_model=64,
                   n_layers=1, batch_size=16, lr=3e-4, seed=0, log_every=50, device="cpu"):
    """variant='labeled': model gets M separate heads and M separate
    (walker-specific) targets. Tests whether per-walker p_m, k_m can each
    be learned when identity is given for free."""
    torch.manual_seed(seed)
    M = len(p_list)
    model = MultiWalkerModel(n_walkers=M, grid=grid, max_frames=T, d_model=d_model,
                              n_heads=4, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(steps):
        frames, all_pos = batch_multi(batch_size, T, grid, p_list, k_list,
                                       base_seed=seed * 100000 + step * batch_size)
        frames_flat = torch.tensor(frames).reshape(batch_size, T * grid * grid).to(device)
        targets = [torch.tensor(flat_index(all_pos[:, m], grid)).to(device) for m in range(M)]  # M x (B,T)

        logits_per_walker = model.forward_multi(frames_flat)
        loss = 0.0
        accs = []
        for m in range(M):
            pred = logits_per_walker[m][:, :-1, :]
            tgt = targets[m][:, 1:]
            l = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1))
            loss = loss + l
            accs.append((pred.argmax(-1) == tgt).float().mean().item())
        opt.zero_grad(); loss.backward(); opt.step()
        history.append({"step": step, "loss": float(loss.detach()), "accs": accs})
        if step % log_every == 0 or step == steps - 1:
            print(f"[multi-labeled] step {step:4d} loss={float(loss.detach()):.4f} "
                  f"per-walker acc={['%.3f' % a for a in accs]}")
    return model, history


def train_unlabeled(p_list, k_list, grid=10, T=20, steps=200, d_model=64,
                     n_layers=1, batch_size=16, lr=3e-4, seed=0, log_every=50, device="cpu"):
    """variant='unlabeled': single coord_head trained with a MULTI-LABEL
    (set) loss over the N grid cells: target is a multi-hot vector marking
    every cell occupied by ANY walker in the next frame. No walker
    identity is ever given. If p_list/k_list are all equal across walkers,
    walkers are exchangeable and the *set* of next locations may still be
    predictable in aggregate, but WHICH lit pixel maps to which walker's
    history is fundamentally unidentifiable from the image stream alone --
    directly analogous to the paper's "uninformative symmetry" failure,
    except the symmetry here lives in the walkers rather than the init."""
    torch.manual_seed(seed)
    model = CausalTransformer(grid=grid, max_frames=T, d_model=d_model, n_heads=4,
                               n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    M = len(p_list)
    history = []
    for step in range(steps):
        frames, all_pos = batch_multi(batch_size, T, grid, p_list, k_list,
                                       base_seed=seed * 100000 + step * batch_size)
        frames_flat = torch.tensor(frames).reshape(batch_size, T * grid * grid).to(device)
        # build multi-hot target (B,T,N): 1 at every cell occupied by some walker
        N = grid * grid
        idx = flat_index(all_pos, grid)  # (B,M,T)
        multihot = torch.zeros(batch_size, T, N, device=device)
        for b in range(batch_size):
            for m in range(M):
                multihot[b, torch.arange(T), idx[b, m]] = 1.0

        out = model(frames_flat, need_coord=True, need_pixel=False)
        logits = out["coord_logits"][:, :-1, :]     # (B,T-1,N)
        target = multihot[:, 1:, :]
        loss = F.binary_cross_entropy_with_logits(logits, target)
        pred_set = (torch.sigmoid(logits) > 0.5).float()
        # "set accuracy": fraction of frames where predicted lit set == true lit set
        set_match = (pred_set == target).all(-1).float().mean().item()

        opt.zero_grad(); loss.backward(); opt.step()
        history.append({"step": step, "loss": float(loss.detach()), "set_acc": set_match})
        if step % log_every == 0 or step == steps - 1:
            print(f"[multi-unlabeled] step {step:4d} loss={float(loss.detach()):.4f} set_acc={set_match:.3f}")
    return model, history


if __name__ == "__main__":
    print("=== labeled variant: 2 walkers, different p ===")
    train_labeled(p_list=[0.1, 0.4], k_list=[1, 1], grid=5, T=8, steps=100, d_model=32, batch_size=16)

    print("\n=== unlabeled variant: 2 walkers, different p (identity NOT given) ===")
    train_unlabeled(p_list=[0.1, 0.4], k_list=[1, 1], grid=5, T=8, steps=100, d_model=32, batch_size=16)
