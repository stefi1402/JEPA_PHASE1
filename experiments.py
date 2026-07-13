"""
experiments.py
==============
Runs the sweeps you asked for:

  - vary p in {0.1, 0.2, ...}
  - vary k (local region / step radius)
  - vary context length t (train at t=50, evaluate/generalize to t=100, etc.)
  - joint (p,k,t) correlation grid
  - p-generalization: train at one p, test at unseen p's

NOTE ON SCALE: the full spec (grid=10x10, T in {50,100}, several (p,k)
combos) means sequence lengths L = T*100 in {5000, 10000}. With plain
O(L^2) causal self-attention this is expensive on CPU (~1-6s per
training step at T=20-50 on a single core in this sandbox). The functions
below are correct and scale-ready; `demo_sweep()` runs a cheap, reduced
version (small grid / short T / few steps) so you can see the whole
pipeline execute end-to-end quickly. For the real experiment, bump
`grid`, `T`, and `steps` back up (see the __main__ block) and run on a
GPU or let it run for longer -- everything else is unchanged.
"""
import json
import os
import gc
import numpy as np
import torch

from train import train
from analyze import estimate_p_and_k, attention_probe, ood_generalization
from walk2d import WalkConfig, simulate_walk, batch_simulate
from rollout import rollout_predict, positions_to_rc


def sweep_p_k_t(p_list, k_list, t_list, grid=10, steps=200, d_model=64,
                 n_layers=1, batch_size=16, step_mode_final="random",
                 seed=0, verbose=False):
    """Joint sweep: for every (p,k,t) combo, train a fresh model and record
    final coord loss/acc, p_hat, and expected-step-size estimate. This is
    what lets you correlate performance/estimation quality against t, k, p
    jointly."""
    rows = []
    for p in p_list:
        for k in k_list:
            for T in t_list:
                model, hist = train(task="coord", p=p, k=k, grid=grid, T=T,
                                     steps=steps, d_model=d_model, n_layers=n_layers,
                                     batch_size=batch_size, curriculum_frac=0.3,
                                     step_mode_final=step_mode_final, seed=seed,
                                     verbose=verbose, log_every=max(1, steps // 4))
                diag = estimate_p_and_k(model, grid, T, p, k, step_mode=step_mode_final)
                diag.update({"T": T, "final_train_loss": hist[-1]["coord_loss"]})
                rows.append(diag)
                print(f"p={p} k={k} T={T} -> coord_acc={diag['coord_acc']:.3f} "
                      f"p_hat={diag['p_move_hat']:.3f} "
                      f"step_dist(true={diag['exp_step_dist_true']:.2f}, "
                      f"hat={diag['exp_step_dist_hat']:.2f})")
    return rows


def estimate_attention_memory_gib(T, grid, batch_size, n_heads=4, n_layers=1, dtype_bytes=4):
    """Rough estimate of the peak memory (GiB) needed just for the causal
    attention score matrix (the O(L^2) part that caused the earlier OOM):
        L = T * grid^2
        memory ~= batch_size * n_heads * L^2 * dtype_bytes  (per layer;
        we use a factor of ~1.5 to loosely account for a couple of
        layers' worth of activations + gradients living at once, but
        this is a rough guide, not an exact figure)."""
    L = T * grid * grid
    bytes_ = batch_size * n_heads * (L ** 2) * dtype_bytes * max(1, n_layers) * 1.5
    return bytes_ / (1024 ** 3)


def rollout_eval(model, grid, p, k, T, directions=4, step_mode_final="fixed",
                  device="cpu", seed=777):
    """Runs the SAME autoregressive rollout check as `main.py rollout`:
    simulate a fresh trajectory, split it into a context half and a
    future half (sized to fit within the model's trained max_frames=T),
    let the model predict the future half on its own (feeding its own
    guesses back in), and report the exact-position match rate against
    the true continuation. This is the piece that was missing from the
    sweep -- training + one-step-ahead diagnostics alone don't tell you
    whether the model is actually useful for multi-step prediction."""
    context = max(1, T // 2)
    future = T - context
    if future <= 0:
        return {"rollout_context": context, "rollout_future": 0, "rollout_match_rate": None}

    wcfg = WalkConfig(grid=grid, p=p, k=k, directions=directions,
                       step_mode="fixed" if step_mode_final == "fixed" else "random",
                       seed=seed)
    truth = simulate_walk(context + future, wcfg)
    context_frames = truth["frames"][:context]
    context_flat = torch.tensor(context_frames).reshape(1, context * grid * grid)

    pred_idx, _ = rollout_predict(model, grid, context_flat, future, device=device)
    pred_rc = positions_to_rc(pred_idx, grid)
    true_rc = truth["positions"][context:context + future]
    match_rate = float((pred_rc == true_rc).all(axis=-1).mean())
    return {"rollout_context": context, "rollout_future": future, "rollout_match_rate": match_rate}


def generate_for_combo(g, p, k, T, n_preview=8, directions=4, step_mode_final="fixed",
                        seed=0, save_frames=False, frames_dir="sweep_frames"):
    """The explicit GENERATE step of the pipeline: simulate `n_preview`
    trajectories for this exact (grid, p, k, T) combo, print a summary
    so generation is visible in the log, and optionally save them to
    disk (--save-frames) so you can inspect the actual frames used for
    this combo.

    Note: this generated batch is NOT what training samples from --
    train() below generates a FRESH random batch every single step (a
    deliberate choice: the walk is cheap to simulate and this is
    essentially free infinite data, so re-simulating avoids overfitting
    to one fixed dataset). This step exists to make generation visible
    and inspectable, not to replace the on-the-fly generation used
    during training."""
    cfg = WalkConfig(grid=g, p=p, k=k, directions=directions,
                      step_mode="fixed" if step_mode_final == "fixed" else "random",
                      boundary="clip", seed=seed)
    data = batch_simulate(n_preview, T, cfg, base_seed=seed * 777 + 1)
    frac_moved = float(data["moved"].mean())
    print(f"[GENERATE] grid={g} p={p} k={k} T={T}: simulated {n_preview} trajectories "
          f"({n_preview}x{T} frames) -- observed move rate {frac_moved:.3f} (target p={p})")

    save_path = None
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
        save_path = os.path.join(frames_dir, f"grid{g}_p{p}_k{k}_T{T}.npz")
        np.savez(save_path, frames=data["frames"], positions=data["positions"],
                 moved=data["moved"], step_size=data["step_size"], p=p, k=k, grid=g)
        print(f"           saved generated frames to {save_path}")
    return {"observed_move_rate": frac_moved, "frames_saved_to": save_path}


def sweep_d_k_t(d_model_list, k_list, t_list, grid_list=None, p=0.3, grid=8, steps=800,
                 n_layers=1, batch_size=8, n_heads=4, step_mode_final="fixed",
                 device="cpu", max_mem_gib=4.0, seed=0, out_json="sweep_dkt_results.json",
                 save_frames=False, frames_dir="sweep_frames", verbose=False):
    """Sweep over (d_model, k, T, grid) at a FIXED p -- 'see results for
    different combinations of d, k, T, grid'. Skips (and warns about) any
    combo whose attention matrix would need more than `max_mem_gib` of
    memory, so it stays safe to run on a laptop.

    `grid_list`: list of board sizes to try (e.g. [6, 8, 10]). If None,
    uses the single fixed `grid` value for every combo (old behavior).

    Runs the FULL pipeline EXPLICITLY, in this order -- but note that
    GENERATE only depends on (grid, p, k, T), not on d_model, so it is
    done ONCE per (grid, k, T) and reused for every d_model in
    d_model_list that shares it, instead of wastefully re-simulating
    identical walk data for every d_model value:
      1. generate     (generate_for_combo): simulate + print/optionally save
                       a preview batch of trajectories for this (grid,k,T)
                       -- runs once per (grid,k,T), cached across d_model
      2. train         (train.py): trains a FRESH model for every
                       (d_model,k,T,grid) combo (generates its own fresh
                       random batches every step internally) -- this part
                       genuinely must rerun per combo, since each one is
                       an independently trained model
      3. one-step eval (analyze.py): coord_loss/coord_acc, p_move_hat,
                       exp_step_dist_hat, attention-by-lag
      4. rollout       (rollout.py): autoregressive multi-step prediction
                       on a held-out trajectory, reported as rollout_match_rate
    """
    if grid_list is None:
        grid_list = [grid]
    rows = []
    for g in grid_list:
        for k in k_list:
            for T in t_list:
                # step 1: GENERATE -- once per (grid,k,T), reused for every d_model below
                gen_info = generate_for_combo(g, p, k, T, directions=4,
                                                step_mode_final=step_mode_final, seed=seed,
                                                save_frames=save_frames, frames_dir=frames_dir)
                for d_model in d_model_list:
                    mem = estimate_attention_memory_gib(T, g, batch_size, n_heads, n_layers)
                    if mem > max_mem_gib:
                        print(f"[SKIP] grid={g} d_model={d_model} k={k} T={T}: estimated "
                              f"{mem:.2f} GiB > max_mem_gib={max_mem_gib} -- would likely OOM. "
                              f"Try smaller T/grid/batch_size, or raise --max-mem-gib if you "
                              f"know your machine can handle it.")
                        rows.append({"grid": g, "d_model": d_model, "k": k, "T": T,
                                     "skipped": True, "estimated_gib": mem})
                        continue

                    print(f"\n=== grid={g} d_model={d_model} k={k} T={T} (est. {mem:.2f} GiB) ===")
                    # step 2: TRAIN (generates its own fresh batches every step internally)
                    model, hist = train(task="coord", p=p, k=k, grid=g, T=T,
                                         steps=steps, d_model=d_model, n_layers=n_layers,
                                         n_heads=n_heads, batch_size=batch_size,
                                         curriculum_frac=0.3, step_mode_final=step_mode_final,
                                         seed=seed, verbose=verbose, device=device,
                                         log_every=max(1, steps // 4))
                    # step 3: ONE-STEP-AHEAD EVAL
                    diag = estimate_p_and_k(model, g, T, p, k, step_mode=step_mode_final,
                                             device=device)
                    lag_mass = attention_probe(model, g, T, p, k, step_mode=step_mode_final,
                                                device=device)
                    # step 4: MULTI-STEP ROLLOUT
                    roll = rollout_eval(model, g, p, k, T, step_mode_final=step_mode_final,
                                         device=device, seed=seed * 10000 + 777)

                    diag.update({"grid": g, "d_model": d_model, "k": k, "T": T,
                                 "final_train_loss": hist[-1]["coord_loss"],
                                 "attn_lag_mass": lag_mass.tolist(), "skipped": False,
                                 "estimated_gib": mem})
                    diag.update(gen_info)
                    diag.update(roll)
                    rows.append(diag)
                    print(f"  -> coord_acc={diag['coord_acc']:.3f} coord_loss={diag['coord_loss']:.3f} "
                          f"p_hat={diag['p_move_hat']:.3f} step_dist_hat={diag['exp_step_dist_hat']:.3f} "
                          f"rollout_match={roll['rollout_match_rate']}")

                    # Free this model's memory before starting the next combo. Without
                    # this, GPU memory (MPS/CUDA) accumulates across the whole sweep --
                    # each individual config may be tiny, but after enough sequential
                    # runs it adds up and eventually OOMs even though no single config
                    # was ever close to the budget.
                    del model
                    gc.collect()
                    if device == "mps" and hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                    elif device.startswith("cuda"):
                        torch.cuda.empty_cache()

    json.dump(rows, open(out_json, "w"), indent=2)
    print(f"\nSaved {len(rows)} results ({sum(not r.get('skipped') for r in rows)} run, "
          f"{sum(r.get('skipped', False) for r in rows)} skipped) to {out_json}")
    return rows


def p_generalization_study(p_train_list, p_test_list, grid=10, k=1, T=20,
                            steps=200, d_model=64, n_layers=1, batch_size=16, seed=0):
    """Trains separately at each p in p_train_list, then evaluates each
    resulting model (no retraining) on every p in p_test_list. Shows
    whether a model trained at a specific p generalizes to unseen p, i.e.
    whether it has learned the *general rule* "attend to previous frame,
    output P(stay)=1-p / P(each neighbor)=p/(#neighbors)" as a function of
    p, versus just memorizing the training p."""
    results = []
    for p_train in p_train_list:
        model, hist = train(task="coord", p=p_train, k=k, grid=grid, T=T, steps=steps,
                             d_model=d_model, n_layers=n_layers, batch_size=batch_size,
                             seed=seed, verbose=False, log_every=steps)
        res = ood_generalization(model, grid, T, p_train, p_test_list, k)
        results.extend(res)
        for r in res:
            print(f"train p={r['p_train']:.2f} -> test p={r['p_test']:.2f}: "
                  f"acc={r['coord_acc']:.3f} loss={r['coord_loss']:.3f}")
    return results


def attention_lag_study(p=0.2, k=1, grid=10, T=20, steps=200, d_model=64,
                         n_layers=1, batch_size=16, seed=0):
    """Trains one model and inspects layer-0 attention mass by lag (in
    frames) from the frame-boundary query token -- the 2D analogue of the
    paper's finding that attention collapses onto the immediately
    preceding ('parent') token. Here the parent is the *entire previous
    frame block* rather than a single token, since the state is a one-hot
    image rather than a single symbol."""
    model, hist = train(task="coord", p=p, k=k, grid=grid, T=T, steps=steps,
                         d_model=d_model, n_layers=n_layers, batch_size=batch_size,
                         seed=seed, verbose=False, log_every=steps)
    lag_mass = attention_probe(model, grid, T, p, k)
    print("attention mass by lag (0=current frame, 1=prev frame, ...):", lag_mass)
    return model, hist, lag_mass


def demo_sweep():
    """Cheap, fast, end-to-end demonstration (small grid/T/steps)."""
    rows = sweep_p_k_t(p_list=[0.2, 0.4], k_list=[1, 2], t_list=[8],
                        grid=5, steps=150, d_model=48, batch_size=16)
    json.dump(rows, open("demo_sweep_results.json", "w"), indent=2)
    return rows


if __name__ == "__main__":
    demo_sweep()

    # ---- to run the FULL-scale experiment from the spec, use e.g.: ----
    # rows = sweep_p_k_t(p_list=[0.1, 0.2], k_list=[1, 2, 3], t_list=[50, 100],
    #                     grid=10, steps=2000, d_model=128, n_layers=1)
    # gen = p_generalization_study(p_train_list=[0.1, 0.2], p_test_list=[0.05, 0.3, 0.5],
    #                                grid=10, T=50, steps=2000)
    # model, hist, lag_mass = attention_lag_study(p=0.2, k=1, grid=10, T=50, steps=2000)
