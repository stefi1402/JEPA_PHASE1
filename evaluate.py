"""
evaluate.py
-----------
Loads a trained model and a (possibly new) dataset, and reports:
    - exact-match accuracy of (row, col) prediction, per future timestep
    - p / k estimation error (if the model was trained with predict_pk)
plus visualizations.

Also supports SWEEP evaluation: given a list of p values (and/or k values),
generates a fresh test set for each value and reports how accuracy /
p-estimation / k-estimation change -- this is the "does it generalize with
p" / "how does p impact performance" study.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from dataset import generate_dataset
from model import DotTransformer
from train import WalkDataset
from torch.utils.data import DataLoader
import viz


def load_model(model_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    model = DotTransformer(
        d=cfg["d"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], max_frames=cfg["t_obs"] + cfg["t_future"] + 5,
        predict_pk=cfg["predict_pk"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, device


@torch.inference_mode()
def evaluate_on_batch(model, cfg, device, batch, batch_size=4, num_workers=2):
    """Returns a dict of aggregate metrics for one SequenceBatch."""
    ds = WalkDataset(batch, cfg["t_obs"], cfg["t_future"])
    use_cuda = str(device).startswith("cuda")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # Accumulate on-device to avoid a host/device sync on every batch;
    # convert to Python scalars once, after the loop.
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_count = 0
    per_step_correct = torch.zeros(cfg["t_future"], device=device)
    p_abs_err = torch.zeros((), device=device)
    k_abs_err = torch.zeros((), device=device)
    n_pk = 0
    n_seq = 0

    for frames_obs, pos_future, p_true, k_true in loader:
        frames_obs = frames_obs.to(device, non_blocking=True)
        pos_future = pos_future.to(device, non_blocking=True)
        p_true = p_true.to(device, non_blocking=True)
        k_true = k_true.to(device, non_blocking=True)

        out = model(frames_obs, cfg["t_future"])
        row_pred = out["row_logits"].argmax(-1)
        col_pred = out["col_logits"].argmax(-1)
        exact = (row_pred == pos_future[..., 0]) & (col_pred == pos_future[..., 1])

        total_correct += exact.sum()
        total_count += exact.numel()
        per_step_correct += exact.float().sum(dim=0)
        n_seq += frames_obs.size(0)

        if cfg["predict_pk"]:
            p_abs_err += (out["p_pred"] - p_true).abs().sum()
            k_abs_err += (out["k_pred"] - k_true).abs().sum()
            n_pk += p_true.numel()

    metrics = {
        "exact_match_acc": total_correct.item() / total_count,
        "per_step_acc": per_step_correct.cpu().numpy() / n_seq,
    }
    if cfg["predict_pk"]:
        metrics["p_mae"] = p_abs_err.item() / n_pk
        metrics["k_mae"] = k_abs_err.item() / n_pk
    return metrics


def run_single_evaluation(model_path: str, n_sequences: int, p: float, k: int,
                            out_dir: str = "eval_out", seed: int = 123, batch_size: int = 4,
                            num_workers: int = 2):
    os.makedirs(out_dir, exist_ok=True)
    model, cfg, device = load_model(model_path)
    seq_len = cfg["t_obs"] + cfg["t_future"]
    batch = generate_dataset(n_sequences=n_sequences, d=cfg["d"], seq_len=seq_len,
                               p=p, k=k, seed=seed)
    metrics = evaluate_on_batch(model, cfg, device, batch, batch_size=batch_size, num_workers=num_workers)

    print(f"[eval] p={p} k={k} | exact_match_acc={metrics['exact_match_acc']:.3f}", end="")
    if "p_mae" in metrics:
        print(f" | p_MAE={metrics['p_mae']:.4f} | k_MAE={metrics['k_mae']:.4f}")
    else:
        print()

    viz.plot_sweep(
        list(range(cfg["t_future"])), metrics["per_step_acc"],
        xlabel="future timestep", ylabel="exact-match accuracy",
        title=f"per-timestep accuracy (p={p}, k={k})",
        save_path=os.path.join(out_dir, f"viz_per_step_acc_p{p}_k{k}.png"),
    )
    return metrics


def run_generalization_sweep(model_path: str, p_values, k_values, n_sequences: int = 200,
                                out_dir: str = "generalize_out", seed: int = 123, batch_size: int = 4,
                                num_workers: int = 2):
    """Evaluates the model across a grid of (p, k) values it may or may not
    have seen during training, to study generalization in p (and k)."""
    os.makedirs(out_dir, exist_ok=True)
    model, cfg, device = load_model(model_path)
    seq_len = cfg["t_obs"] + cfg["t_future"]

    results = {"p": [], "k": [], "exact_match_acc": [], "p_mae": [], "k_mae": []}

    for k in k_values:
        accs, p_maes, k_maes = [], [], []
        for p in p_values:
            batch = generate_dataset(n_sequences=n_sequences, d=cfg["d"], seq_len=seq_len,
                                       p=p, k=k, seed=seed)
            metrics = evaluate_on_batch(model, cfg, device, batch, batch_size=batch_size, num_workers=num_workers)
            accs.append(metrics["exact_match_acc"])
            p_maes.append(metrics.get("p_mae", np.nan))
            k_maes.append(metrics.get("k_mae", np.nan))
            results["p"].append(p); results["k"].append(k)
            results["exact_match_acc"].append(metrics["exact_match_acc"])
            results["p_mae"].append(metrics.get("p_mae", np.nan))
            results["k_mae"].append(metrics.get("k_mae", np.nan))
            print(f"[sweep] p={p} k={k} -> acc={metrics['exact_match_acc']:.3f}")

        viz.plot_sweep(
            p_values, accs, xlabel="p", ylabel="exact-match accuracy",
            title=f"accuracy vs p (k={k})",
            save_path=os.path.join(out_dir, f"viz_acc_vs_p_k{k}.png"),
        )
        if cfg["predict_pk"]:
            viz.plot_sweep(
                p_values, p_maes, xlabel="true p", ylabel="p estimation MAE",
                title=f"p-estimation error vs p (k={k})",
                save_path=os.path.join(out_dir, f"viz_pmae_vs_p_k{k}.png"),
            )
    return results
