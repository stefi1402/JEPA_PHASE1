"""
train.py
--------
Trains DotTransformer to predict the (row, col) of the dot for the next
t_future frames given the first t_obs observed frames, optionally also
estimating p and k from the observed sequence.

Run as a phase from main.py, e.g.:
    python main.py train --data_path data/train.npz --epochs 20 ...
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import load_dataset, SequenceBatch
from model import DotTransformer
import viz


class WalkDataset(Dataset):
    """Wraps a SequenceBatch, splitting each sequence into observed / future."""

    def __init__(self, batch: SequenceBatch, t_obs: int, t_future: int):
        total_needed = t_obs + t_future
        assert batch.frames.shape[1] >= total_needed, (
            f"sequences have length {batch.frames.shape[1]} but t_obs+t_future="
            f"{total_needed}"
        )
        self.frames = batch.frames[:, :total_needed]
        self.positions = batch.positions[:, :total_needed]
        self.p_values = batch.p_values
        self.k_values = batch.k_values
        self.t_obs = t_obs
        self.t_future = t_future
        self.d = batch.d

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        frames_obs = self.frames[idx, : self.t_obs]
        pos_future = self.positions[idx, self.t_obs:self.t_obs + self.t_future]
        return (
            torch.from_numpy(frames_obs).float(),
            torch.from_numpy(pos_future).long(),
            torch.tensor(self.p_values[idx], dtype=torch.float32),
            torch.tensor(self.k_values[idx], dtype=torch.float32),
        )


def run_training(
    data_path: str,
    out_dir: str = "runs/exp1",
    t_obs: int = 50,
    t_future: int = 50,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 3e-4,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 1,
    predict_pk: bool = True,
    lambda_xy: float = 1.0,
    lambda_p: float = 0.1,
    lambda_k: float = 0.1,
    val_split: float = 0.1,
    seed: int = 0,
    device: str = None,
    viz_every: int = 5,
    num_workers: int = 2,
    amp: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[info] training on device: {device}")
    use_cuda = device.startswith("cuda")
    # AMP only helps (and is only supported the same way) on CUDA.
    amp_enabled = amp and use_cuda

    batch = load_dataset(data_path)
    dataset = WalkDataset(batch, t_obs, t_future)

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # quick sanity-check visualization of the raw data before training starts
    viz.plot_sample_sequences(
        batch.frames, batch.positions, n_sequences=3, n_frames=8,
        save_path=os.path.join(out_dir, "viz_data_samples.png"),
    )
    viz.plot_trajectory(
        batch.positions[0], d=batch.d,
        save_path=os.path.join(out_dir, "viz_data_trajectory_example.png"),
    )

    model = DotTransformer(
        d=batch.d, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_frames=t_obs + t_future + 5, predict_pk=predict_pk,
    ).to(device)

    n_tokens = model.num_tokens(t_obs, t_future)
    print(f"[info] sequence length fed to transformer: {n_tokens} tokens "
          f"({t_obs} obs frames x {batch.d}x{batch.d} pixels + {t_future} query tokens)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {"train_loss": [], "val_loss": [], "val_xy_acc": []}
    if predict_pk:
        history["val_p_mae"] = []
        history["val_k_mae"] = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        # Accumulate on-device; a single .item() per epoch instead of one
        # per batch avoids a GPU/CPU sync point on every training step.
        total_loss = torch.zeros((), device=device)
        for frames_obs, pos_future, p_true, k_true in train_loader:
            frames_obs = frames_obs.to(device, non_blocking=True)
            pos_future = pos_future.to(device, non_blocking=True)
            p_true = p_true.to(device, non_blocking=True)
            k_true = k_true.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(frames_obs, t_future)
                row_loss = ce(out["row_logits"].reshape(-1, batch.d), pos_future[..., 0].reshape(-1))
                col_loss = ce(out["col_logits"].reshape(-1, batch.d), pos_future[..., 1].reshape(-1))
                loss = lambda_xy * (row_loss + col_loss)

                if predict_pk:
                    loss = loss + lambda_p * mse(out["p_pred"], p_true)
                    loss = loss + lambda_k * mse(out["k_pred"], k_true)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * frames_obs.size(0)

        train_loss = total_loss.item() / len(train_set)

        # ---- validation ----
        model.eval()
        val_loss_total = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        count = 0
        p_abs_err = torch.zeros((), device=device)
        k_abs_err = torch.zeros((), device=device)
        n_pk = 0
        with torch.no_grad():
            for frames_obs, pos_future, p_true, k_true in val_loader:
                frames_obs = frames_obs.to(device, non_blocking=True)
                pos_future = pos_future.to(device, non_blocking=True)
                p_true = p_true.to(device, non_blocking=True)
                k_true = k_true.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", enabled=amp_enabled):
                    out = model(frames_obs, t_future)
                    row_loss = ce(out["row_logits"].reshape(-1, batch.d), pos_future[..., 0].reshape(-1))
                    col_loss = ce(out["col_logits"].reshape(-1, batch.d), pos_future[..., 1].reshape(-1))
                    loss = lambda_xy * (row_loss + col_loss)
                    if predict_pk:
                        loss = loss + lambda_p * mse(out["p_pred"], p_true)
                        loss = loss + lambda_k * mse(out["k_pred"], k_true)
                val_loss_total += loss.detach() * frames_obs.size(0)

                row_pred = out["row_logits"].argmax(-1)
                col_pred = out["col_logits"].argmax(-1)
                correct += ((row_pred == pos_future[..., 0]) & (col_pred == pos_future[..., 1])).sum()
                count += pos_future[..., 0].numel()

                if predict_pk:
                    p_abs_err += (out["p_pred"] - p_true).abs().sum()
                    k_abs_err += (out["k_pred"] - k_true).abs().sum()
                    n_pk += p_true.numel()

        val_loss = val_loss_total.item() / len(val_set)
        val_acc = correct.item() / count
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_xy_acc"].append(val_acc)

        msg = (f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
               f"val_loss {val_loss:.4f} | val_xy_exact_acc {val_acc:.3f} | "
               f"{time.time()-t0:.1f}s")
        if predict_pk:
            p_mae = p_abs_err.item() / n_pk
            k_mae = k_abs_err.item() / n_pk
            history["val_p_mae"].append(p_mae)
            history["val_k_mae"].append(k_mae)
            msg += f" | val_p_MAE {p_mae:.4f} | val_k_MAE {k_mae:.4f}"
        print(msg)

        if epoch % viz_every == 0 or epoch == epochs:
            _visualize_one_prediction(model, val_set, batch.d, t_obs, t_future,
                                       device, out_dir, epoch)

    viz.plot_training_curves(history, save_path=os.path.join(out_dir, "viz_training_curves.png"))
    ckpt_path = os.path.join(out_dir, "model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": dict(d=batch.d, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        t_obs=t_obs, t_future=t_future, predict_pk=predict_pk),
    }, ckpt_path)
    print(f"[info] saved checkpoint to {ckpt_path}")
    return model, history


def _visualize_one_prediction(model, val_set, d, t_obs, t_future, device, out_dir, epoch):
    model.eval()
    frames_obs, pos_future, p_true, k_true = val_set[0]
    with torch.no_grad():
        out = model(frames_obs.unsqueeze(0).to(device), t_future)
        row_pred = out["row_logits"].argmax(-1).squeeze(0).cpu().numpy()
        col_pred = out["col_logits"].argmax(-1).squeeze(0).cpu().numpy()
    pred_pos = np.stack([row_pred, col_pred], axis=-1)
    viz.plot_prediction_vs_truth(
        pos_future.numpy(), pred_pos, d, t_obs,
        save_path=os.path.join(out_dir, f"viz_prediction_epoch{epoch:03d}.png"),
    )
