"""
rollout.py
==========
Autoregressive rollout: given a trained model and a context of real frames,
predict the walker's position for the next N frames, one frame at a time.

How it works:
  1. Start from `context_flat`, the flattened pixel sequence of the first
     `t_context` real frames (ground truth, e.g. from walk2d.simulate_walk).
  2. Run the model on the sequence so far; read the coord_head's
     prediction at the LAST frame-boundary token -- that's the model's
     belief about the walker's position in the NEXT frame.
  3. Turn that predicted position into a new synthetic frame (a -1/+1
     image with +1 at the predicted cell) and append it to the sequence.
  4. Repeat for `n_future` steps.

This is exactly "train on generated frames, then ask the model to predict
the walker's position for the next 50 frames": the model never sees the
true future, only its own previous predictions, fed back in.
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def rollout_predict(model, grid, context_flat, n_future, sampling="argmax", device="cpu"):
    """
    context_flat: (1, t_context*grid*grid) tensor of -1/+1 pixel values
                  (a single sequence, batch size 1).
    Returns:
      pred_positions: (n_future,) array of flattened position indices
                      (one per generated future frame)
      full_sequence:  (1, (t_context+n_future)*grid*grid) tensor -- the
                      context plus all the generated frames, so you can
                      keep extending it or visualize it.
    """
    model.eval()
    N = grid * grid
    seq = context_flat.clone().to(device)
    pred_positions = []

    for _ in range(n_future):
        out = model(seq, need_coord=True, need_pixel=False)
        logits = out["coord_logits"][:, -1, :]     # prediction for the NEXT frame
        probs = F.softmax(logits, dim=-1)
        if sampling == "argmax":
            next_idx = probs.argmax(dim=-1)          # (1,)
        else:  # "sample"
            next_idx = torch.multinomial(probs, 1).squeeze(-1)
        pred_positions.append(int(next_idx.item()))

        new_frame = -torch.ones(1, N, device=device)
        new_frame[0, next_idx] = 1.0
        seq = torch.cat([seq, new_frame], dim=1)

    return np.array(pred_positions), seq


def positions_to_rc(idx_array, grid):
    """flattened index array -> (row, col) array."""
    idx_array = np.asarray(idx_array)
    return np.stack([idx_array // grid, idx_array % grid], axis=-1)
