# Dot-on-a-grid random walk + Transformer

A single dot performs a lazy random walk on a `d x d` grid (rendered as a
+1/-1 image). At every timestep, with probability **p** the dot moves one
step of size **k** in a direction picked uniformly from {up, down, left,
right}; with probability **1-p** it stays put. A Transformer observes the
first `t_obs` frames and predicts the (row, col) of the dot for the next
`t_future` frames, and (optionally) estimates **p** and **k** from what it
observed.

## Project layout

```
dataset.py    simulate the random walk, save/load .npz datasets
model.py      the Transformer (DotTransformer)
train.py      training loop (used by `main.py train`)
evaluate.py   single evaluation + p/k sweep (used by `main.py evaluate|generalize`)
viz.py        all plotting helpers
main.py       argparse entrypoint, 4 independent phases
```

## Architecture notes (matches the two-positional-encoding spec)

Each frame is flattened into `d*d` pixel tokens. All `t_obs` observed
frames are concatenated into one long sequence of
`t_obs * d * d` pixel tokens. Every token gets **two** positional
encodings added to it:

- **pixel positional encoding** — which of the `d*d` grid cells it is
  (`nn.Embedding(d*d, d_model)`)
- **frame positional encoding** — which timestep/frame it came from
  (`nn.Embedding(max_frames, d_model)`)

To predict the future, `t_future` learned **query tokens** are appended
(no pixel content, since the future is unobserved) — each carries the
frame positional encoding of the future timestep it's asking about. The
Transformer encoder attends over observed pixels + queries, and the
(row, col) prediction is read off each query token's output (as two
classification heads over `d` classes each). `p`/`k` estimates are read
off a mean-pooled representation of the observed tokens.

**Compute note:** sequence length = `t_obs * d*d + t_future`. E.g.
`d=10, t_obs=50` → 5000 pixel tokens + up to 50 query tokens ≈ 5050
tokens per sample. This is the literal architecture from the spec, but
it does mean CPU training will be slow at full scale (10x10, t_obs=50).
Start with a smaller `d` / `t_obs` while iterating, then scale up.

**Important — for p/k estimation & generalization to be meaningful:**
if every sequence in a dataset uses the exact same `p` (or `k`), the
model has nothing to estimate — it will just learn a constant. Use
`--p_range PMIN PMAX` (and/or `--k_range KMIN KMAX`) when generating data
for those studies; use a fixed `--p`/`--k` for the plain trajectory
prediction phase, per the "keep the same step size in the beginning"
instruction.

## Quickstart

```bash
pip install torch numpy matplotlib

cd dot_transformer

# ---- Phase 1: fixed p, k — pure trajectory prediction ----
python main.py generate --out_path data/train.npz --n_sequences 2000 \
    --d 10 --seq_len 100 --p 0.2 --k 1 --seed 0

python main.py train --data_path data/train.npz --out_dir runs/exp1 \
    --t_obs 50 --t_future 50 --epochs 20 --n_layers 1 --predict_pk

python main.py evaluate --model_path runs/exp1/model.pt \
    --p 0.2 --k 1 --n_sequences 200 --out_dir runs/exp1/eval_p0.2

# ---- Phase 2: p varies per sequence -> model can learn to estimate p ----
python main.py generate --out_path data/train_pvar.npz --n_sequences 4000 \
    --d 10 --seq_len 100 --p_range 0.05 0.4 --k 1 --seed 0

python main.py train --data_path data/train_pvar.npz --out_dir runs/exp2 \
    --t_obs 50 --t_future 50 --epochs 20 --predict_pk

# ---- Generalization / sensitivity sweep across p (and k) ----
python main.py generalize --model_path runs/exp2/model.pt \
    --p_values 0.05 0.1 0.2 0.3 0.4 0.5 --k_values 1 2 \
    --out_dir runs/exp2/generalize
```

## What each phase visualizes

- `generate`: saves a strip of example frames per sequence + one full
  trajectory plot, so you can eyeball whether movement frequency/step
  size look right before you spend time training.
- `train`: training/validation loss curves, validation exact-match
  accuracy, p/k MAE curves (if `--predict_pk`), and a predicted-vs-true
  trajectory overlay every `--viz_every` epochs (so you can watch the
  prediction improve during training).
- `evaluate`: per-future-timestep accuracy (does accuracy degrade the
  further into the future you predict?).
- `generalize`: accuracy vs p (and vs k), and p-estimation-error vs true
  p, so you can see how performance/estimation quality depends on p, k,
  and see whether the model generalizes to p values it wasn't
  necessarily trained on.

## Things you'll likely want to tweak next

- `--n_layers` (spec suggests 1–2 is enough), `--n_heads`, `--d_model`
  (make large enough that behavior stops changing much as you increase
  it further — an "asymptotic in d_model" check).
- `--step_size_random` to randomize step size on every move rather than
  keeping it fixed per sequence.
- Boundary behavior currently clips at the grid edge (the dot doesn't
  wrap around, and if a proposed move would leave the grid it's clipped
  to the edge cell). Reflecting or wrapping would be a one-line change
  in `dataset.py::simulate_sequence` if you'd rather do that.
- Correlations between t (=`t_obs`/`t_future`), k, and p: run the
  `generalize` sweep at a few different `t_obs` (i.e., train separate
  models with different `--t_obs`) to see how the length of observation
  interacts with how well p/k can be estimated.
