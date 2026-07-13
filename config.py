"""
config.py
=========
Edit the values below and just run e.g. `python main.py train` with NO
flags — main.py reads its defaults from here. You can still override any
individual value from the command line (e.g. `python main.py train --p 0.5`)
if you want a one-off change without editing this file.

Each subcommand ("generate", "train", "rollout", "analyze", "sweep",
"multi") has its own dict below. Keys match the --flag names used in
main.py (with dashes turned into underscores).
"""

GENERATE = {
    "p": 0.3,
    "k": 1,
    "grid": 10,
    "T": 100,
    "directions": 4,
    "step_mode": "fixed",     # "fixed" or "random"
    "boundary": "clip",       # "clip" or "wrap"
    "seed": 0,
    "out": "trajectory.npz",
}

DEVICE = "auto"   # "auto" (cuda if available, else cpu), or force "cpu" / "cuda" / "cuda:0" / "mps"

TRAIN = {
    "task": "coord",         # "coord", "full", or "both"
    "p": 0.3,
    "k": 1,
    "grid": 10,
    "T": 100,                 # frames per training sequence
    "directions": 4,
    "n_layers": 1,
    "d_model": 64,
    "n_heads": 4,
    "steps": 2000,
    "batch_size": 16,
    "lr": 3e-4,
    "curriculum_frac": 0.3,   # fraction of steps to keep step size FIXED first
    "step_mode_final": "fixed",  # "fixed" or "random", used after the curriculum switch
    "seed": 0,
    "log_every": 50,
    "save": "checkpoint.pt",
    "history_out": "history.json",
    "device": DEVICE,
}

ROLLOUT = {
    "checkpoint": "checkpoint.pt",
    "context": 50,           # how many real frames to condition on
    "future": 50,            # how many frames to predict
    "p": None,                # None = use the p the checkpoint was trained with
    "k": None,                # None = use the k the checkpoint was trained with
    "sampling": "argmax",     # "argmax" or "sample"
    "seed": 123,
    "out": "rollout.png",
    "device": DEVICE,
}

ANALYZE = {
    "checkpoint": "checkpoint.pt",
    "p": None,
    "k": None,
    "T": None,
    "device": DEVICE,
}

SWEEP = {
    "p_list": "0.1,0.2,0.3",
    "k_list": "1,2",
    "t_list": "30,60",
    "grid": 10,
    "steps": 1000,
    "d_model": 64,
    "n_layers": 1,
    "batch_size": 16,
    "seed": 0,
    "out": "sweep_results.json",
}

SWEEP_DKT = {
    # sweep d_model x k x T x grid at a FIXED p -- sized to be safe on a
    # laptop. See the "memory budget" note in the README for how these
    # numbers were chosen (est. attention memory stays well under a few GiB).
    "d_model_list": "32,64,128",
    "k_list": "1,2",
    "t_list": "10,20,30",
    "grid_list": "6,8,10",
    "p": 0.3,
    "steps": 800,
    "n_layers": 1,
    "n_heads": 4,
    "batch_size": 8,
    "step_mode_final": "fixed",
    "device": DEVICE,
    "max_mem_gib": 4.0,
    "seed": 0,
    "out": "sweep_dkt_results.json",
    "save_frames": False,     # save each combo's generated preview frames to disk
    "frames_dir": "sweep_frames",
}

MULTI = {
    "variant": "labeled",    # "labeled" or "unlabeled"
    "p_list": "0.1,0.4",     # comma-separated, one per walker
    "k_list": "1,1",
    "grid": 10,
    "T": 30,
    "steps": 500,
    "d_model": 64,
    "n_layers": 1,
    "batch_size": 16,
    "seed": 0,
    "log_every": 50,
    "device": DEVICE,
}
