"""
main.py
-------
Single entrypoint with 4 phases, run independently so you can test each
part of the pipeline in isolation:

    python main.py generate  ...   # simulate a dataset + visualize samples
    python main.py train     ...   # train the transformer + visualize training
    python main.py evaluate  ...   # evaluate a trained model on one (p, k)
    python main.py generalize ...  # sweep p (and k) to study generalization

Examples
--------
# Phase 1: fixed p, k dataset (pure trajectory prediction)
python main.py generate --out_path data/train.npz --n_sequences 2000 \
    --d 10 --seq_len 100 --p 0.2 --k 1 --seed 0

python main.py generate --out_path data/val.npz --n_sequences 200 \
    --d 10 --seq_len 100 --p 0.2 --k 1 --seed 1

python main.py train --data_path data/train.npz --out_dir runs/exp1 \
    --t_obs 50 --t_future 50 --epochs 20 --n_layers 1 --predict_pk

python main.py evaluate --model_path runs/exp1/model.pt --p 0.2 --k 1 \
    --n_sequences 200 --out_dir runs/exp1/eval_p0.2

# Phase 2: p varies per sequence -> the model can actually learn to
# estimate p (train k fixed, p sampled in [0.05, 0.4])
python main.py generate --out_path data/train_pvar.npz --n_sequences 4000 \
    --d 10 --seq_len 100 --p_range 0.05 0.4 --k 1 --seed 0

python main.py train --data_path data/train_pvar.npz --out_dir runs/exp2 \
    --t_obs 50 --t_future 50 --epochs 20 --predict_pk

# Generalization study: does accuracy / p-estimation hold up across p not
# necessarily seen in exactly that value during training?
python main.py generalize --model_path runs/exp2/model.pt \
    --p_values 0.05 0.1 0.2 0.3 0.4 0.5 --k_values 1 \
    --out_dir runs/exp2/generalize
"""

import argparse

from dataset import generate_dataset, save_dataset
import viz
import train as train_mod
import evaluate as eval_mod


def cmd_generate(args):
    p_range = tuple(args.p_range) if args.p_range else None
    k_range = tuple(args.k_range) if args.k_range else None

    batch = generate_dataset(
        n_sequences=args.n_sequences,
        d=args.d,
        seq_len=args.seq_len,
        p=args.p,
        k=args.k,
        p_range=p_range,
        k_range=k_range,
        step_size_random=args.step_size_random,
        seed=args.seed,
    )
    save_dataset(batch, args.out_path)
    print(f"[generate] saved {args.n_sequences} sequences to {args.out_path}")
    print(f"[generate] p range in data: [{batch.p_values.min():.3f}, {batch.p_values.max():.3f}]")
    print(f"[generate] k range in data: [{batch.k_values.min():.1f}, {batch.k_values.max():.1f}]")

    if args.visualize:
        viz.plot_sample_sequences(batch.frames, batch.positions, n_sequences=3, n_frames=8,
                                    save_path=args.viz_path.replace(".png", "_samples.png"))
        viz.plot_trajectory(batch.positions[0], d=args.d,
                              save_path=args.viz_path.replace(".png", "_trajectory.png"))
        print(f"[generate] wrote visualizations next to {args.viz_path}")


def cmd_train(args):
    train_mod.run_training(
        data_path=args.data_path,
        out_dir=args.out_dir,
        t_obs=args.t_obs,
        t_future=args.t_future,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        predict_pk=args.predict_pk,
        lambda_xy=args.lambda_xy,
        lambda_p=args.lambda_p,
        lambda_k=args.lambda_k,
        val_split=args.val_split,
        seed=args.seed,
        viz_every=args.viz_every,
    )


def cmd_evaluate(args):
    eval_mod.run_single_evaluation(
        model_path=args.model_path,
        n_sequences=args.n_sequences,
        p=args.p,
        k=args.k,
        out_dir=args.out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
    )


def cmd_generalize(args):
    eval_mod.run_generalization_sweep(
        model_path=args.model_path,
        p_values=args.p_values,
        k_values=args.k_values,
        n_sequences=args.n_sequences,
        out_dir=args.out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Dot-on-a-grid random walk + transformer project")
    sub = parser.add_subparsers(dest="phase", required=True)

    # ---- generate ----
    g = sub.add_parser("generate", help="simulate a dataset of random-walk sequences")
    g.add_argument("--out_path", type=str, required=True)
    g.add_argument("--n_sequences", type=int, default=1000)
    g.add_argument("--d", type=int, default=10, help="grid side length (grid is d x d)")
    g.add_argument("--seq_len", type=int, default=100, help="total frames per sequence (>= t_obs+t_future)")
    g.add_argument("--p", type=float, default=0.2, help="fixed move probability (used if --p_range not given)")
    g.add_argument("--k", type=int, default=1, help="fixed step size (used if --k_range not given)")
    g.add_argument("--p_range", type=float, nargs=2, default=None, metavar=("PMIN", "PMAX"),
                    help="sample p per-sequence from this range instead of using a fixed p")
    g.add_argument("--k_range", type=int, nargs=2, default=None, metavar=("KMIN", "KMAX"),
                    help="sample step size per-sequence from this range instead of using a fixed k")
    g.add_argument("--step_size_random", action="store_true",
                    help="randomize the step size on every individual move (within [1, k])")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--visualize", action="store_true", default=True)
    g.add_argument("--viz_path", type=str, default="data/viz.png")

    # ---- train ----
    t = sub.add_parser("train", help="train the transformer on a generated dataset")
    t.add_argument("--data_path", type=str, required=True)
    t.add_argument("--out_dir", type=str, default="runs/exp1")
    t.add_argument("--t_obs", type=int, default=50, help="number of observed frames")
    t.add_argument("--t_future", type=int, default=50, help="number of frames to predict")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d_model", type=int, default=64, help="transformer embedding dim (make large enough for asymptotic behavior)")
    t.add_argument("--n_heads", type=int, default=4)
    t.add_argument("--n_layers", type=int, default=1)
    t.add_argument("--predict_pk", action="store_true", default=True)
    t.add_argument("--no_predict_pk", dest="predict_pk", action="store_false")
    t.add_argument("--lambda_xy", type=float, default=1.0)
    t.add_argument("--lambda_p", type=float, default=0.1)
    t.add_argument("--lambda_k", type=float, default=0.1)
    t.add_argument("--val_split", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--viz_every", type=int, default=5, help="visualize a sample prediction every N epochs")

    # ---- evaluate ----
    e = sub.add_parser("evaluate", help="evaluate a trained model on a fresh test set at one (p, k)")
    e.add_argument("--model_path", type=str, required=True)
    e.add_argument("--n_sequences", type=int, default=200)
    e.add_argument("--p", type=float, default=0.2)
    e.add_argument("--k", type=int, default=1)
    e.add_argument("--out_dir", type=str, default="eval_out")
    e.add_argument("--seed", type=int, default=123)
    e.add_argument("--batch_size", type=int, default=4,
                    help="keep this small: sequence length is t_obs*d*d + t_future tokens, "
                         "so attention memory grows fast with batch_size")

    # ---- generalize ----
    gen = sub.add_parser("generalize", help="sweep p (and k) to study generalization / sensitivity")
    gen.add_argument("--model_path", type=str, required=True)
    gen.add_argument("--p_values", type=float, nargs="+", required=True)
    gen.add_argument("--k_values", type=int, nargs="+", default=[1])
    gen.add_argument("--n_sequences", type=int, default=200)
    gen.add_argument("--out_dir", type=str, default="generalize_out")
    gen.add_argument("--seed", type=int, default=123)
    gen.add_argument("--batch_size", type=int, default=4)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "generalize": cmd_generalize,
    }[args.phase](args)


if __name__ == "__main__":
    main()
