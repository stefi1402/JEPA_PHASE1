"""
sweep.py
--------
Runs the full generate -> train -> evaluate pipeline across a grid of
(d, k, p, seq_len) combinations, and logs the resulting metrics as rows
directly into a Google Sheet via the Sheets API (through a service account).

Drop this file into the root of the JEPA_PHASE1 repo (next to main.py,
dataset.py, train.py, evaluate.py) -- it imports those modules directly.

------------------------------------------------------------------------
ONE-TIME SETUP: Google Sheets access via a service account
------------------------------------------------------------------------
An HPC batch job has no browser, so normal Google OAuth ("click Allow")
doesn't work here. A service account is a machine identity with its own
key file instead of a human login -- you share your Sheet with it like
you'd share it with a person.

1. Go to https://console.cloud.google.com/, create a project (or reuse
   one you have).
2. In "APIs & Services" -> "Library", enable:
       - Google Sheets API
       - Google Drive API
3. In "APIs & Services" -> "Credentials" -> "Create Credentials" ->
   "Service Account". Give it any name (e.g. "jepa-sweep").
4. Open the new service account -> "Keys" -> "Add Key" -> "Create new
   key" -> JSON. This downloads a .json file -- this is your credential.
5. Open the JSON file and copy the "client_email" value (looks like
   jepa-sweep@your-project.iam.gserviceaccount.com).
6. Open your target Google Sheet in a browser, click "Share", and share
   it with that email address, with Editor access.
7. Copy the JSON key file onto the HPC, e.g.:
       scp gsheet_key.json youruser@cluster:~/JEPA_PHASE1/
   Do NOT commit this file to git -- add it to .gitignore.
8. Install the two extra packages (from inside your repo / venv):
       uv pip install gspread google-auth
   (or: pip install --user gspread google-auth)

------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------
python sweep.py \
    --sheet_id YOUR_SHEET_ID \
    --creds_path gsheet_key.json \
    --d_values 6 10 \
    --k_values 1 2 \
    --p_values 0.1 0.2 0.3 \
    --seq_len_values 60 100 \
    --epochs 10 \
    --n_sequences 1000

The sheet ID is the long string in the sheet's URL:
    https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit

Note on scale: this runs generate+train+evaluate SEQUENTIALLY for every
combination in the grid, so the number of runs is the *product* of all
the value lists (e.g. 2 d_values x 2 k_values x 3 p_values x 2 seq_lens =
24 full training runs). Each training run can take a while, especially
at larger d/seq_len (see the repo README's compute note on sequence
length = t_obs*d*d + t_future). Start with a small grid and few epochs
to sanity check the whole loop (incl. the Sheets writing) before scaling
up. If you have GPU access via Slurm, consider running this as one job
per combination in a Slurm array instead of one long sequential job --
ask if you'd like help writing that array script.
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
import traceback

import gspread
from google.oauth2.service_account import Credentials

from dataset import generate_dataset, save_dataset
from train import run_training
from evaluate import run_single_evaluation

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADER = [
    "run_id", "d", "k", "p", "seq_len", "t_obs", "t_future",
    "n_sequences", "epochs", "exact_match_acc", "p_mae", "k_mae",
    "train_seconds", "status", "error",
]


def get_worksheet(sheet_id: str, creds_path: str, worksheet_name: str = "Sheet1"):
    """Authorizes with the service account and returns a worksheet handle,
    creating it (and writing the header row) if it doesn't exist yet."""
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=len(HEADER))
    if ws.row_values(1) != HEADER:
        ws.update("A1", [HEADER])
    return ws


def append_row_with_retry(ws, row: list, max_retries: int = 5):
    """Sheets API has per-minute rate limits; back off and retry rather
    than losing a result if a single append call gets throttled."""
    for attempt in range(max_retries):
        try:
            ws.append_row(row, value_input_option="USER_ENTERED")
            return
        except Exception as e:
            wait = 2 ** attempt
            print(f"[sheets] append failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    print("[sheets] giving up on this row after retries -- printing it instead:")
    print(row)


def run_one_combo(run_id, d, k, p, seq_len, epochs, n_sequences,
                   n_sequences_eval, work_dir):
    """Runs generate -> train -> evaluate for a single (d, k, p, seq_len)
    combination and returns the evaluation metrics dict."""
    t_obs = seq_len // 2
    t_future = seq_len - t_obs

    data_path = os.path.join(work_dir, f"data_{run_id}.npz")
    out_dir = os.path.join(work_dir, f"run_{run_id}")

    batch = generate_dataset(
        n_sequences=n_sequences, d=d, seq_len=seq_len, p=p, k=k, seed=0,
    )
    save_dataset(batch, data_path)

    run_training(
        data_path=data_path, out_dir=out_dir, t_obs=t_obs, t_future=t_future,
        epochs=epochs, predict_pk=True, seed=0,
    )

    model_path = os.path.join(out_dir, "model.pt")
    metrics = run_single_evaluation(
        model_path=model_path, n_sequences=n_sequences_eval, p=p, k=k,
        out_dir=os.path.join(out_dir, "eval"), seed=123,
    )
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Sweep (d, k, p, seq_len) and log results to Google Sheets")
    ap.add_argument("--sheet_id", type=str, required=True, help="Google Sheet ID (from its URL)")
    ap.add_argument("--creds_path", type=str, required=True, help="path to the service-account JSON key")
    ap.add_argument("--worksheet_name", type=str, default="Sheet1")
    ap.add_argument("--d_values", type=int, nargs="+", default=[6, 10])
    ap.add_argument("--k_values", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--p_values", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    ap.add_argument("--seq_len_values", type=int, nargs="+", default=[60, 100])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n_sequences", type=int, default=1000, help="sequences to generate for training")
    ap.add_argument("--n_sequences_eval", type=int, default=200, help="sequences to generate for evaluation")
    ap.add_argument("--work_dir", type=str, default="sweep_runs", help="where per-combo data/checkpoints go")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    ws = get_worksheet(args.sheet_id, args.creds_path, args.worksheet_name)

    combos = list(itertools.product(
        args.d_values, args.k_values, args.p_values, args.seq_len_values,
    ))
    print(f"[sweep] {len(combos)} combinations to run")

    for i, (d, k, p, seq_len) in enumerate(combos):
        run_id = f"d{d}_k{k}_p{p}_s{seq_len}"
        print(f"\n[sweep] ({i + 1}/{len(combos)}) {run_id}")

        row = [run_id, d, k, p, seq_len, seq_len // 2, seq_len - seq_len // 2,
               args.n_sequences, args.epochs, "", "", "", "", "", ""]

        t0 = time.time()
        try:
            metrics = run_one_combo(
                run_id, d, k, p, seq_len, args.epochs,
                args.n_sequences, args.n_sequences_eval, args.work_dir,
            )
            elapsed = time.time() - t0
            row[9] = metrics.get("exact_match_acc", "")
            row[10] = metrics.get("p_mae", "")
            row[11] = metrics.get("k_mae", "")
            row[12] = round(elapsed, 1)
            row[13] = "ok"
            row[14] = ""
            print(f"[sweep] {run_id} -> acc={row[9]} p_mae={row[10]} k_mae={row[11]} ({elapsed:.1f}s)")
        except Exception as e:
            row[13] = "error"
            row[14] = str(e)
            print(f"[sweep] {run_id} FAILED: {e}")
            traceback.print_exc()

        append_row_with_retry(ws, row)

    print("\n[sweep] done")


if __name__ == "__main__":
    main()
