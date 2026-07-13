"""
sheets_logger.py
================
Append evaluate.py / generalize sweep results to a Google Sheet, one row
per (p, k) combo evaluated.

Columns written (in this order):
  n_sequences | p | k | exact_match_acc | p_mae | k_mae

--------------------------------------------------------------------------
ONE-TIME SETUP (needed once, not per run):
  1. https://console.cloud.google.com/ -> create/select a project.
  2. "APIs & Services" -> "Library" -> enable "Google Sheets API".
  3. "APIs & Services" -> "Credentials" -> "Create Credentials" ->
     "Service Account" (any name, no special roles needed).
  4. Open the service account -> "Keys" -> "Add Key" -> "Create new key"
     -> JSON. Save the downloaded file, e.g. gsheet_creds.json, in this
     project folder (do NOT commit it to git -- it's a secret).
  5. Open that JSON file, copy the "client_email" value.
  6. Open your Google Sheet -> "Share" -> paste that email -> "Editor" ->
     Share.
  7. Install the two packages this needs:
       uv pip install gspread google-auth
  8. Set GSHEET_ENABLED / GSHEET_ID / GSHEET_CREDS_PATH below (or just
     pass gsheet_enabled=True explicitly from your own script/call).
--------------------------------------------------------------------------
"""

# Defaults -- edit these, or override per-call via the gsheet_* kwargs
# added to run_single_evaluation / run_generalization_sweep.
GSHEET_ENABLED = True
GSHEET_ID = "1yollsNzsx537_d9qvimnrUxt79xdEe9pIVro-GYKYlw"
GSHEET_CREDS_PATH = "gsheet_creds.json"
GSHEET_WORKSHEET = "Sheet1"

HEADER = ["n_sequences", "p", "k", "exact_match_acc", "p_mae", "k_mae"]

_worksheet_cache = {}


def _get_worksheet(sheet_id, creds_path, worksheet_name="Sheet1"):
    import gspread
    from google.oauth2.service_account import Credentials

    key = (sheet_id, creds_path, worksheet_name)
    if key in _worksheet_cache:
        return _worksheet_cache[key]

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.sheet1

    existing = ws.get_all_values()
    if not existing:
        ws.append_row(HEADER)

    _worksheet_cache[key] = ws
    return ws


def log_eval_result(n_sequences, p, k, metrics, sheet_id=None, creds_path=None,
                     worksheet_name=None):
    """metrics: the dict returned by evaluate_on_batch() -- must contain
    exact_match_acc, and optionally p_mae/k_mae (present when the model
    was trained with predict_pk). Never raises -- logs a warning instead,
    so a Sheets hiccup never kills a long-running sweep."""
    sheet_id = sheet_id or GSHEET_ID
    creds_path = creds_path or GSHEET_CREDS_PATH
    worksheet_name = worksheet_name or GSHEET_WORKSHEET
    try:
        ws = _get_worksheet(sheet_id, creds_path, worksheet_name)
        row = [n_sequences, p, k, metrics.get("exact_match_acc"),
               metrics.get("p_mae"), metrics.get("k_mae")]
        ws.append_row(row)
        print(f"           logged to Google Sheet: p={p} k={k}")
    except ImportError:
        print("           [WARN] gspread/google-auth not installed -- skipping Sheet log. "
              "Run: uv pip install gspread google-auth")
    except Exception as e:
        print(f"           [WARN] failed to log to Google Sheet: {e}")
