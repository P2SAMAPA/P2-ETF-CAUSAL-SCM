"""
persistence.py — Rolling history + streak-based persistence gate
===================================================================

Stores one OOS R² value per (universe, window, ticker, method) per day,
trimmed to config.HISTORY_RETENTION_DAYS. Used by trainer.py to gate the
top-N picks: a combo only qualifies if it shows positive OOS R² on
config.MIN_PERSISTENCE_DAYS CONSECUTIVE most-recent daily runs, including
today — not just today's snapshot. A single lucky day should never be
enough to earn a "top pick" label on its own (see config.py's comment on
MIN_PERSISTENCE_DAYS for the concrete example that motivated this).

History schema
--------------
{
  "_meta": {"run_dates": ["2026-08-08", "2026-08-09", ...]},
  "FI_COMMODITIES": {
    "63": {
      "TLT": {
        "varlingam": [{"date": "2026-08-08", "r2": -0.05},
                      {"date": "2026-08-09", "r2": 0.01}, ...],
        "pcmci": [...],
        "timino": [...]
      },
      ...
    },
    ...
  },
  ...
}
Entries are stored oldest-first, trimmed to the trailing
HISTORY_RETENTION_DAYS on every update. Re-running trainer.py on the same
calendar day updates that day's entry in place rather than appending a
duplicate (idempotent).
"""

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

import config


def load_history() -> dict:
    """
    Download the existing history file from HF. Returns a fresh, empty
    history structure on first run (file doesn't exist yet) — this is
    expected, not an error condition.
    """
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    try:
        path = hf_hub_download(
            repo_id=config.OUTPUT_REPO,
            filename=config.HISTORY_FILENAME,
            repo_type="dataset",
            token=token or None,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"_meta": {"run_dates": []}}


def record_day(history: dict, run_date: str, universe: str, window: int,
                win_result: dict) -> None:
    """
    Append today's OOS R² for every (ticker, method) scored in this
    universe/window into history, mutating history in place. win_result has
    the same shape process_universe_window() returns:
    {ticker: {method: {...}, "regime_shift": {...}}}.
    """
    win_key = str(window)
    uni_node = history.setdefault(universe, {})
    win_node = uni_node.setdefault(win_key, {})

    for ticker, methods in win_result.items():
        ticker_node = win_node.setdefault(ticker, {})
        for method in config.CAUSAL_METHODS:
            if method not in methods:
                continue
            r2 = methods[method]["oos_r2"]
            entries = ticker_node.setdefault(method, [])
            if entries and entries[-1]["date"] == run_date:
                entries[-1] = {"date": run_date, "r2": r2}  # idempotent same-day rerun
            else:
                entries.append({"date": run_date, "r2": r2})
            if len(entries) > config.HISTORY_RETENTION_DAYS:
                ticker_node[method] = entries[-config.HISTORY_RETENTION_DAYS:]


def record_run_date(history: dict, run_date: str, data_latest_date: str) -> dict:
    """
    Track every calendar date trainer.py has run AND the latest date present
    in the underlying price data at that run, independent of any specific
    ticker/window/method.

    Returns {"is_new_data": bool} — False means the underlying master
    dataset's latest date hasn't advanced since the last run (e.g. the data
    pipeline hasn't refreshed yet, or trainer.py was re-triggered same-day).
    Caller (trainer.py) should skip calling record_day() entirely when
    is_new_data is False — recording a "day" against stale data would
    silently inflate persistence streaks with a duplicate observation
    instead of a genuine new one. This was found directly: a method with a
    fully deterministic fit (PCMCI) produced bit-for-bit identical OOS R²
    across two different calendar run dates — the only way that happens is
    if the input data was byte-identical, meaning no new trading day had
    actually landed in the master dataset between those two runs.
    """
    meta = history.setdefault("_meta", {"run_dates": [], "data_dates": []})
    run_dates = meta.setdefault("run_dates", [])
    data_dates = meta.setdefault("data_dates", [])

    last_data_date = data_dates[-1] if data_dates else None
    is_new_data = (last_data_date != data_latest_date)

    if not run_dates or run_dates[-1] != run_date:
        run_dates.append(run_date)
    if not data_dates or data_dates[-1] != data_latest_date:
        data_dates.append(data_latest_date)

    if len(run_dates) > config.HISTORY_RETENTION_DAYS:
        meta["run_dates"] = run_dates[-config.HISTORY_RETENTION_DAYS:]
    if len(data_dates) > config.HISTORY_RETENTION_DAYS:
        meta["data_dates"] = data_dates[-config.HISTORY_RETENTION_DAYS:]

    return {"is_new_data": is_new_data, "last_data_date": last_data_date}


def history_days_available(history: dict) -> int:
    """Counts genuinely distinct DATA days (data_dates), not calendar run
    days (run_dates) — a run against stale/unchanged underlying data
    doesn't count toward the persistence gate. See record_run_date's
    docstring for why this distinction matters."""
    return len(history.get("_meta", {}).get("data_dates", []))


def compute_streak(history: dict, universe: str, window: int, ticker: str,
                    method: str) -> dict:
    """
    Consecutive most-recent days (walking backward from today) with
    oos_r2 > 0. Because this walks backward from the latest entry,
    `qualifies=True` already implies today's own R² is positive.
    """
    entries = (
        history.get(universe, {})
        .get(str(window), {})
        .get(ticker, {})
        .get(method, [])
    )
    if not entries:
        return {"streak": 0, "days_tracked": 0, "qualifies": False, "latest_r2": None}

    streak = 0
    for e in reversed(entries):
        if e["r2"] > 0:
            streak += 1
        else:
            break

    return {
        "streak": streak,
        "days_tracked": len(entries),
        "qualifies": streak >= config.MIN_PERSISTENCE_DAYS,
        "latest_r2": entries[-1]["r2"],
    }


def save_history(history: dict, local_path: Path) -> None:
    with open(local_path, "w") as f:
        json.dump(history, f, indent=2)
