"""
live_tracker.py — Forward paper-trading tracker for persistence-qualified picks
====================================================================================

Answers a different question than the rest of this engine. Everything else
here measures BACKWARD-looking out-of-sample skill (would this have
predicted the recent past). This module asks: if you had actually acted on
today's persistence-qualified top picks, did they make money going
FORWARD? That's the question that actually matters for "should we trade
this," and it's the one thing the engine couldn't answer until now.

Mechanics
---------
Every day, for each universe's current top_etfs (already gated by
persistence + adequate sample size — see trainer.py), a paper position is
opened at that day's closing price if one isn't already open for that exact
(universe, ticker, window, method) combination. Each subsequent day:
  - the first new trading day's realized return after entry is recorded
    once available, checked against the model's predicted_direction, to
    compute a genuine forward next-day hit rate (not a backtest number)
  - the position's cumulative return-to-date is updated
  - if the combo drops out of today's top_etfs (loses its persistence
    streak, or its OOS R² decays), the position is closed and its final
    cumulative return frozen

Stale-data protection: identical principle to persistence.py's fix for the
same underlying issue — a position's cumulative return and next-day return
only advance when the underlying price series has a genuinely new latest
date, never on a re-run against unrefreshed data. This is checked directly
against the price data itself, not the calendar run date.
"""

import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

import config

LIVE_TRACKING_FILENAME = "causal_scm_live_tracking.json"


def load_tracking() -> dict:
    """Download existing tracking state from HF. Returns a fresh empty
    structure on first run — expected, not an error."""
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    try:
        path = hf_hub_download(
            repo_id=config.OUTPUT_REPO,
            filename=LIVE_TRACKING_FILENAME,
            repo_type="dataset",
            token=token or None,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"positions": []}


def _position_id(universe: str, ticker: str, window: int, method: str, entry_date: str) -> str:
    return f"{universe}|{ticker}|{window}|{method}|{entry_date}"


def update_tracking(tracking: dict, run_date: str, prices: pd.DataFrame,
                     tab1_universes: dict) -> dict:
    """
    prices: the same raw closing-price DataFrame trainer.py already loaded
    for the whole pipeline (DatetimeIndex, one column per ticker).
    tab1_universes: the fully-built Tab 1 payload (universe -> {"top_etfs": [...], ...}),
    already gated by persistence + low_sample exclusion — this function
    doesn't re-decide who qualifies, it just tracks forward outcomes for
    whoever already does.
    """
    positions = tracking.setdefault("positions", [])

    today_top = {}  # (universe, ticker, window, method) -> top_etfs entry
    for universe, udata in tab1_universes.items():
        for e in udata.get("top_etfs", []):
            key = (universe, e["ticker"], e["best_window"], e["best_method"])
            today_top[key] = e

    # ── Open new positions for combos that qualify today and aren't ────
    # already being tracked as an open position.
    for (universe, ticker, window, method), e in today_top.items():
        already_open = any(
            p["universe"] == universe and p["ticker"] == ticker and
            p["window"] == window and p["method"] == method and p["status"] == "open"
            for p in positions
        )
        if already_open or ticker not in prices.columns:
            continue

        series = prices[ticker].dropna()
        if series.empty:
            continue
        entry_price = float(series.iloc[-1])
        entry_date_actual = str(series.index[-1].date())

        positions.append({
            "id": _position_id(universe, ticker, window, method, entry_date_actual),
            "universe": universe, "ticker": ticker, "window": window, "method": method,
            "entry_date": entry_date_actual, "entry_price": entry_price,
            "predicted_direction": e["predicted_direction"],
            "status": "open",
            "next_day_return": None, "next_day_correct": None,
            "last_price_date": entry_date_actual, "last_price": entry_price,
            "cumulative_return": 0.0,
            "exit_date": None,
        })

    # ── Update every OPEN position's price-derived fields ───────────────
    # Closed positions are intentionally skipped here — their
    # cumulative_return must freeze at whatever it was on the exit day,
    # representing what you'd actually have realized by exiting when the
    # model said to. Previously this loop updated every position
    # regardless of status, so a "closed" position's return kept drifting
    # with the market indefinitely after exit — found directly: a position
    # closed at -2.9% showed -5.2% two updates later despite being
    # labeled closed the whole time, with no corresponding change to its
    # exit_date. That's not what "closed" is supposed to mean.
    for p in positions:
        if p["status"] != "open":
            continue
        if p["ticker"] not in prices.columns:
            continue
        series = prices[p["ticker"]].dropna()
        if series.empty:
            continue
        latest_date = str(series.index[-1].date())
        latest_price = float(series.iloc[-1])

        if latest_date == p["last_price_date"]:
            continue  # no genuinely new price data since last update — skip,
            # same stale-data discipline as persistence.py

        p["last_price_date"] = latest_date
        p["last_price"] = latest_price
        p["cumulative_return"] = (latest_price / p["entry_price"]) - 1.0

        if p["next_day_return"] is None and latest_date != p["entry_date"]:
            raw_ret = (latest_price / p["entry_price"]) - 1.0
            p["next_day_return"] = raw_ret
            predicted_up = (p["predicted_direction"] == "up")
            actual_up = raw_ret > 0
            p["next_day_correct"] = bool(predicted_up == actual_up)

        key = (p["universe"], p["ticker"], p["window"], p["method"])
        if p["status"] == "open" and key not in today_top:
            p["status"] = "closed"
            p["exit_date"] = run_date

    # ── Aggregate stats ──────────────────────────────────────────────────
    with_next_day = [p for p in positions if p["next_day_return"] is not None]
    closed = [p for p in positions if p["status"] == "closed"]

    tracking["aggregate"] = {
        "n_positions": len(positions),
        "n_open": sum(1 for p in positions if p["status"] == "open"),
        "n_closed": len(closed),
        "next_day_hit_rate": (
            sum(1 for p in with_next_day if p["next_day_correct"]) / len(with_next_day)
            if with_next_day else None
        ),
        "mean_next_day_return": (
            sum(p["next_day_return"] for p in with_next_day) / len(with_next_day)
            if with_next_day else None
        ),
        "mean_cumulative_return_closed": (
            sum(p["cumulative_return"] for p in closed) / len(closed)
            if closed else None
        ),
    }

    return tracking


def save_tracking(tracking: dict, local_path: Path) -> None:
    with open(local_path, "w") as f:
        json.dump(tracking, f, indent=2)
