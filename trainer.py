"""
trainer.py  —  Causal SCM Engine orchestrator
================================================

For every universe × window:
  1. Build the stationarised (log-return / pct-change) matrix.
  2. Run the OOS backtest (chronological train/test split) for all three
     causal discovery methods — VarLiNGAM, PCMCI (tsFCI substitute), TiMINo
     — ONCE per (universe, window). A single causal graph fit scores every
     ticker in the universe simultaneously, so this is NOT repeated per
     ticker (see causal_scm.py's fit_and_backtest docstring).
  3. Per ticker, pick the WINNING method = whichever of the three has the
     best genuine out-of-sample R² for that ticker — never picked by
     in-sample graph statistics. This is the "optimise to pick which
     method actually works" step the whole engine exists to do.
  4. Refit the winning method(s) on the FULL window (live fit) and take that
     method's one-step-ahead forecast for the ticker as the live signal.
  5. Per ticker, "best window" = whichever window had the best OOS R² for
     that ticker's winning method — a genuine backtested-skill pick, not an
     in-sample one.
  6. Record today's OOS R² for every (universe, window, ticker, method) into
     a rolling history file — but ONLY if the underlying master data's
     latest date has genuinely advanced since the last run (a run against
     stale/unrefreshed data is skipped for history purposes, logged as a
     warning, so a duplicate-data re-run can't silently inflate a streak).
     Then gate the top-N picks (Tabs 1 & 2) to only combos with a proven
     streak — positive OOS R² on config.MIN_PERSISTENCE_DAYS CONSECUTIVE
     genuinely-new-data days, including today. A single lucky day is not
     enough on its own, and neither is the same day's data scored twice
     under different calendar labels.
  7. Build four JSON result files (+ the history file) and upload them.

JSON schema — Tab 4 (causal_scm_persistence_YYYY-MM-DD.json) — track record
--------------------------------------------------------------------------------
{
  "run_date": "YYYY-MM-DD", "history_days": 12, "min_persistence_days": 3,
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63": {
          "TLT": {
            "varlingam": {"streak": 2, "days_tracked": 8, "qualifies": false, "latest_r2": 0.03},
            "pcmci":     {...},
            "timino":    {...}
          },
          ...
        }
      }
    }
  }
}

JSON schema — Tab 1 (causal_scm_YYYY-MM-DD.json) — best window+method per ETF
--------------------------------------------------------------------------------
{
  "run_date": "YYYY-MM-DD",
  "universes": {
    "FI_COMMODITIES": {
      "top_etfs": [
        {"ticker": "TLT", "causal_score": 0.42, "best_window": 252,
         "best_method": "pcmci", "oos_r2": 0.18, "oos_correlation": 0.51,
         "oos_hit_rate": 0.61},
        ...
      ],
      "full_scores": { "TLT": {...same fields...}, ... }
    },
    ...
  }
}

JSON schema — Tab 2 (causal_scm_windows_YYYY-MM-DD.json) — explore by window
--------------------------------------------------------------------------------
{
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63":  {"top_etfs": [...], "full_ranking": [[ticker, score, method], ...]},
        ...
      }
    }
  }
}

JSON schema — Tab 3 (causal_scm_methods_YYYY-MM-DD.json) — method comparison
--------------------------------------------------------------------------------
{
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63": {
          "TLT": {
            "varlingam": {"oos_r2":.., "oos_correlation":.., "oos_hit_rate":..},
            "pcmci":     {...},
            "timino":    {...}
          },
          ...
        }
      }
    }
  }
}
"""

import json
import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_manager
import push_results
import persistence
import causal_scm as cs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _safe_float(val) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else 0.0
    except Exception:
        return 0.0


def process_universe_window(prices: pd.DataFrame, macro: pd.DataFrame,
                             tickers: list, window: int) -> dict:
    """
    Run all 3 methods' OOS backtest + live forecast for one universe/window.
    Returns {ticker: {method: {...oos metrics..., "live_forecast": float}}}
    or None if there wasn't enough data in this window.
    """
    # Trim to the trailing `window` days before stationarising, so the
    # window setting means what it says (days of raw history considered),
    # consistent with the rest of the P2Quant suite.
    p_win = prices.tail(window + config.MAX_LAG + 5)
    m_win = macro.tail(window + config.MAX_LAG + 5)

    X, var_names, dates = cs.build_stationary_matrix(p_win, m_win, tickers)
    if X is None or len(X) < config.MIN_SAMPLES:
        return None

    regime = cs.detect_regime_shift(X)

    ticker_positions = {t: var_names.index(t) for t in tickers if t in var_names}
    if not ticker_positions:
        return None

    per_method_results = {}
    for method in config.CAUSAL_METHODS:
        bt = cs.fit_and_backtest(X, var_names, method, config.MAX_LAG)
        if bt is None:
            continue
        live = cs.fit_live_and_forecast(X, var_names, method, config.MAX_LAG)
        per_method_results[method] = {"backtest": bt, "live": live}

    if not per_method_results:
        return None

    out = {}
    for ticker, idx in ticker_positions.items():
        out[ticker] = {"regime_shift": regime}
        for method, res in per_method_results.items():
            oos = res["backtest"]["per_var"][idx]
            live_val = None
            if res["live"] is not None:
                live_val = _safe_float(res["live"]["forecast"][idx])
            out[ticker][method] = {
                "oos_r2": _safe_float(oos["oos_r2"]),
                "oos_correlation": _safe_float(oos["oos_correlation"]),
                "oos_hit_rate": _safe_float(oos["oos_hit_rate"]),
                "n_train": res["backtest"]["n_train"],
                "n_test": res["backtest"]["n_test"],
                "low_sample": res["backtest"]["low_sample"],
                "live_forecast": live_val,
            }
    return out


def main():
    run_date = date.today().isoformat()
    logger.info(f"=== Causal SCM Engine  |  {run_date} ===")

    token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    prices, macro = data_manager.load_master_data(hf_token=token)
    data_manager.validate_data(prices, macro)

    data_latest_date = str(prices.index[-1].date())

    history = persistence.load_history()
    run_info = persistence.record_run_date(history, run_date, data_latest_date)
    is_new_data = run_info["is_new_data"]

    if not is_new_data:
        logger.warning(
            f"⚠️  Underlying master data's latest date ({data_latest_date}) hasn't "
            f"advanced since the last run ({run_info['last_data_date']}) — the data "
            "pipeline likely hasn't refreshed yet, or this run was re-triggered "
            "same-day. Persistence history will NOT be updated this run, to avoid "
            "inflating streaks with a duplicate observation on data already scored. "
            "Today's snapshot (Tabs 1-3) still reflects this run normally."
        )

    logger.info(f"Loaded persistence history: {persistence.history_days_available(history)} "
                f"genuine data day(s) on record")

    # results[universe][window][ticker][method] = {...}
    results: dict = {}

    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"Universe: {universe_name}  ({len(tickers)} tickers)")
        results[universe_name] = {}

        for window in config.WINDOWS:
            logger.info(f"  window={window}d …")
            win_result = process_universe_window(prices, macro, tickers, window)
            results[universe_name][window] = win_result or {}

            if win_result:
                logger.info(f"    → {len(win_result)} tickers scored")
                if is_new_data:
                    persistence.record_day(history, run_date, universe_name, window, win_result)
            else:
                logger.warning(f"    → no results for {universe_name} w={window}")

    history_days = persistence.history_days_available(history)

    # ── Cross-sectional z-scoring of live forecasts, per universe per window,
    #    per method — so scores are comparable within a snapshot, same
    #    convention as the rest of the suite. ──────────────────────────────
    def _zscore_method_forecasts(win_result: dict, method: str) -> dict:
        vals = {t: r[method]["live_forecast"] for t, r in win_result.items()
                if method in r and r[method]["live_forecast"] is not None}
        if len(vals) < 2:
            return {t: 0.0 for t in vals}
        arr = np.array(list(vals.values()))
        mu, sd = arr.mean(), arr.std()
        if sd < 1e-12:
            return {t: 0.0 for t in vals}
        return {t: float((v - mu) / sd) for t, v in vals.items()}

    # ── Tab 1 + Tab 2 + Tab 3 + Tab 4 payload construction ─────────────────
    tab1_universes, tab2_universes, tab3_universes, tab4_universes = {}, {}, {}, {}

    for universe_name, tickers in config.UNIVERSES.items():
        # best (window, method) per ticker by OOS R² — genuine backtested skill
        best = {}  # ticker -> {window, method, oos_r2, oos_correlation, oos_hit_rate, score, streak, days_tracked, qualifies}

        windows_out = {}
        methods_out = {}
        persistence_out = {}

        for window in config.WINDOWS:
            win_result = results[universe_name].get(window, {})
            if not win_result:
                windows_out[str(window)] = {"top_etfs": [], "full_ranking": []}
                methods_out[str(window)] = {}
                persistence_out[str(window)] = {}
                continue

            methods_out[str(window)] = {
                t: {m: {k: v for k, v in r[m].items() if k != "live_forecast"}
                    for m in config.CAUSAL_METHODS if m in r}
                for t, r in win_result.items()
            }

            # Persistence table for this window: every (ticker, method) that
            # was scored today, with its streak/days_tracked/qualifies —
            # this IS Tab 4's content, and it's built here (not filtered)
            # so the dashboard can show the full track record, not just
            # whichever combo happened to win today.
            persistence_out[str(window)] = {
                t: {
                    m: persistence.compute_streak(history, universe_name, window, t, m)
                    for m in config.CAUSAL_METHODS if m in r
                }
                for t, r in win_result.items()
            }

            zscores_by_method = {
                m: _zscore_method_forecasts(win_result, m) for m in config.CAUSAL_METHODS
            }

            window_scores = {}  # ticker -> {score, method, oos_r2, oos_corr, oos_hit, low_sample, streak, days_tracked, qualifies}
            for ticker, r in win_result.items():
                candidates = []
                for m in config.CAUSAL_METHODS:
                    if m not in r or r[m]["live_forecast"] is None:
                        continue
                    z = zscores_by_method[m].get(ticker, 0.0)
                    candidates.append((r[m]["oos_r2"], m, z, r[m]["oos_correlation"],
                                        r[m]["oos_hit_rate"], r[m]["low_sample"]))
                if not candidates:
                    continue
                # winning method for THIS ticker at THIS window = best OOS R²
                # (low_sample is surfaced, never used to silently reorder this —
                # see README for why: transparency over a heuristic tie-break)
                candidates.sort(key=lambda c: c[0], reverse=True)
                oos_r2, method, z, oos_corr, oos_hit, low_sample = candidates[0]
                streak_info = persistence.compute_streak(history, universe_name, window, ticker, method)
                window_scores[ticker] = {
                    "score": z, "method": method, "oos_r2": oos_r2,
                    "oos_correlation": oos_corr, "oos_hit_rate": oos_hit,
                    "low_sample": low_sample,
                    "streak": streak_info["streak"], "days_tracked": streak_info["days_tracked"],
                    "qualifies": streak_info["qualifies"],
                }

                # track global best window for this ticker (by raw OOS R² —
                # the persistence gate is applied afterward, when building
                # top_etfs below, same 2-stage pattern as the low_sample flag)
                prev = best.get(ticker)
                if prev is None or oos_r2 > prev["oos_r2"]:
                    best[ticker] = {
                        "window": window, "method": method, "score": z,
                        "oos_r2": oos_r2, "oos_correlation": oos_corr,
                        "oos_hit_rate": oos_hit, "low_sample": low_sample,
                        "streak": streak_info["streak"], "days_tracked": streak_info["days_tracked"],
                        "qualifies": streak_info["qualifies"],
                    }

            ranked = sorted(window_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
            # Top-N picks require PERSISTENCE, not just today's oos_r2 > 0:
            # a combo only qualifies if it's shown positive OOS R² on
            # config.MIN_PERSISTENCE_DAYS CONSECUTIVE most-recent days,
            # including today (which already implies today's R² > 0 — see
            # persistence.compute_streak). This can legitimately leave fewer
            # than TOP_N entries, or none — especially in the first
            # MIN_PERSISTENCE_DAYS-1 runs of a new repo, before enough
            # history has accumulated (see history_days below, surfaced to
            # the dashboard so this reads as "still building history," not
            # a silent bug). full_ranking stays completely unfiltered — every
            # ticker, negative R² and non-persistent combos included — as
            # the transparency view, and stays sorted by causal_score (the
            # raw signal ranking).
            #
            # Among QUALIFIED combos, top_etfs is sorted by OOS R² — NOT by
            # causal_score. causal_score is a z-scored predicted-MOVE
            # magnitude; it says nothing about how much to trust the
            # prediction. Sorting the headline picks by magnitude let a
            # barely-qualifying combo (R²=0.008, large predicted move)
            # outrank the most reliable combo found anywhere in this
            # engine's testing (IWD: R²=0.11, 71% hit rate, streak, but a
            # modest predicted move) — backwards from the entire point of
            # the persistence gate. R² measures reliability directly;
            # that's what should decide which 3 cards get shown.
            #
            # low_sample combos are EXCLUDED from top-3 entirely, even if
            # qualifies=True. Observed directly: the 21d window (~8 OOS
            # days) produced a streak with R² climbing to 0.60 — a level of
            # apparent skill that isn't plausible for a real financial
            # forecast, and is far more consistent with small-sample
            # regression instability than a genuine signal maturing. A
            # streak built on an estimate we already know is statistically
            # unreliable (that's what low_sample means) isn't made
            # trustworthy by being consistently unreliable in the same
            # direction for a few days — the gate needs both persistence
            # AND adequate sample size, not persistence alone.
            qualified = [(t, v) for t, v in ranked if v["qualifies"] and not v["low_sample"]]
            ranked_qualified = sorted(qualified, key=lambda kv: kv[1]["oos_r2"], reverse=True)
            top_etfs = [
                {"ticker": t, "causal_score": _safe_float(v["score"]),
                 "method": v["method"], "oos_r2": _safe_float(v["oos_r2"]),
                 "low_sample": v["low_sample"], "streak": v["streak"],
                 "days_tracked": v["days_tracked"]}
                for t, v in ranked_qualified[:config.TOP_N]
            ]
            full_ranking = [[t, _safe_float(v["score"]), v["method"], v["low_sample"],
                              v["streak"], v["qualifies"]]
                             for t, v in ranked]
            windows_out[str(window)] = {"top_etfs": top_etfs, "full_ranking": full_ranking}

        tab2_universes[universe_name] = {"windows": windows_out}
        tab3_universes[universe_name] = {"windows": methods_out}
        tab4_universes[universe_name] = {"windows": persistence_out}

        if not best:
            tab1_universes[universe_name] = {"top_etfs": [], "full_scores": {}}
            continue

        ranked_best = sorted(best.items(), key=lambda kv: kv[1]["score"], reverse=True)
        # Same fix as Tab 2: among combos that pass the persistence gate,
        # rank by OOS R² (trustworthiness) rather than causal_score
        # (predicted-move magnitude), and exclude low_sample combos
        # entirely — a streak on a statistically unreliable estimate isn't
        # trustworthy no matter how many days long. full_scores below
        # stays unfiltered and sorted by causal_score — the transparency
        # view.
        qualified_best = [(t, v) for t, v in ranked_best if v["qualifies"] and not v["low_sample"]]
        ranked_best_qualified = sorted(qualified_best, key=lambda kv: kv[1]["oos_r2"], reverse=True)
        top_etfs = [
            {
                "ticker": t, "causal_score": _safe_float(v["score"]),
                "best_window": v["window"], "best_method": v["method"],
                "oos_r2": _safe_float(v["oos_r2"]),
                "oos_correlation": _safe_float(v["oos_correlation"]),
                "oos_hit_rate": _safe_float(v["oos_hit_rate"]),
                "low_sample": v["low_sample"],
                "streak": v["streak"], "days_tracked": v["days_tracked"],
            }
            for t, v in ranked_best_qualified[:config.TOP_N]
        ]
        full_scores = {
            t: {
                "causal_score": _safe_float(v["score"]),
                "best_window": v["window"], "best_method": v["method"],
                "oos_r2": _safe_float(v["oos_r2"]),
                "oos_correlation": _safe_float(v["oos_correlation"]),
                "oos_hit_rate": _safe_float(v["oos_hit_rate"]),
                "low_sample": v["low_sample"],
                "streak": v["streak"], "days_tracked": v["days_tracked"],
                "qualifies": v["qualifies"],
            }
            for t, v in ranked_best
        }
        tab1_universes[universe_name] = {"top_etfs": top_etfs, "full_scores": full_scores}

        logger.info(f"  {universe_name} top {config.TOP_N} (persistence-qualified): "
                    f"{[e['ticker'] for e in top_etfs]}")

    tab1_payload = {"run_date": run_date, "history_days": history_days, "universes": tab1_universes}
    tab2_payload = {"run_date": run_date, "history_days": history_days, "universes": tab2_universes}
    tab3_payload = {"run_date": run_date, "universes": tab3_universes}
    tab4_payload = {
        "run_date": run_date, "history_days": history_days,
        "min_persistence_days": config.MIN_PERSISTENCE_DAYS,
        "universes": tab4_universes,
    }

    tab1_path = Path(f"causal_scm_{run_date}.json")
    tab2_path = Path(f"causal_scm_windows_{run_date}.json")
    tab3_path = Path(f"causal_scm_methods_{run_date}.json")
    tab4_path = Path(f"causal_scm_persistence_{run_date}.json")
    history_path = Path(config.HISTORY_FILENAME)

    for path, payload in [(tab1_path, tab1_payload), (tab2_path, tab2_payload),
                           (tab3_path, tab3_payload), (tab4_path, tab4_payload)]:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote {path}")

    persistence.save_history(history, history_path)
    logger.info(f"Wrote {history_path} ({history_days} day(s) of history)")

    push_results.push_daily_result(tab1_path)
    push_results.push_daily_result(tab2_path)
    push_results.push_daily_result(tab3_path)
    push_results.push_daily_result(tab4_path)
    push_results.push_daily_result(history_path)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
