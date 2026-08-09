"""
ridge_alpha_sweep.py — Validate config.RIDGE_ALPHA against real data
========================================================================

The default (0.01) was chosen from a synthetic sweep — see the comment
above config.RIDGE_ALPHA. Synthetic tests validate the general principle
(ridge should be strong enough to bound coefficients on rank-deficient
design matrices, but not so strong it dominates the natural scale of real
returns data), not necessarily the exact optimal value for THIS specific
data-generating process. This script re-runs that sweep against your actual
HuggingFace data and reports, per candidate alpha, the same summary
statistics used throughout this engine's development: how many
(universe, window, ticker, method) combinations show genuine out-of-sample
skill, and how bad the worst-case blowup is.

Usage:
    python ridge_alpha_sweep.py
    python ridge_alpha_sweep.py --alphas 0.001,0.003,0.01,0.03,0.1,1.0
    python ridge_alpha_sweep.py --windows 63,126,252,504   # skip 21d/1008d for speed

Runtime note: this multiplies the normal trainer.py runtime (~3-6 min) by
the number of alpha candidates tested. The default 5 candidates × full
scope takes roughly 15-30 min — still comfortably within a single CI job's
budget, but consider narrowing --windows for a faster iteration loop.
"""

import argparse
import os
import numpy as np

import config
import data_manager
import causal_scm as cs


def summarize(all_r2: list, all_low_sample: list) -> dict:
    r2 = np.array(all_r2)
    low = np.array(all_low_sample)
    reliable = r2[~low] if (~low).any() else r2
    return {
        "n": len(r2),
        "n_positive": int((r2 > 0).sum()),
        "n_above_p05": int((r2 > 0.05).sum()),
        "median_r2": float(np.median(r2)),
        "worst_r2": float(r2.min()),
        "best_r2": float(r2.max()),
        "median_r2_reliable_only": float(np.median(reliable)) if len(reliable) else float("nan"),
        "best_r2_reliable_only": float(reliable.max()) if len(reliable) else float("nan"),
    }


def run_sweep(alphas: list, windows: list, universes: dict):
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    prices, macro = data_manager.load_master_data(hf_token=token)
    data_manager.validate_data(prices, macro)

    results_by_alpha = {}

    for alpha in alphas:
        config.RIDGE_ALPHA = alpha
        print(f"\n{'='*70}")
        print(f"alpha = {alpha}")
        print(f"{'='*70}")

        all_r2, all_low_sample = [], []

        for universe_name, tickers in universes.items():
            for window in windows:
                p_win = prices.tail(window + config.MAX_LAG + 5)
                m_win = macro.tail(window + config.MAX_LAG + 5)
                X, var_names, _ = cs.build_stationary_matrix(p_win, m_win, tickers)
                if X is None or len(X) < config.MIN_SAMPLES:
                    continue

                for method in config.CAUSAL_METHODS:
                    bt = cs.fit_and_backtest(X, var_names, method, config.MAX_LAG)
                    if bt is None:
                        continue
                    for pv in bt["per_var"]:
                        all_r2.append(pv["oos_r2"])
                        all_low_sample.append(bt["low_sample"])

            print(f"  {universe_name}: done ({len(all_r2)} combos so far)")

        summary = summarize(all_r2, all_low_sample)
        results_by_alpha[alpha] = summary
        print(f"\n  n={summary['n']}  positive={summary['n_positive']}  "
              f">0.05={summary['n_above_p05']}  "
              f"median={summary['median_r2']:+.4f}  "
              f"worst={summary['worst_r2']:+.3f}  best={summary['best_r2']:+.4f}")
        print(f"  (reliable/non-low-sample only) median={summary['median_r2_reliable_only']:+.4f}  "
              f"best={summary['best_r2_reliable_only']:+.4f}")

    print(f"\n\n{'='*70}")
    print("SUMMARY — pick the alpha with the best combination of:")
    print("  - worst_r2 not catastrophically negative (blowup protection)")
    print("  - best n_above_0.05 / best 'best_r2_reliable_only' (signal preserved)")
    print(f"{'='*70}")
    header = f"{'alpha':>10s}  {'n>0.05':>7s}  {'median':>8s}  {'worst':>10s}  {'best(reliable)':>15s}"
    print(header)
    for alpha, s in results_by_alpha.items():
        print(f"{alpha:10.4f}  {s['n_above_p05']:7d}  {s['median_r2']:+8.4f}  "
              f"{s['worst_r2']:+10.3f}  {s['best_r2_reliable_only']:+15.4f}")

    return results_by_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", type=str, default="0.003,0.01,0.03,0.1,1.0",
                         help="Comma-separated candidate alpha values")
    parser.add_argument("--windows", type=str, default=None,
                         help="Comma-separated windows to test (default: config.WINDOWS)")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    windows = [int(w) for w in args.windows.split(",")] if args.windows else config.WINDOWS

    run_sweep(alphas, windows, config.UNIVERSES)
