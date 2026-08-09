# P2-ETF-CAUSAL-SCM

Causal Discovery + Structural Causal Models (SCM) for Trading.

Moves beyond pure correlation/predictive signals toward more durable alpha:
time-series causal discovery → directed causal graph → intervention-aware
one-step-ahead forecast, with the graph re-estimated fresh every rolling
window (a form of regime adaptation) and a genuine out-of-sample backtest
deciding which method to trust for each ticker — never an in-sample
graph-fit statistic.

## Methods used, honestly described

The original ask was VarLiNGAM, tsFCI, and TiMINo. Two of the three have no
maintained Python implementation, so this repo uses documented, principled
substitutes rather than silently reimplementing something different under
the same name:

| Method | Status | What's actually running |
|---|---|---|
| **VarLiNGAM** | Real, as specified | `lingam.VARLiNGAM` (Hyvärinen et al. 2010) — non-Gaussian instantaneous + lagged causal ordering |
| **tsFCI → PCMCI** | **Substituted** | tsFCI (Entner & Hoyer) lives in the R/Tetrad ecosystem with no maintained Python port. **PCMCI** (Runge et al. 2019, `tigramite`) is used instead: same family (constraint-based conditional-independence testing for time series), actively maintained, same purpose |
| **TiMINo** | **Simplified** | The original (Peters, Janzing, Schölkopf 2013) uses nonlinear regression + HSIC (kernel-based nonparametric independence testing) and also has no maintained Python package. This repo implements a **from-scratch linear proxy**: iterative causal-ordering search using Pearson correlation as the independence test instead of HSIC |

Every result surfaced by the dashboard is labeled with which method actually
produced it, and the substitutions above are stated in the dashboard's
warning banner, not just here.

## How the forecast works

Each method discovers a directed graph over ETF log-returns + macro
pct-changes (lags 1..`MAX_LAG`, contemporaneous edges shown for diagnostics
but never used for forecasting — a same-day cause isn't known yet at
forecast time). The graph's structural equations then produce tomorrow's
one-step-ahead forecast directly:

- **VarLiNGAM**: `x̂_t = (I - B0)⁻¹ Σ_k Bk @ x_{t-k}` — a genuine SCM forecast from the estimated structural equations.
- **PCMCI / TiMINo**: OLS regression on each variable's discovered lagged parents (hand-rolled, not the source library's own prediction machinery — kept uniform across all three methods so the OOS validation methodology is identical for all of them).

## Method selection — the actual point of this engine

For every ticker, in every universe/window, all three methods are backtested
on a chronological 70/30 train/test split: fit on TRAIN, walk the frozen
structure forward through TEST, no re-fitting. The **winning method per
ticker is whichever had the best genuine out-of-sample R²** on that split —
this selection is what the 🧪 Method Comparison tab shows in full, and it is
the *primary* output of this engine, not an ablation bolted on afterward.

OOS R² can come back negative — that means the model is worse than just
guessing the mean, and it's shown honestly rather than hidden. A ticker with
no method beating that bar is still shown, with a low/zero score, rather
than silently dropped.

## Repository structure

```
config.py            Universes, macro signals, methods, windows, HF repos
data_manager.py       HuggingFace loader — same data source, same safe
                       dropna pattern as the rest of the P2Quant suite
causal_scm.py          Core engine: 3 discovery methods, shared OLS
                       forecasting contract, regime-shift flag, OOS backtest
trainer.py             Orchestrator — loops universe × window, runs the OOS
                       backtest ONCE per (universe, window) [not per ticker
                       — a single graph fit scores every ticker at once],
                       picks winning method per ticker, writes 3 JSON files
push_results.py        Thin HfApi wrapper — uploads to OUTPUT_REPO
us_calendar.py         Next-trading-day helper
streamlit_app.py        3-tab dashboard
.github/workflows/daily.yml   Scheduled run (single job — see note below)
```

## Why one workflow job, not a CI matrix

A different engine in this suite (SAMBA) needed to split its training across
parallel CI jobs after genuinely overshooting GitHub Actions' free-tier
6-hour limit. Before assuming the same here, the actual runtime was
benchmarked: causal discovery is a single-shot statistical fit, not
iterative epoch-based neural network training — a fundamentally different
and much cheaper compute profile. The full 3-universe × 5-window × 3-method
run measured at roughly **2-3 minutes**. A single job with a 60-minute
timeout is generous headroom, not a tight fit — no matrix parallelization
needed. `MAX_SECONDS_PER_FIT` in `config.py` still bounds worst-case
per-fit time as defense-in-depth, same discipline as the rest of the suite.

## Why windows are shorter here than other engines

`WINDOWS = [63, 126, 252, 504, 1008]` — capped at ~4 years, versus up to
4536 days (~18 years) elsewhere in this suite. This is a deliberate,
principled choice, not just a compute optimization: every method here
assumes a roughly stationary data-generating process within the window.
Multi-year windows spanning several macro regimes badly violate that
assumption on top of being more expensive to fit.

## Data source (same as the rest of the suite)

`P2SAMAPA/fi-etf-macro-signal-master-data` — same master parquet, same
ticker universes, same macro signal set (VIX, T10Y2Y, DXY core; IG/HY spread
extended), same ffill/dropna conventions as `data_manager.py` in the other
P2Quant repos.
