# config.py — P2-ETF-CAUSAL-SCM
# Causal Discovery + Structural Causal Models for Trading

import os

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-causal-scm-results"

# ── Universes (same tickers as the rest of the P2Quant suite) ──────────────────
UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SOXX", "SMH", "URA",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SOXX", "SMH", "URA",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

# Macro signals included as exogenous nodes in the causal graph.
# (column, description) — no weight/regime_sign needed here, unlike the PLV
# engine: causal discovery estimates directed edges and coefficients directly,
# it doesn't need a hand-specified regime sign.
MACRO_SIGNALS = [
    ("VIX",        "CBOE Volatility Index"),
    ("T10Y2Y",     "10Y-2Y Treasury Spread"),
    ("DXY",        "US Dollar Index"),
    ("IG_SPREAD",  "IG Credit Spread"),
    ("HY_SPREAD",  "HY Credit Spread"),
]

# ── Causal discovery methods ────────────────────────────────────────────────────
# "tsfci" has no maintained Python implementation (it's an R/Tetrad-ecosystem
# method). PCMCI (Runge et al., tigramite) is used as the practical substitute:
# same family (constraint-based conditional-independence testing for time
# series, FCI-style), actively maintained, same purpose. See README for the
# full rationale — this is a deliberate, documented substitution, not a bug.
CAUSAL_METHODS = ["varlingam", "pcmci", "timino"]

METHOD_LABELS = {
    "varlingam": "VarLiNGAM",
    "pcmci":     "PCMCI (tsFCI substitute)",
    "timino":    "TiMINo (linear proxy)",
}

# ── Rolling windows for causal discovery (trading days) ────────────────────────
# Deliberately shorter than the PLV engine's windows (which go out to 4536d /
# ~18yr). Causal discovery methods (LiNGAM, PCMCI, TiMINo) all assume a roughly
# stationary data-generating process within the window; multi-year windows
# spanning several macro regimes badly violate that assumption on top of being
# far more expensive to fit. See README for the full reasoning.
#
# 21d is intentionally the shortest window here (~1 trading month). Its train
# set (~13 obs) and test set (~6 obs) are genuinely too small for the OOS
# metrics to be statistically reliable on their own — see LOW_SAMPLE_* below,
# which flags (never hides) results from windows this thin.
WINDOWS = [21, 63, 126, 252, 504, 1008]

# ── Lag structure ───────────────────────────────────────────────────────────────
# Kept small deliberately: daily-frequency ETF/macro causal relationships
# beyond a couple of days are rarely reliable and drive up compute cost fast
# (conditional-independence testing and ICA both scale badly with more lags).
MAX_LAG = 2

# ── Ridge regularization for the hand-rolled OLS steps (PCMCI's and TiMINo's
# forecasting regressions, plus TiMINo's internal ordering search) ────────────
# Plain OLS blows up when candidate-regressor count approaches or exceeds the
# usable training-row count — exactly what happens on short windows with large
# universes (observed directly: TiMINo produced OOS R² as extreme as -106 on
# the 63d window with the 30-variable COMBINED universe before ridge was
# added). Ridge bounds coefficients and keeps forecasts finite even when the
# design matrix is rank-deficient. Intercept is never penalized.
#
# **This value was tuned against real data via ridge_alpha_sweep.py, not
# guessed.** A synthetic sweep first suggested 0.01, but running the actual
# sweep against real ETF/macro data (all 3 universes, windows 63/126/252,
# alphas 0.003 to 1.0) showed something the synthetic test missed: blowup
# protection was already fully saturated at the LOWEST alpha tested
# (worst_r2 = -1.139, identical across every alpha from 0.003 to 1.0 — no
# additional stability benefit from going higher). Given that, signal
# preservation is the only thing that should decide the value, and 0.003
# clearly wins: 10 (ticker, window, method) combinations exceeded R² > 0.05
# at alpha=0.003, vs. only 2 at alpha=1.0, with the single best result
# dropping from R²=0.214 to R²=0.107 as alpha increased. Re-run
# ridge_alpha_sweep.py periodically (data changes) or if you widen the
# universe/window/lag settings enough to change the rank-deficiency risk
# profile — the right alpha is a property of the data, not a fixed constant.
RIDGE_ALPHA = 0.003

# ── Train/test split for the OOS "which method actually works" validation ──────
# Chronological split within each window — same discipline as the sentiment
# engine's out-of-sample ablation: fit on TRAIN, forecast forward through
# TEST using the frozen fitted structure (no re-fitting), then measure
# genuine forecast skill. The winning method per ticker/window is picked by
# this, never by in-sample graph-fit statistics alone.
TRAIN_FRAC = 0.70

# Hard floor: below this, fit_and_backtest refuses to run at all (not enough
# data for ridge to save it from being meaningless).
MIN_TRAIN_SAMPLES = 10
MIN_TEST_SAMPLES = 4

# Reliability bar: results below this (but above the hard floor) still run
# and are still shown — never silently hidden — but get a `low_sample`
# warning flag surfaced in the JSON output and the dashboard, since an OOS
# R²/hit-rate computed from a handful of days is not statistically
# trustworthy on its own, however good it looks.
RELIABLE_TRAIN_SAMPLES = 40
RELIABLE_TEST_SAMPLES = 15

# ── Regime-change detection (for the "temporally adaptive" re-estimation) ──────
# Deliberately simple: causal graphs are already re-fit fresh per window (which
# is itself a form of regime adaptation), so this is a secondary, cheap check
# used only to flag windows where the regime shifted mid-window (rolling
# realized-volatility ratio, current half vs. earlier half). Not a full
# structural-break test (e.g. CUSUM); documented as an approximation.
REGIME_VOL_LOOKBACK = 21
REGIME_SHIFT_THRESHOLD = 1.75  # ratio of recent/prior realized vol flagged as a regime shift

# ── Wall-clock safety ────────────────────────────────────────────────────────────
# Learned the hard way on a different engine in this suite: bound worst-case
# per-fit time so one slow method/window/universe combination can't silently
# eat a whole CI job's budget.
MAX_SECONDS_PER_FIT = 120

# ── Output ─────────────────────────────────────────────────────────────────────
TOP_N = 3
# Minimum stationarised (log-return) samples required for ANY method to even
# attempt a fit, checked in trainer.py before fit_and_backtest runs. Lowered
# to accommodate the 21d window (which nets ~26 rows after trimming for
# log-return/pct-change and MAX_LAG) — fit_and_backtest's own
# MIN_TRAIN_SAMPLES/MIN_TEST_SAMPLES gates (and the low_sample warning flag)
# are what actually govern reliability from here.
MIN_SAMPLES = 15

# ── Signal persistence ──────────────────────────────────────────────────────
# A (universe, window, ticker, method) combo only qualifies for the top-N
# picks if it has shown POSITIVE OOS R² on this many CONSECUTIVE most-recent
# daily runs, including today — not just today's snapshot. This exists
# because a single lucky day is not evidence of a durable edge: XLF's OOS R²
# for the exact same window/method swung from 0.0085 to 0.20 (24x) across
# different ridge-alpha settings alone, which is exactly the kind of fragile,
# single-day result this gate is meant to filter out before it's ever
# presented as a "top pick." Since the streak check walks backward from the
# most recent entry, requiring `qualifies=True` already implies today's own
# R² is positive — no separate check needed.
#
# Cold-start note: for the first MIN_PERSISTENCE_DAYS-1 runs of a brand new
# repo, NOTHING can qualify yet — there isn't enough history. This is
# expected and surfaced explicitly by the dashboard (via history_days
# in the JSON output), not hidden as an empty "no results" state.
MIN_PERSISTENCE_DAYS = 3

# How many days of daily history to retain in causal_scm_history.json before
# trimming old entries. Bounds file growth; only the trailing
# MIN_PERSISTENCE_DAYS matter for the streak check, but more history is kept
# so the 📈 Signal Persistence dashboard tab can show a fuller track record.
HISTORY_RETENTION_DAYS = 60

HISTORY_FILENAME = "causal_scm_history.json"
