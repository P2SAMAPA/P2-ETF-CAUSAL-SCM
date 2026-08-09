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
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SOXX", "SMH", "URA",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
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
WINDOWS = [63, 126, 252, 504, 1008]

# ── Lag structure ───────────────────────────────────────────────────────────────
# Kept small deliberately: daily-frequency ETF/macro causal relationships
# beyond a couple of days are rarely reliable and drive up compute cost fast
# (conditional-independence testing and ICA both scale badly with more lags).
MAX_LAG = 2

# ── Train/test split for the OOS "which method actually works" validation ──────
# Chronological split within each window — same discipline as the sentiment
# engine's out-of-sample ablation: fit on TRAIN, forecast forward through
# TEST using the frozen fitted structure (no re-fitting on test data), then
# measure genuine forecast skill. The winning method per ticker/window is
# picked by this, never by in-sample graph-fit statistics alone.
TRAIN_FRAC = 0.70
MIN_TRAIN_SAMPLES = 40
MIN_TEST_SAMPLES  = 10

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
MIN_SAMPLES = 40  # minimum non-NaN samples required for any method to attempt a fit
