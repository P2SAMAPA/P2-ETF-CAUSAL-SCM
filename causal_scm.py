"""
causal_scm.py — Causal Discovery + Structural Causal Models for Trading
=========================================================================

Three time-series causal discovery methods, one shared forecasting contract,
and an honest out-of-sample backtest to pick which method (if any) actually
has forecast skill for a given ticker/window — rather than trusting any
method's in-sample graph-fit statistics, which the rest of this suite has
repeatedly found to be a poor guide to real predictive value.

Methods
-------
1. VarLiNGAM  (Hyvärinen et al. 2010, via the `lingam` package)
   Non-Gaussian instantaneous + lagged causal ordering. Structural equations:
       x_t = B0 @ x_t + Σ_k Bk @ x_{t-k} + e_t
   Forecast (set e_t → 0, its expectation):
       x̂_t = (I - B0)^{-1} Σ_k Bk @ x_{t-k}
   This is a genuine SCM forecast — it uses the estimated causal structural
   equations, not just a fitted regression.

2. PCMCI  — used as the practical substitute for tsFCI.
   **Honest substitution note**: tsFCI (Entner & Hoyer's time-series FCI) has
   no maintained Python implementation — it lives in the R/Tetrad ecosystem.
   PCMCI (Runge et al. 2019, via `tigramite`) is the closest actively
   maintained equivalent: constraint-based conditional-independence testing
   for time series, same purpose (recover a causal graph, robust to some
   spurious correlations FCI-style methods target). PCMCI is used purely for
   *graph discovery* here; forecasting uses a hand-rolled OLS regression on
   the discovered lagged parents (tigramite's own `Prediction` class was
   deliberately not used, to keep the forecasting methodology — and its OOS
   validation — identical across all three methods).

3. TiMINo — **simplified linear proxy**, not the full method.
   The original TiMINo (Peters, Janzing, Schölkopf 2013) uses nonlinear
   regression + HSIC (a kernel-based nonparametric independence test) to find
   a causal ordering, and has no maintained Python implementation either.
   This is a from-scratch linear approximation of the same idea: iteratively
   identify the variable whose residual (after regressing on its own lags)
   is *least correlated* with other variables' lagged values — i.e. the most
   "exogenous" / source-like variable — assign it earliest in the causal
   order, then repeat on the remainder. Correlation is a linear proxy for
   independence; the real TiMINo tests full nonparametric independence via
   HSIC. This trade-off is deliberate: full HSIC testing does not fit this
   suite's GitHub Actions CPU budget across 3 universes × 5 windows × 3
   methods run daily.

Forecasting contract (shared across all 3 methods)
----------------------------------------------------
Every fit_* function returns a dict:
    {
        "method": str,
        "var_names": [...],
        "parents": {var_idx: [(parent_idx, lag), ...]},   # lag >= 1 only —
            # contemporaneous (lag 0) edges may appear in the *graph* for
            # diagnostics but are never used for forecasting, since a
            # same-day cause isn't known yet at forecast time.
        "forecast_fn": callable(X_recent) -> np.ndarray of length n_vars,
    }
X_recent is the trailing MAX_LAG rows of the (already-stationarised) data,
most recent last.
"""

import time
import warnings
import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")


class FitTimeout(Exception):
    pass


def _check_deadline(t0: float, budget: float, where: str):
    if time.time() - t0 > budget:
        raise FitTimeout(f"{where} exceeded {budget}s budget")


# ── Data prep ────────────────────────────────────────────────────────────────

def build_stationary_matrix(prices: pd.DataFrame, macro: pd.DataFrame,
                             tickers: list) -> tuple:
    """
    Build the stationarised (log-return / pct-change) matrix used by every
    causal discovery method. Same transforms as the Hilbert PLV engine:
    ETF prices → log returns, macro levels → pct_change.

    Returns (X, var_names, dates) where X is (T, n_vars) float64, NaN-free.
    """
    avail_tickers = [t for t in tickers if t in prices.columns]
    if not avail_tickers:
        return None, [], None

    log_ret = np.log(prices[avail_tickers] / prices[avail_tickers].shift(1))

    macro_cols = [c for c, _ in config.MACRO_SIGNALS if c in macro.columns]
    macro_chg = macro[macro_cols].pct_change(fill_method=None) if macro_cols else pd.DataFrame(index=prices.index)

    combined = pd.concat([log_ret, macro_chg], axis=1).dropna(how="any")
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    var_names = avail_tickers + macro_cols
    return combined.values.astype(np.float64), var_names, combined.index


# ── Regime-shift detection ──────────────────────────────────────────────────

def detect_regime_shift(X: np.ndarray, lookback: int = None,
                         threshold: float = None) -> dict:
    """
    Cheap regime-shift flag: ratio of realized volatility in the most recent
    `lookback` days vs. the `lookback` days before that, averaged across all
    variables. NOT a formal structural-break test (e.g. CUSUM) — see the
    caveat in config.py. Used only to annotate results, not to gate them.
    """
    lookback = lookback or config.REGIME_VOL_LOOKBACK
    threshold = threshold or config.REGIME_SHIFT_THRESHOLD

    if len(X) < lookback * 2:
        return {"regime_shift": False, "vol_ratio": 1.0}

    recent = X[-lookback:]
    prior = X[-lookback * 2:-lookback]

    recent_vol = np.std(recent, axis=0).mean()
    prior_vol = np.std(prior, axis=0).mean() + 1e-10

    ratio = float(recent_vol / prior_vol)
    return {
        "regime_shift": bool(ratio > threshold or ratio < 1.0 / threshold),
        "vol_ratio": round(ratio, 3),
    }


# ── Shared OLS forecasting helper (used by PCMCI and TiMINo) ────────────────

def _fit_ols_forecast(X: np.ndarray, parents: dict, max_lag: int):
    """
    Given a discovered parent set {var_idx: [(parent_idx, lag), ...]} (lag>=1
    only), fit one OLS regression per variable on its parents' lagged values,
    and return a forecast_fn(X_recent) -> predicted next-step values for all
    variables. Variables with no parents fall back to predicting their
    training-sample mean (honest baseline, not zero).
    """
    n_vars = X.shape[1]
    T = X.shape[0]
    coefs = {}       # var_idx -> (intercept, [coef per parent in order])
    fallback_mean = X.mean(axis=0)

    for j in range(n_vars):
        plist = parents.get(j, [])
        if not plist:
            coefs[j] = (float(fallback_mean[j]), [])
            continue

        rows = []
        targets = []
        for t in range(max_lag, T):
            row = [X[t - lag, p] for p, lag in plist]
            rows.append(row)
            targets.append(X[t, j])
        Xd = np.column_stack([np.ones(len(rows)), np.array(rows)])
        yd = np.array(targets)
        try:
            beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        except Exception:
            coefs[j] = (float(fallback_mean[j]), [])
            continue
        coefs[j] = (float(beta[0]), beta[1:].tolist())

    def forecast_fn(X_recent: np.ndarray) -> np.ndarray:
        # X_recent: last max_lag rows, most recent last (index -1 = t-1)
        out = np.zeros(n_vars)
        for j in range(n_vars):
            intercept, betas = coefs[j]
            plist = parents.get(j, [])
            if not plist:
                out[j] = intercept
                continue
            val = intercept
            for (p, lag), b in zip(plist, betas):
                val += b * X_recent[-lag, p]
            out[j] = val
        return out

    return forecast_fn, coefs


# ── Method 1: VarLiNGAM ─────────────────────────────────────────────────────

def fit_varlingam(X: np.ndarray, var_names: list, max_lag: int,
                   t0: float, budget: float) -> dict:
    import lingam

    _check_deadline(t0, budget, "varlingam")
    model = lingam.VARLiNGAM(lags=max_lag, criterion=None)
    model.fit(X)
    _check_deadline(t0, budget, "varlingam (post-fit)")

    B_mats = model.adjacency_matrices_  # [B0, B1, ..., B_maxlag]
    n_vars = X.shape[1]

    try:
        I_minus_B0_inv = np.linalg.inv(np.eye(n_vars) - B_mats[0])
    except np.linalg.LinAlgError:
        I_minus_B0_inv = np.eye(n_vars)  # degenerate fallback: ignore contemporaneous effects

    # Build parents dict (lag>=1 only) for diagnostics/graph display
    parents = {j: [] for j in range(n_vars)}
    for lag in range(1, len(B_mats)):
        Bk = B_mats[lag]
        for i in range(n_vars):      # effect
            for k in range(n_vars):  # cause
                if abs(Bk[i, k]) > 1e-8:
                    parents[i].append((k, lag))

    def forecast_fn(X_recent: np.ndarray) -> np.ndarray:
        rhs = np.zeros(n_vars)
        for lag in range(1, len(B_mats)):
            rhs += B_mats[lag] @ X_recent[-lag]
        return I_minus_B0_inv @ rhs

    return {
        "method": "varlingam",
        "var_names": var_names,
        "parents": parents,
        "forecast_fn": forecast_fn,
    }


# ── Method 2: PCMCI (tsFCI substitute) ──────────────────────────────────────

def fit_pcmci(X: np.ndarray, var_names: list, max_lag: int,
              t0: float, budget: float) -> dict:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    _check_deadline(t0, budget, "pcmci")
    dataframe = pp.DataFrame(X, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=0.05)
    _check_deadline(t0, budget, "pcmci (post-fit)")

    graph = results["graph"]  # (n_vars, n_vars, tau_max+1), graph[i,j,tau] == '-->' means i -tau-> j
    n_vars = X.shape[1]

    parents = {j: [] for j in range(n_vars)}
    for tau in range(1, max_lag + 1):     # lag>=1 only, for forecasting
        for i in range(n_vars):
            for j in range(n_vars):
                if graph[i, j, tau] == "-->":
                    parents[j].append((i, tau))

    forecast_fn, _ = _fit_ols_forecast(X, parents, max_lag)

    return {
        "method": "pcmci",
        "var_names": var_names,
        "parents": parents,
        "forecast_fn": forecast_fn,
    }


# ── Method 3: TiMINo (simplified linear proxy) ──────────────────────────────

def fit_timino(X: np.ndarray, var_names: list, max_lag: int,
               t0: float, budget: float) -> dict:
    """
    Simplified linear TiMINo. See module docstring for the honest description
    of what's approximated vs. the original HSIC-based method.
    """
    n_vars = X.shape[1]
    T = X.shape[0]

    def _self_ar_residual(j, remaining):
        """Regress var j on its own lags only; return residuals over the
        usable sample range."""
        rows, targets = [], []
        for t in range(max_lag, T):
            rows.append([X[t - lag, j] for lag in range(1, max_lag + 1)])
            targets.append(X[t, j])
        Xd = np.column_stack([np.ones(len(rows)), np.array(rows)])
        yd = np.array(targets)
        try:
            beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
            resid = yd - Xd @ beta
        except Exception:
            resid = yd - yd.mean()
        return resid

    remaining = list(range(n_vars))
    order = []  # earliest = most "source-like" / least explained by others

    _check_deadline(t0, budget, "timino (ordering)")
    while remaining:
        best_var, best_score = None, np.inf
        for j in remaining:
            resid = _self_ar_residual(j, remaining)
            # Score: max |correlation| between this residual and any other
            # remaining variable's lagged values (linear proxy for
            # independence — see module docstring caveat).
            max_corr = 0.0
            for k in remaining:
                if k == j:
                    continue
                for lag in range(1, max_lag + 1):
                    other_lagged = X[max_lag - lag: T - lag, k]
                    if len(other_lagged) != len(resid):
                        n = min(len(other_lagged), len(resid))
                        other_lagged, r = other_lagged[-n:], resid[-n:]
                    else:
                        r = resid
                    if np.std(other_lagged) < 1e-12 or np.std(r) < 1e-12:
                        continue
                    c = abs(np.corrcoef(other_lagged, r)[0, 1])
                    max_corr = max(max_corr, c)
            if max_corr < best_score:
                best_score = max_corr
                best_var = j
        order.append(best_var)
        remaining.remove(best_var)
        _check_deadline(t0, budget, "timino (ordering loop)")

    # Given the order, each variable may be caused by any variable earlier in
    # the order (at any lag 1..max_lag) or its own lags. Fit OLS with all
    # such candidates, then prune to |t-stat| > 2 (simple significance gate
    # to avoid a dense, overfit graph).
    position = {v: idx for idx, v in enumerate(order)}
    parents = {j: [] for j in range(n_vars)}

    for j in range(n_vars):
        candidates = []
        for k in range(n_vars):
            if k != j and position[k] > position[j]:
                # k comes AFTER j in the causal order → k cannot be a valid
                # cause of j under this ordering. Own lags (k == j) are
                # always allowed; anything preceding j (position[k] <
                # position[j]) is also allowed.
                continue
            for lag in range(1, max_lag + 1):
                candidates.append((k, lag))

        if not candidates:
            continue

        rows, targets = [], []
        for t in range(max_lag, T):
            rows.append([X[t - lag, k] for k, lag in candidates])
            targets.append(X[t, j])
        Xd = np.column_stack([np.ones(len(rows)), np.array(rows)])
        yd = np.array(targets)
        try:
            beta, res, rank, sv = np.linalg.lstsq(Xd, yd, rcond=None)
            n, p = Xd.shape
            resid = yd - Xd @ beta
            dof = max(n - p, 1)
            sigma2 = float((resid @ resid) / dof)
            XtX_inv = np.linalg.pinv(Xd.T @ Xd)
            se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 1e-12, None))
            tstats = beta / se
        except Exception:
            continue

        for (k, lag), b, tstat in zip(candidates, beta[1:], tstats[1:]):
            if abs(tstat) > 2.0:
                parents[j].append((k, lag))

        _check_deadline(t0, budget, "timino (parent fit)")

    forecast_fn, _ = _fit_ols_forecast(X, parents, max_lag)

    return {
        "method": "timino",
        "var_names": var_names,
        "parents": parents,
        "forecast_fn": forecast_fn,
    }


FIT_FUNCS = {
    "varlingam": fit_varlingam,
    "pcmci": fit_pcmci,
    "timino": fit_timino,
}


# ── Out-of-sample backtest: which method actually works? ───────────────────

def fit_and_backtest(X: np.ndarray, var_names: list, method: str,
                      max_lag: int, train_frac: float = None) -> dict:
    """
    Chronological train/test split (same discipline as the sentiment engine's
    ablation): fit ONE causal model on TRAIN, then walk forward through TEST
    using the FROZEN fitted structure (no re-fitting) to generate one-step-
    ahead forecasts for EVERY variable simultaneously — a single SCM fit
    scores the whole universe at once, so this is fit once per
    (universe, window, method), not once per ticker.

    Returns a dict with the fitted `model`, and `per_var` — a list of
    per-variable OOS metrics (oos_r2, oos_correlation, oos_hit_rate) indexed
    the same way as var_names — or None if there isn't enough data.
    """
    train_frac = train_frac or config.TRAIN_FRAC
    T = X.shape[0]
    n_vars = X.shape[1]
    n = T - max_lag
    if n < config.MIN_TRAIN_SAMPLES:
        return None

    split = int(n * train_frac) + max_lag
    n_test = T - split
    if (split - max_lag) < config.MIN_TRAIN_SAMPLES or n_test < config.MIN_TEST_SAMPLES:
        return None

    X_train = X[:split]

    t0 = time.time()
    try:
        model = FIT_FUNCS[method](X_train, var_names, max_lag, t0, config.MAX_SECONDS_PER_FIT)
    except (FitTimeout, Exception):
        return None

    preds = np.full((n_test, n_vars), np.nan)
    actuals = X[split:T]
    for row, t in enumerate(range(split, T)):
        X_recent = X[t - max_lag: t]
        try:
            preds[row] = model["forecast_fn"](X_recent)
        except Exception:
            continue

    valid_rows = ~np.isnan(preds).any(axis=1)
    if valid_rows.sum() < config.MIN_TEST_SAMPLES:
        return None
    preds = preds[valid_rows]
    actuals = actuals[valid_rows]

    per_var = []
    for j in range(n_vars):
        p, a = preds[:, j], actuals[:, j]
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        oos_r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
        if np.std(p) > 1e-12 and np.std(a) > 1e-12:
            oos_corr = float(np.corrcoef(p, a)[0, 1])
        else:
            oos_corr = 0.0
        oos_hit_rate = float(np.mean(np.sign(p) == np.sign(a)))
        per_var.append({
            "oos_r2": oos_r2,
            "oos_correlation": oos_corr,
            "oos_hit_rate": oos_hit_rate,
        })

    return {
        "method": method,
        "model": model,
        "per_var": per_var,
        "n_train": split - max_lag,
        "n_test": int(valid_rows.sum()),
    }


def fit_live_and_forecast(X: np.ndarray, var_names: list, method: str,
                           max_lag: int) -> dict:
    """
    Refit the winning method on the FULL window (train+test combined — this
    is the live, deployable fit, not part of the OOS validation) and produce
    tomorrow's one-step-ahead forecast for every variable.
    """
    t0 = time.time()
    try:
        model = FIT_FUNCS[method](X, var_names, max_lag, t0, config.MAX_SECONDS_PER_FIT)
    except (FitTimeout, Exception):
        return None

    X_recent = X[-max_lag:]
    try:
        forecast = model["forecast_fn"](X_recent)
    except Exception:
        return None

    return {"model": model, "forecast": forecast}
