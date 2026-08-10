import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem

import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Causal SCM Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#0d1b2a;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
padding-left:0.5rem;border-left:5px solid #06a77d}
.hero-card{background:linear-gradient(135deg,#0d1b2a 0%,#1b263b 60%,#06a77d 130%);
color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
box-shadow:0 6px 20px rgba(6,167,125,0.25)}
.win-card{background:linear-gradient(135deg,#1b263b 0%,#415a77 100%);color:white;
border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
box-shadow:0 4px 12px rgba(65,90,119,0.3)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.method-badge{border-radius:6px;padding:2px 8px;font-size:0.72rem;font-weight:700;color:white}
.badge-varlingam{background:#2a9d8f}
.badge-pcmci{background:#e76f51}
.badge-timino{background:#e9c46a;color:#333}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⇢ Causal Discovery + SCM Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Time-series causal discovery (VarLiNGAM · PCMCI · TiMINo) → '
    'structural causal model → intervention-aware one-step forecast · '
    'Genuine out-of-sample method selection, not in-sample graph-fit statistics</div>',
    unsafe_allow_html=True)

st.warning(
    "⚠️ **Honest methodology notes**: tsFCI has no maintained Python implementation — "
    "**PCMCI** (Runge et al., `tigramite`) is used as the practical substitute (same "
    "family: constraint-based conditional-independence testing for time series). "
    "**TiMINo** here is a from-scratch **linear proxy** of the original method — it uses "
    "correlation as an independence test instead of the original's nonparametric HSIC, "
    "to stay within this suite's compute budget. See the README for the full rationale. "
    "The method-selection logic (🧪 Method Comparison tab) always picks by genuine "
    "out-of-sample forecast skill, never by in-sample fit."
)

HF_TOKEN = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO

METHOD_BADGE_CLASS = {
    "varlingam": "badge-varlingam",
    "pcmci": "badge-pcmci",
    "timino": "badge-timino",
}


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True)
                if f["type"] == "file"], None
    except Exception as e:
        return [], str(e)


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def method_badge(method: str) -> str:
    label = config.METHOD_LABELS.get(method, method)
    cls = METHOD_BADGE_CLASS.get(method, "badge-pcmci")
    return f'<span class="method-badge {cls}">{label}</span>'


st.sidebar.markdown("## ⇢ Causal SCM")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Max lag:** {config.MAX_LAG}d")
st.sidebar.markdown("**Methods:**")
for m in config.CAUSAL_METHODS:
    st.sidebar.markdown(f" • {config.METHOD_LABELS[m]}")
st.sidebar.markdown(f"**OOS train/test split:** {config.TRAIN_FRAC:.0%} / {1-config.TRAIN_FRAC:.0%}")

files, list_error = list_repo_files()

with st.expander("🔧 Debug: what the dashboard sees on HuggingFace", expanded=bool(list_error)):
    st.markdown(f"**Repo:** `{OUTPUT_REPO}` · **Token set:** {'yes' if bool(HF_TOKEN) else 'no'}")
    if list_error:
        st.error(f"Could not list repo files: {list_error}")
    else:
        st.write(f"{len(files)} file(s) found:")
        st.code("\n".join(sorted(files)) if files else "(empty)")

tab1_path = find_latest(files, "causal_scm_2")
tab2_path = find_latest(files, "causal_scm_windows_")
tab3_path = find_latest(files, "causal_scm_methods_")
tab4_path = find_latest(files, "causal_scm_persistence_")

if not tab1_path:
    if list_error:
        st.error("Could not reach HuggingFace to look for results (see 🔧 Debug above).")
    else:
        st.error(
            "Connected to HuggingFace successfully, but no file matching "
            "`causal_scm_2*.json` was found. Run `trainer.py` first."
        )
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2 = load_json(tab2_path) if tab2_path else None
data3 = load_json(tab3_path) if tab3_path else None
data4 = load_json(tab4_path) if tab4_path else None

universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None
universes3 = data3["universes"] if data3 and "error" not in data3 else None
universes4 = data4["universes"] if data4 and "error" not in data4 else None

history_days = data1.get("history_days", 0)
min_persistence_days = data4.get("min_persistence_days", config.MIN_PERSISTENCE_DAYS) if data4 else config.MIN_PERSISTENCE_DAYS

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")
st.sidebar.markdown(f"**History:** {history_days} day(s) tracked")
st.sidebar.markdown(f"**Persistence gate:** {min_persistence_days} consecutive positive day(s)")

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Best Window & Method per ETF",
    "🔍 Explore by Window",
    "🧪 Method Comparison",
    "📈 Signal Persistence",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Causal-Driven Forecast")

    with st.expander("📖 How This Engine Works", expanded=True):
        st.markdown("""
**1. Causal discovery** — for each universe × rolling window, three methods
independently discover a directed causal graph over ETF returns + macro
changes:

| Method | What it actually is |
|---|---|
| **VarLiNGAM** | Non-Gaussian instantaneous + lagged causal ordering (Hyvärinen et al. 2010) — real, maintained (`lingam`) |
| **PCMCI** | Constraint-based conditional-independence testing for time series (Runge et al. 2019, `tigramite`) — used as the practical substitute for **tsFCI**, which has no maintained Python implementation |
| **TiMINo** | A from-scratch **linear proxy** of the original method (Peters et al. 2013) — correlation instead of HSIC as the independence test, to fit this suite's compute budget |

**2. Structural forecast** — each method's discovered graph + coefficients
form a structural causal model. Tomorrow's forecast for a ticker is computed
from its estimated causal parents' most recent values through the fitted
structural equations — an *intervention-aware* forecast, not a black-box
regression fit to correlation.

**3. Honest method selection** — for every ticker, all 3 methods are
backtested on a chronological train/test split within each window (fit on
TRAIN, forecast forward through TEST with the frozen structure — no
peeking). The **winning method per ticker is whichever had the best genuine
out-of-sample R²** — never picked by in-sample graph statistics. See the
🧪 Method Comparison tab for the full breakdown.

**4. Regime adaptation** — causal graphs are refit fresh every window (a
form of regime adaptation by construction), plus a lightweight
realized-volatility-ratio check flags windows where the regime likely
shifted mid-window.
        """)

    ntd = next_trading_day()
    for universe_name in UNIVERSE_ORDER:
        uni_data = universes1.get(universe_name, {})
        full_scores = uni_data.get("full_scores", {})
        top_etfs = uni_data.get("top_etfs", [])
        if not full_scores:
            continue  # genuinely no data for this universe at all

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not top_etfs:
            if history_days < min_persistence_days:
                st.info(
                    f"📊 **Building history**: {history_days}/{min_persistence_days} day(s) "
                    f"tracked so far. Top picks require {min_persistence_days} consecutive "
                    "days of positive out-of-sample skill, not just today's snapshot — "
                    "nothing can qualify yet. Check back once more daily runs have "
                    "accumulated, or see the full ranking below for today's raw numbers."
                )
            else:
                st.info(
                    f"No ETFs in {label} currently show a persistent, out-of-sample "
                    f"positive track record ({min_persistence_days}+ consecutive days). "
                    "Shown honestly rather than padding the list with one-off picks — "
                    "see the full ranking below, or the 📈 Signal Persistence tab for "
                    "the complete track record."
                )
        else:
            cols = st.columns(3)
            for idx, etf in enumerate(top_etfs):
                with cols[idx]:
                    divs = [
                        f'<div class="ticker">{etf["ticker"]}</div>',
                        f'<div class="score">causal score = {etf["causal_score"]:+.4f}</div>',
                        f'<div class="score">{method_badge(etf["best_method"])}</div>',
                        f'<div class="score">OOS R\u00b2 = {etf["oos_r2"]:.3f} \u00b7 window = {etf["best_window"]}d</div>',
                        f'<div class="score">\U0001F525 {etf.get("streak", 0)}-day streak '
                        f'(of {etf.get("days_tracked", 0)} tracked)</div>',
                    ]
                    if etf.get("low_sample"):
                        divs.append(
                            '<div class="score" style="color:#ffb703">'
                            '⚠️ low sample — treat with caution</div>'
                        )
                    divs.append(f'<div class="next-day">\U0001F4C5 {ntd}</div>')
                    card_html = '<div class="hero-card">' + "".join(divs) + "</div>"
                    st.markdown(card_html, unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label}"):
            full = full_scores
            if full:
                rows = [{
                    "ETF": t,
                    "Causal Score": info["causal_score"],
                    "Best Window (d)": info["best_window"],
                    "Best Method": config.METHOD_LABELS.get(info["best_method"], info["best_method"]),
                    "OOS R²": info["oos_r2"],
                    "OOS Correlation": info["oos_correlation"],
                    "OOS Hit Rate": info["oos_hit_rate"],
                    "Streak (days)": info.get("streak", 0),
                    "Days Tracked": info.get("days_tracked", 0),
                    "✅ Persistent": "yes" if info.get("qualifies") else "",
                    "⚠️ Low Sample": "yes" if info.get("low_sample") else "",
                } for t, info in full.items()]
                df = pd.DataFrame(rows).sort_values("Causal Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Causal score is cross-sectionally z-scored per universe/window/method. "
        "OOS R² can be negative — that means the model is worse than guessing the mean, "
        "and is shown honestly rather than hidden.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Causal Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.py.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window", options=win_options, index=default_idx,
        format_func=lambda w: f"{w}d (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
Windows here are deliberately shorter than other engines in this suite
(max 1008d / ~4yr, vs. up to 4536d elsewhere) — causal discovery methods
assume a roughly stationary data-generating process within the window, and
multi-year windows spanning several macro regimes badly violate that
assumption on top of being far more expensive to fit.
        """)

    st.markdown(f"### Causal Rankings at **{selected_win}d** window")

    for universe_name in UNIVERSE_ORDER:
        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data or not win_data.get("full_ranking"):
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        top_etfs = win_data.get("top_etfs", [])
        if not top_etfs:
            if history_days < min_persistence_days:
                st.info(
                    f"📊 Building history: {history_days}/{min_persistence_days} day(s) "
                    "tracked — top picks require consecutive positive days, not just "
                    "today. See the full ranking below for today's raw numbers."
                )
            else:
                st.info(
                    f"No ETFs in {label} at {selected_win}d currently show a persistent "
                    f"track record ({min_persistence_days}+ consecutive positive days). "
                    "See the full ranking below."
                )
        else:
            cols = st.columns(3)
            for idx, etf in enumerate(top_etfs):
                with cols[idx]:
                    divs = [
                        f'<div class="ticker">{etf["ticker"]}</div>',
                        f'<div class="score">causal score = {etf["causal_score"]:+.4f}</div>',
                        f'<div class="score">{method_badge(etf["method"])}</div>',
                        f'<div class="score">OOS R\u00b2 = {etf["oos_r2"]:.3f}</div>',
                        f'<div class="score">\U0001F525 {etf.get("streak", 0)}-day streak</div>',
                    ]
                    if etf.get("low_sample"):
                        divs.append('<div class="score" style="color:#ffb703">⚠️ low sample</div>')
                    card_html = '<div class="win-card">' + "".join(divs) + "</div>"
                    st.markdown(card_html, unsafe_allow_html=True)

        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                # Defensive against schema drift across trainer.py versions:
                # 3-element [ticker, score, method] (earliest), 4-element
                # [..., low_sample] (added with the ridge-fix release), or
                # current 6-element [..., streak, qualifies]. Pad rather
                # than crash if an older JSON file is ever loaded against
                # this dashboard version.
                def _normalize(r):
                    r = list(r)
                    defaults = [None, None, None, False, 0, False]  # ticker/score/method have no default
                    return r + defaults[len(r):6]
                normalized = [_normalize(r) for r in rows]
                df = pd.DataFrame(normalized, columns=[
                    "ETF", "Causal Score", "Method", "Low Sample", "Streak", "Persistent"
                ])
                df["Method"] = df["Method"].map(lambda m: config.METHOD_LABELS.get(m, m))
                df["Low Sample"] = df["Low Sample"].map(lambda b: "⚠️ yes" if b else "")
                df["Persistent"] = df["Persistent"].map(lambda b: "✅ yes" if b else "")
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Method Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🧪 Method Comparison — Which Causal Method Actually Works")

    st.markdown("""
The other tabs already pick the winning method per ticker automatically.
This tab shows the **full breakdown** behind that choice — every method's
out-of-sample forecast skill for every ticker, side by side, so the
selection is auditable rather than a black box.

**How to read it**: prefer **OOS R²** and **OOS Correlation** over hit rate
— hit rate on overlapping/short test windows can look deceptively good even
when the model has no real skill. A method with the highest in-sample fit
but a poor OOS R² is exactly the failure mode this comparison exists to
catch.
    """)

    if not universes3:
        st.info("No method-comparison data found yet. Run `trainer.py` to populate this tab.")
    else:
        for universe_name in UNIVERSE_ORDER:
            label = UNIVERSE_LABELS.get(universe_name, universe_name)
            uni_data = universes3.get(universe_name, {})
            windows_data = uni_data.get("windows", {})
            if not windows_data:
                continue

            st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

            win_options = sorted([int(w) for w in windows_data.keys()])
            sel = st.selectbox(
                "Window", options=win_options, format_func=lambda w: f"{w}d",
                key=f"method_win_{universe_name}",
            )
            detail = windows_data.get(str(sel), {})

            if not detail:
                st.info(f"No tickers had enough data for {sel}d in this universe.")
                st.divider()
                continue

            rows = []
            for ticker, methods in detail.items():
                for m in config.CAUSAL_METHODS:
                    if m not in methods:
                        continue
                    d = methods[m]
                    rows.append({
                        "ETF": ticker,
                        "Method": config.METHOD_LABELS.get(m, m),
                        "OOS R²": d["oos_r2"],
                        "OOS Correlation": d["oos_correlation"],
                        "OOS Hit Rate": d["oos_hit_rate"],
                        "N Train": d["n_train"],
                        "N Test": d["n_test"],
                        "⚠️ Low Sample": "yes" if d.get("low_sample") else "",
                    })

            if rows:
                df = pd.DataFrame(rows).sort_values(["ETF", "OOS R²"], ascending=[True, False])
                st.dataframe(
                    df.style.format({
                        "OOS R²": "{:.3f}", "OOS Correlation": "{:.3f}", "OOS Hit Rate": "{:.2f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

                best_per_ticker = df.loc[df.groupby("ETF")["OOS R²"].idxmax()]
                win_counts = best_per_ticker["Method"].value_counts()
                st.caption(
                    "Wins by OOS R² in this window: " +
                    " · ".join(f"{m} ({c})" for m, c in win_counts.items())
                )

            st.divider()

        st.caption(
            f"Run date: {data3.get('run_date','?')} · "
            "Chronological train/test split per window · frozen structure walked forward through test."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Signal Persistence
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📈 Signal Persistence — Is This a Real Edge or a Lucky Day?")

    st.markdown(f"""
Every other tab shows a single day's snapshot. This tab is why the top-N
picks on Tabs 1 & 2 are gated the way they are: a (ticker, window, method)
combination only earns a "top pick" label once it has shown **positive
out-of-sample R² on {min_persistence_days} consecutive most-recent daily
runs, including today** — not just today's number.

This exists because a single day is not evidence of a durable edge. During
this engine's development, the exact same window/method combination for one
ticker showed OOS R² of 0.0085 on one day's ridge-regularization setting and
0.20 on another — a 24x swing from a hyperparameter choice alone, with
nothing about the actual market changing in between. Persistence — the same
combination showing up positive, day after day — is a much stronger signal
than any single day's R², however good it looks.
    """)

    st.info(
        f"📊 **History status**: {history_days} day(s) tracked so far "
        f"(gate requires {min_persistence_days}). "
        + ("Persistence-qualified picks are now possible." if history_days >= min_persistence_days
           else "Top-N picks on Tabs 1 & 2 will stay empty until enough history accumulates — "
                "this is expected, not a bug.")
    )

    if not universes4:
        st.info("No persistence data found yet. Run `trainer.py` to populate this tab.")
    else:
        for universe_name in UNIVERSE_ORDER:
            label = UNIVERSE_LABELS.get(universe_name, universe_name)
            uni_data = universes4.get(universe_name, {})
            windows_data = uni_data.get("windows", {})
            if not windows_data:
                continue

            st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

            win_options = sorted([int(w) for w in windows_data.keys()])
            sel = st.selectbox(
                "Window", options=win_options, format_func=lambda w: f"{w}d",
                key=f"persist_win_{universe_name}",
            )
            detail = windows_data.get(str(sel), {})

            if not detail:
                st.info(f"No tickers had enough data for {sel}d in this universe.")
                st.divider()
                continue

            rows = []
            for ticker, methods in detail.items():
                for m in config.CAUSAL_METHODS:
                    if m not in methods:
                        continue
                    d = methods[m]
                    rows.append({
                        "ETF": ticker,
                        "Method": config.METHOD_LABELS.get(m, m),
                        "Current Streak": d["streak"],
                        "Days Tracked": d["days_tracked"],
                        "Qualifies": "✅ yes" if d["qualifies"] else "",
                        "Latest R²": d["latest_r2"] if d["latest_r2"] is not None else float("nan"),
                    })

            if rows:
                df = pd.DataFrame(rows).sort_values(
                    ["Current Streak", "Latest R²"], ascending=[False, False]
                )
                st.dataframe(
                    df.style.format({"Latest R²": "{:.3f}"}),
                    use_container_width=True, hide_index=True,
                )
                n_qualified = (df["Qualifies"] != "").sum()
                st.caption(
                    f"{n_qualified} of {len(df)} (ticker, method) combination(s) "
                    f"at {sel}d currently qualify (streak ≥ {min_persistence_days} days)."
                )

            st.divider()

        st.caption(
            f"Run date: {data4.get('run_date','?')} · "
            f"History retained: up to {config.HISTORY_RETENTION_DAYS} days · "
            "Streak = consecutive most-recent days with OOS R² > 0, walking backward from today."
        )
