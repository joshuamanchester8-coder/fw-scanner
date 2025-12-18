import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import textwrap

# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="Financial Wisdom Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial Wisdom Market Scanner")
st.caption("Weekly breakout + consolidation box + risk-defined entries. Explainable BUY / PASS + TradingView export.")

# =========================
# Beginner-proof defaults
# =========================
DEFAULT_CFG = {
    "min_weeks_history": 80,
    "ma_weeks": 20,
    "lookback_high_weeks": 10,
    "consolidation_weeks": 12,
    "natr_max": 8.0,
    "min_weekly_move_pct": 5.0,
    "max_weekly_move_pct": 20.0,
    "max_upper_wick_pct": 50.0,
    "require_volume_spike": True,
    "vol_spike_min_pct": 0.0,   # 0 => >= prior week
    "max_stop_pct": 20.0,
    "max_workers": 8,
}


# =========================
# Data structures
# =========================
@dataclass
class ScanResult:
    symbol: str
    decision: str  # BUY / PASS
    score: int
    entry: Optional[float]
    stop: Optional[float]
    risk_pct: Optional[float]
    exit_rule: str
    reasons_pass: List[str]
    reasons_fail: List[str]
    metrics: Dict[str, Optional[float]]


# =========================
# Utility helpers
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_float(x) -> Optional[float]:
    try:
        if x is None: 
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(str(x))
    except Exception:
        return None

def pct(a: float, b: float) -> Optional[float]:
    # percent change from b to a
    if b is None or b == 0:
        return None
    return ((a - b) / b) * 100.0

def candle_metrics(curr, prev) -> Tuple[Optional[float], Optional[float]]:
    """
    upper_wick_pct = upper wick / range * 100
    weekly_move_pct = close change vs previous close
    """
    try:
        o = float(curr["Open"])
        h = float(curr["High"])
        l = float(curr["Low"])
        c = float(curr["Close"])
        rng = max(1e-9, h - l)
        upper_wick = h - max(o, c)
        upper_wick_pct = (upper_wick / rng) * 100.0
        weekly_move_pct = None
        if prev is not None:
            weekly_move_pct = pct(c, float(prev["Close"]))
        return weekly_move_pct, upper_wick_pct
    except Exception:
        return None, None

def consolidation_box(df_weekly: pd.DataFrame, weeks: int) -> Optional[Tuple[float, float]]:
    """
    Consolidation box from prior weeks (exclude latest candle)
    returns (box_high, box_low)
    """
    if df_weekly is None or len(df_weekly) < weeks + 2:
        return None
    prior = df_weekly.iloc[:-1].tail(weeks)
    if prior.empty:
        return None
    return float(prior["High"].max()), float(prior["Low"].min())

def ten_week_high_close(df_weekly: pd.DataFrame, lookback: int) -> Optional[float]:
    if df_weekly is None or len(df_weekly) < lookback:
        return None
    return float(df_weekly["Close"].tail(lookback).max())

def compute_macd(df_weekly: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (macd_line_last, macd_signal_last)
    """
    try:
        macd_df = ta.macd(df_weekly["Close"])
        if macd_df is None or macd_df.empty:
            return None, None
        # columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        macd_line = safe_float(macd_df.iloc[-1, 0])
        macd_sig = safe_float(macd_df.iloc[-1, 1])
        return macd_line, macd_sig
    except Exception:
        return None, None

def compute_natr(df_weekly: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        natr_s = ta.natr(df_weekly["High"], df_weekly["Low"], df_weekly["Close"], length=period)
        if natr_s is None or len(natr_s) == 0:
            return None
        return safe_float(natr_s.iloc[-1])
    except Exception:
        return None

def compute_sma(df_weekly: pd.DataFrame, period: int) -> Optional[float]:
    try:
        sma_s = ta.sma(df_weekly["Close"], length=period)
        if sma_s is None or len(sma_s) == 0:
            return None
        return safe_float(sma_s.iloc[-1])
    except Exception:
        return None

def tv_symbol(symbol: str, exchange_prefix: str) -> str:
    """
    TradingView expects EXCHANGE:SYMBOL. We'll let user choose prefix.
    """
    return f"{exchange_prefix}:{symbol}"


# =========================
# Data fetching (cached)
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1 hour cache
def fetch_weekly(symbol: str) -> Optional[pd.DataFrame]:
    """
    Pulls weekly OHLCV using yfinance (auto-adjust disabled for candle shape consistency).
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", interval="1wk", auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.dropna().copy()
        # Ensure required columns
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            return None
        return df
    except Exception:
        return None


# =========================
# Financial Wisdom evaluation
# =========================
def evaluate_fw(symbol: str, df_weekly: pd.DataFrame, cfg: dict) -> ScanResult:
    reasons_pass = []
    reasons_fail = []
    exit_rule = "Exit: Weekly MACD crosses DOWN below signal (raised stop)."

    # Basic history gate
    if df_weekly is None or len(df_weekly) < cfg["min_weeks_history"]:
        reasons_fail.append(f"Not enough weekly history (need ≥ {cfg['min_weeks_history']}).")
        return ScanResult(symbol, "PASS", 0, None, None, None, exit_rule, reasons_pass, reasons_fail, metrics={})

    curr = df_weekly.iloc[-1]
    prev = df_weekly.iloc[-2] if len(df_weekly) >= 2 else None
    close = float(curr["Close"])

    # Trend: 20W MA
    ma20 = compute_sma(df_weekly, cfg["ma_weeks"])
    if ma20 is None:
        reasons_fail.append(f"Cannot compute {cfg['ma_weeks']}-week MA.")
    elif close >= ma20:
        reasons_pass.append(f"Close above {cfg['ma_weeks']}-week MA (trend aligned).")
    else:
        reasons_fail.append(f"Close below {cfg['ma_weeks']}-week MA.")

    # MACD momentum
    macd_line, macd_sig = compute_macd(df_weekly)
    if macd_line is None or macd_sig is None:
        reasons_fail.append("Cannot compute MACD.")
    elif macd_line >= macd_sig:
        reasons_pass.append("MACD line ≥ signal (momentum supportive).")
    else:
        reasons_fail.append("MACD line < signal (momentum not supportive).")

    # NATR tightness
    natr = compute_natr(df_weekly, 14)
    if natr is None:
        reasons_fail.append("Cannot compute NATR.")
    elif natr < cfg["natr_max"]:
        reasons_pass.append(f"NATR {natr:.2f}% < {cfg['natr_max']}% (tight consolidation proxy).")
    else:
        reasons_fail.append(f"NATR {natr:.2f}% ≥ {cfg['natr_max']}% (too volatile).")

    # Consolidation box (exclude current candle)
    box = consolidation_box(df_weekly, cfg["consolidation_weeks"])
    if box is None:
        reasons_fail.append(f"Cannot compute consolidation box ({cfg['consolidation_weeks']}w).")
        box_high = None
        box_low = None
    else:
        box_high, box_low = box
        reasons_pass.append(f"Consolidation box computed ({cfg['consolidation_weeks']}w).")

    # Breakout: 10W high close
    ten_high = ten_week_high_close(df_weekly, cfg["lookback_high_weeks"])
    if ten_high is None:
        reasons_fail.append(f"Cannot compute {cfg['lookback_high_weeks']}-week closing high.")
    elif close >= ten_high:
        reasons_pass.append(f"Close is ≥ {cfg['lookback_high_weeks']}-week closing high.")
    else:
        reasons_fail.append(f"Close is NOT a {cfg['lookback_high_weeks']}-week closing high.")

    # Breakout candle move + wick filter
    weekly_move_pct, upper_wick_pct = candle_metrics(curr, prev)
    if weekly_move_pct is None:
        reasons_fail.append("Cannot compute weekly move %.")  # rare
    else:
        if cfg["min_weekly_move_pct"] <= weekly_move_pct <= cfg["max_weekly_move_pct"]:
            reasons_pass.append(
                f"Weekly move {weekly_move_pct:.2f}% within {cfg['min_weekly_move_pct']}–{cfg['max_weekly_move_pct']}%."
            )
        else:
            reasons_fail.append(
                f"Weekly move {weekly_move_pct:.2f}% outside {cfg['min_weekly_move_pct']}–{cfg['max_weekly_move_pct']}%."
            )

    if upper_wick_pct is None:
        reasons_fail.append("Cannot compute upper wick %.")  # rare
    else:
        if upper_wick_pct <= cfg["max_upper_wick_pct"]:
            reasons_pass.append(f"Upper wick {upper_wick_pct:.2f}% ≤ {cfg['max_upper_wick_pct']}%.")
        else:
            reasons_fail.append(f"Upper wick {upper_wick_pct:.2f}% > {cfg['max_upper_wick_pct']}% (selling pressure).")

    # Close above consolidation high
    if box_high is not None:
        if close > box_high:
            reasons_pass.append(f"Close above box high ({box_high:.2f}).")
        else:
            reasons_fail.append(f"Close not above box high ({box_high:.2f}).")

    # Volume spike vs prior week
    vol_spike_pct = None
    if cfg["require_volume_spike"] and prev is not None:
        prev_v = float(prev["Volume"])
        curr_v = float(curr["Volume"])
        if prev_v > 0:
            vol_spike_pct = ((curr_v - prev_v) / prev_v) * 100.0
            if vol_spike_pct >= cfg["vol_spike_min_pct"]:
                reasons_pass.append(f"Volume spike {vol_spike_pct:.1f}% ≥ {cfg['vol_spike_min_pct']}%.")
            else:
                reasons_fail.append(f"Volume spike {vol_spike_pct:.1f}% < {cfg['vol_spike_min_pct']}%.")

    # Stop / Risk logic (box-low stop)
    entry = close
    stop = None
    risk_pct = None
    if box_low is not None:
        stop = float(box_low)
        risk_pct = ((entry - stop) / entry) * 100.0
        if risk_pct <= cfg["max_stop_pct"]:
            reasons_pass.append(f"Stop risk {risk_pct:.2f}% ≤ {cfg['max_stop_pct']}%.")
        else:
            reasons_fail.append(f"Stop risk {risk_pct:.2f}% > {cfg['max_stop_pct']}% (reject).")
    else:
        reasons_fail.append("No box low → cannot place structured stop.")

    # Score / decision
    fail_count = len(reasons_fail)
    pass_count = len(reasons_pass)
    score = int(clamp(round((pass_count / max(1, pass_count + fail_count)) * 100), 0, 100))
    decision = "BUY" if fail_count == 0 else "PASS"

    metrics = {
        "close": close,
        "ma20": ma20,
        "macd": macd_line,
        "macd_signal": macd_sig,
        "natr": natr,
        "weekly_move_pct": weekly_move_pct,
        "upper_wick_pct": upper_wick_pct,
        "ten_week_high_close": ten_high,
        "box_high": box_high,
        "box_low": box_low,
        "vol_spike_pct": vol_spike_pct,
    }

    return ScanResult(symbol, decision, score, entry, stop, risk_pct, exit_rule, reasons_pass, reasons_fail, metrics)


# =========================
# Universe options (starter)
# =========================
STARTER_UNIVERSES = {
    "Tech (NASDAQ)": ["NVDA", "MSFT", "AMD", "AAPL", "GOOGL", "META"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "Financials": ["JPM", "BAC", "GS", "MS", "C"],
    "Healthcare": ["JNJ", "UNH", "LLY", "PFE", "MRK"],
    "Industrials": ["CAT", "DE", "HON", "GE", "BA"],
    "Defense": ["LMT", "NOC", "RTX", "GD"],
    "Retail": ["WMT", "COST", "TGT", "HD", "LOW"],
    "ETFs (Market)": ["SPY", "QQQ", "IWM", "XLK", "XLE", "XLF"],
}
  

def parse_tickers(text: str) -> List[str]:
    if not text:
        return []
    # split by commas, spaces, new lines
    parts = []
    for chunk in text.replace(",", " ").split():
        t = chunk.strip().upper()
        if t:
            parts.append(t)
    # de-dupe preserving order
    seen = set()
    out = []
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def read_uploaded_file(uploaded) -> List[str]:
    try:
        raw = uploaded.read()
        # Try CSV first
        try:
            df = pd.read_csv(io.BytesIO(raw))
            # If has 'symbol' column use it, else first column
            if "symbol" in df.columns:
                return parse_tickers("\n".join(df["symbol"].astype(str).tolist()))
            else:
                return parse_tickers("\n".join(df.iloc[:, 0].astype(str).tolist()))
        except Exception:
            # fallback: treat as plain text
            return parse_tickers(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return []


# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Controls")

mode = st.sidebar.selectbox(
    "Ticker Source",
    ["Starter List", "Paste Tickers", "Upload Tickers File"],
    index=0,
)

exchange_prefix = st.sidebar.selectbox(
    "TradingView Exchange Prefix (export)",
    ["NASDAQ", "NYSE", "AMEX", "CBOE", "TVC", "FX", "CRYPTO"],
    index=0
)

cfg = dict(DEFAULT_CFG)

with st.sidebar.expander("Financial Wisdom Rules (Weekly)", expanded=True):
    cfg["consolidation_weeks"] = st.number_input("Consolidation box weeks", 6, 30, cfg["consolidation_weeks"], 1)
    cfg["natr_max"] = st.number_input("NATR max (%)", 1.0, 25.0, float(cfg["natr_max"]), 0.5)
    cfg["ma_weeks"] = st.number_input("MA weeks", 10, 50, cfg["ma_weeks"], 1)
    cfg["lookback_high_weeks"] = st.number_input("Closing high lookback (weeks)", 5, 30, cfg["lookback_high_weeks"], 1)

    cfg["min_weekly_move_pct"] = st.number_input("Min breakout weekly move (%)", 0.0, 50.0, float(cfg["min_weekly_move_pct"]), 0.5)
    cfg["max_weekly_move_pct"] = st.number_input("Max breakout weekly move (%)", 1.0, 100.0, float(cfg["max_weekly_move_pct"]), 0.5)
    cfg["max_upper_wick_pct"] = st.number_input("Max upper wick (%)", 0.0, 100.0, float(cfg["max_upper_wick_pct"]), 1.0)

    cfg["require_volume_spike"] = st.checkbox("Require volume spike vs prior week", value=cfg["require_volume_spike"])
    cfg["vol_spike_min_pct"] = st.number_input("Volume spike min (%)", -50.0, 500.0, float(cfg["vol_spike_min_pct"]), 5.0)

    cfg["max_stop_pct"] = st.number_input("Max stop risk (%)", 5.0, 50.0, float(cfg["max_stop_pct"]), 0.5)
    cfg["min_weeks_history"] = st.number_input("Min weeks history", 40, 200, cfg["min_weeks_history"], 5)

with st.sidebar.expander("Performance", expanded=False):
    cfg["max_workers"] = st.slider("Parallel workers", 1, 20, cfg["max_workers"], 1)
    st.caption("More workers = faster but may trigger rate limits.")

st.sidebar.divider()
st.sidebar.caption("Tip: For best accuracy, scan a curated universe (50–500 tickers).")


# =========================
# Get tickers
# =========================
tickers: List[str] = []

if mode == "Starter List":
    pick = st.sidebar.selectbox("Pick a starter universe", list(STARTER_UNIVERSES.keys()), index=0)
    tickers = STARTER_UNIVERSES[pick]

elif mode == "Paste Tickers":
    pasted = st.sidebar.text_area(
        "Paste tickers (comma/space/newline separated)",
        value="NVDA MSFT AMD AAPL GOOGL META",
        height=120
    )
    tickers = parse_tickers(pasted)

elif mode == "Upload Tickers File":
    uploaded = st.sidebar.file_uploader("Upload .csv or .txt of tickers", type=["csv", "txt"])
    if uploaded:
        tickers = read_uploaded_file(uploaded)

tickers = [t for t in tickers if t]  # clean
if len(tickers) == 0:
    st.warning("Add at least 1 ticker in the sidebar to scan.")
    st.stop()


# =========================
# Run scan button
# =========================
colA, colB, colC = st.columns([1.2, 1.2, 2.0])
with colA:
    run = st.button("🚀 Run Scan", use_container_width=True)
with colB:
    show_only_buy = st.checkbox("Show only BUY", value=False)
with colC:
    search = st.text_input("Search results (ticker)", value="").strip().upper()

if not run:
    st.info("Click **Run Scan** to scan the tickers.")
    st.stop()


# =========================
# Scanner execution
# =========================
st.write("")
progress = st.progress(0)
status = st.empty()

results: List[ScanResult] = []

def scan_one(sym: str) -> ScanResult:
    df = fetch_weekly(sym)
    if df is None or df.empty:
        return ScanResult(
            sym, "PASS", 0, None, None, None,
            "Exit: Weekly MACD crosses DOWN below signal (raised stop).",
            [], [f"No data returned for {sym} (check ticker spelling)."], {}
        )
    return evaluate_fw(sym, df, cfg)

max_workers = int(cfg["max_workers"])
total = len(tickers)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(scan_one, sym): sym for sym in tickers}
    done = 0
    for fut in as_completed(futures):
        sym = futures[fut]
        try:
            r = fut.result()
        except Exception as e:
            r = ScanResult(
                sym, "PASS", 0, None, None, None,
                "Exit: Weekly MACD crosses DOWN below signal (raised stop).",
                [], [f"Error scanning {sym}: {e}"], {}
            )
        results.append(r)
        done += 1
        progress.progress(done / total)
        status.write(f"Scanning… {done}/{total}")

status.write("✅ Scan complete.")
progress.empty()


# =========================
# Sort + filter results
# =========================
results.sort(key=lambda r: (0 if r.decision == "BUY" else 1, -r.score, r.symbol))

filtered = results
if show_only_buy:
    filtered = [r for r in filtered if r.decision == "BUY"]
if search:
    filtered = [r for r in filtered if search in r.symbol]


# =========================
# TradingView export
# =========================
buy_syms = [r.symbol for r in results if r.decision == "BUY"]
tv_list_comma = ",".join([tv_symbol(s, exchange_prefix) for s in buy_syms])
tv_list_lines = "\n".join([tv_symbol(s, exchange_prefix) for s in buy_syms])

st.subheader("✅ Output")
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.4])
c1.metric("Tickers scanned", len(results))
c2.metric("BUY", sum(1 for r in results if r.decision == "BUY"))
c3.metric("PASS", sum(1 for r in results if r.decision == "PASS"))

with c4:
    st.markdown("**TradingView Export**")
    export_mode = st.radio("Format", ["Comma-separated", "Newline-separated"], horizontal=True, label_visibility="collapsed")
    export_text = tv_list_comma if export_mode == "Comma-separated" else tv_list_lines
    st.text_area("Copy this into TradingView watchlist import / your notes", value=export_text, height=80)
    st.download_button(
        "⬇️ Download watchlist.txt",
        data=export_text.encode("utf-8"),
        file_name="watchlist.txt",
        mime="text/plain",
        use_container_width=True
    )


# =========================
# Results table
# =========================
st.write("")
st.subheader("📋 Scan Results")

table_rows = []
for r in filtered:
    table_rows.append({
        "Symbol": r.symbol,
        "Decision": r.decision,
        "Score": r.score,
        "Entry": None if r.entry is None else round(r.entry, 2),
        "Stop": None if r.stop is None else round(r.stop, 2),
        "Risk %": None if r.risk_pct is None else round(r.risk_pct, 2),
        "Close": None if r.metrics.get("close") is None else round(r.metrics.get("close"), 2),
        "NATR%": None if r.metrics.get("natr") is None else round(r.metrics.get("natr"), 2),
        "Box High": None if r.metrics.get("box_high") is None else round(r.metrics.get("box_high"), 2),
        "Box Low": None if r.metrics.get("box_low") is None else round(r.metrics.get("box_low"), 2),
    })

df_out = pd.DataFrame(table_rows)
st.dataframe(df_out, use_container_width=True, height=360)


# =========================
# Explainable detail view
# =========================
st.write("")
st.subheader("🔎 Why was it picked? (Click a ticker)")

symbols_in_view = [r.symbol for r in filtered]
default_choice = symbols_in_view[0] if symbols_in_view else results[0].symbol
selected = st.selectbox("Select ticker", options=symbols_in_view if symbols_in_view else [r.symbol for r in results], index=0)

sel = next((r for r in results if r.symbol == selected), None)
if sel:
    left, right = st.columns([1.1, 1.4])

    with left:
        badge_color = "🟢" if sel.decision == "BUY" else "🔴"
        st.markdown(f"### {badge_color} {sel.symbol} — **{sel.decision}** (Score {sel.score})")

        st.write("**Ideal Entry / Stop (structure-based)**")
        st.write(f"- Entry: **{('—' if sel.entry is None else f'${sel.entry:.2f}') }**")
        st.write(f"- Stop: **{('—' if sel.stop is None else f'${sel.stop:.2f}') }**")
        st.write(f"- Risk: **{('—' if sel.risk_pct is None else f'{sel.risk_pct:.2f}%') }**")

        st.write("**Exit Rule**")
        st.write(f"- {sel.exit_rule}")

        st.write("**Key Metrics**")
        m = sel.metrics
        st.write(f"- Close: {m.get('close'):.2f}" if m.get("close") is not None else "- Close: —")
        st.write(f"- MA{cfg['ma_weeks']}: {m.get('ma20'):.2f}" if m.get("ma20") is not None else f"- MA{cfg['ma_weeks']}: —")
        st.write(f"- MACD: {m.get('macd'):.4f} | Signal: {m.get('macd_signal'):.4f}"
                 if m.get("macd") is not None and m.get("macd_signal") is not None else "- MACD: —")
        st.write(f"- NATR: {m.get('natr'):.2f}%"
                 if m.get("natr") is not None else "- NATR: —")
        st.write(f"- Box High/Low: {m.get('box_high'):.2f} / {m.get('box_low'):.2f}"
                 if m.get("box_high") is not None and m.get("box_low") is not None else "- Box High/Low: —")
        st.write(f"- Weekly Move: {m.get('weekly_move_pct'):.2f}%"
                 if m.get("weekly_move_pct") is not None else "- Weekly Move: —")
        st.write(f"- Upper Wick: {m.get('upper_wick_pct'):.2f}%"
                 if m.get("upper_wick_pct") is not None else "- Upper Wick: —")
        if m.get("vol_spike_pct") is not None:
            st.write(f"- Volume spike vs prior: {m.get('vol_spike_pct'):.1f}%")

    with right:
        st.markdown("### ✅ Passed Gates")
        if sel.reasons_pass:
            for x in sel.reasons_pass:
                st.success(x)
        else:
            st.info("None.")

        st.markdown("### ❌ Failed Gates (Blockers)")
        if sel.reasons_fail:
            for x in sel.reasons_fail:
                st.error(x)
        else:
            st.success("None. (This would be a BUY.)")


# =========================
# Beginner help section
# =========================
with st.expander("🧠 Beginner Help (How to use this fast)", expanded=False):
    st.markdown(
        textwrap.dedent(
            """
            **Fast workflow**
            1) Pick a ticker source (Starter / Paste / Upload)
            2) Click **Run Scan**
            3) Filter to **Show only BUY**
            4) Copy or download the **TradingView export**
            5) Click tickers to see exactly why they passed/failed

            **If you get mostly PASS**
            - Increase breakout tolerance (ex: allow max move 25%)
            - Relax NATR max (ex: 10%)
            - Turn off volume spike temporarily
            - Reduce consolidation weeks (ex: 10)

            **What “Entry/Stop” means here**
            - Entry = breakout close (MVP)
            - Stop = consolidation box low
            - If stop risk > max stop %, it becomes PASS
            """
        )
    )
