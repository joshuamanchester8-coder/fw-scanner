import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import io
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="Financial Wisdom Market Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial Wisdom Market Scanner")
st.caption("Weekly breakout + consolidation box + risk-defined entries. Explainable READY / WATCHLIST / PASS + TradingView export + Journal export.")

# =========================
# Defaults (beginner-proof)
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
    "vol_spike_min_pct": 0.0,     # 0 => >= prior week
    "max_stop_pct": 20.0,         # reject if box-low stop is too far
    "max_workers": 8,
}

# =========================
# Data structures
# =========================
@dataclass
class ScanResult:
    symbol: str
    decision_base: str            # BUY or PASS based on gates only
    score: int
    entry: Optional[float]
    stop: Optional[float]
    risk_pct: Optional[float]
    exit_rule: str
    reasons_pass: List[str]
    reasons_fail: List[str]
    metrics: Dict[str, Optional[float]]

@dataclass
class TradePlan:
    symbol: str
    status: str                   # READY / WATCHLIST / PASS / SKIP_RISK_CAP / SKIP_NO_SHARES
    score: int
    entry: Optional[float]
    stop: Optional[float]
    shares: int
    position_value: float
    risk_dollars: float
    risk_pct_stop: Optional[float]  # same as ScanResult.risk_pct
    reason_status: str

# =========================
# Helpers
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

def lookback_high_close_excluding_current(df_weekly: pd.DataFrame, lookback: int) -> Optional[float]:
    """
    Breakout should be relative to PRIOR closes (exclude current candle)
    """
    if df_weekly is None or len(df_weekly) < lookback + 1:
        return None
    prior = df_weekly["Close"].iloc[:-1].tail(lookback)
    if prior.empty:
        return None
    return float(prior.max())

def compute_macd(df_weekly: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    try:
        macd_df = ta.macd(df_weekly["Close"])
        if macd_df is None or macd_df.empty:
            return None, None
        cols = list(macd_df.columns)
        if len(cols) < 2:
            return None, None
        macd_line = safe_float(macd_df[cols[0]].iloc[-1])
        macd_sig = safe_float(macd_df[cols[1]].iloc[-1])
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
    return f"{exchange_prefix}:{symbol}"

def now_eastern() -> datetime:
    """
    Returns current time in US/Eastern if available; else UTC.
    """
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        pass
    return datetime.now(timezone.utc)

def weekly_close_confirmed() -> bool:
    """
    Conservative confirmation:
    - True only on Friday after 4:10pm ET, or Saturday/Sunday (week is closed).
    This avoids treating mid-week "current weekly candle" as confirmed.
    """
    t = now_eastern()
    wd = t.weekday()  # Mon=0 .. Sun=6
    # Saturday/Sunday -> week effectively closed
    if wd in (5, 6):
        return True
    # Friday after ~4:10pm ET -> closed
    if wd == 4:
        if (t.hour > 16) or (t.hour == 16 and t.minute >= 10):
            return True
    return False

# =========================
# Data fetching (cached)
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1 hour cache
def fetch_weekly(symbol: str) -> Optional[pd.DataFrame]:
    """
    Weekly OHLCV from yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", interval="1wk", auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.dropna().copy()
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(set(df.columns)):
            return None
        return df
    except Exception:
        return None

# =========================
# Financial Wisdom evaluation (weekly)
# =========================
def evaluate_fw(symbol: str, df_weekly: pd.DataFrame, cfg: dict) -> ScanResult:
    reasons_pass: List[str] = []
    reasons_fail: List[str] = []
    exit_rule = "Exit: Weekly MACD crosses DOWN below signal (raise stop as structure improves)."

    if df_weekly is None or len(df_weekly) < cfg["min_weeks_history"]:
        reasons_fail.append(f"Not enough weekly history (need ≥ {cfg['min_weeks_history']}).")
        return ScanResult(symbol, "PASS", 0, None, None, None, exit_rule, reasons_pass, reasons_fail, metrics={})

    curr = df_weekly.iloc[-1]
    prev = df_weekly.iloc[-2] if len(df_weekly) >= 2 else None
    close = float(curr["Close"])

    # Trend: MA
    ma = compute_sma(df_weekly, cfg["ma_weeks"])
    if ma is None:
        reasons_fail.append(f"Cannot compute MA{cfg['ma_weeks']}.")
    elif close >= ma:
        reasons_pass.append(f"Close above MA{cfg['ma_weeks']} (trend aligned).")
    else:
        reasons_fail.append(f"Close below MA{cfg['ma_weeks']}.")

    # Momentum: MACD
    macd_line, macd_sig = compute_macd(df_weekly)
    if macd_line is None or macd_sig is None:
        reasons_fail.append("Cannot compute MACD.")
    elif macd_line >= macd_sig:
        reasons_pass.append("MACD line ≥ signal (momentum supportive).")
    else:
        reasons_fail.append("MACD line < signal (momentum not supportive).")

    # Tightness proxy: NATR
    natr = compute_natr(df_weekly, 14)
    if natr is None:
        reasons_fail.append("Cannot compute NATR.")
    elif natr < cfg["natr_max"]:
        reasons_pass.append(f"NATR {natr:.2f}% < {cfg['natr_max']}% (tight).")
    else:
        reasons_fail.append(f"NATR {natr:.2f}% ≥ {cfg['natr_max']}% (too volatile).")

    # Consolidation box (exclude current candle)
    box = consolidation_box(df_weekly, cfg["consolidation_weeks"])
    box_high = box_low = None
    if box is None:
        reasons_fail.append(f"Cannot compute consolidation box ({cfg['consolidation_weeks']}w).")
    else:
        box_high, box_low = box
        reasons_pass.append(f"Consolidation box ({cfg['consolidation_weeks']}w) computed.")

    # Breakout: close > prior lookback closing high
    prior_high_close = lookback_high_close_excluding_current(df_weekly, cfg["lookback_high_weeks"])
    if prior_high_close is None:
        reasons_fail.append(f"Cannot compute prior {cfg['lookback_high_weeks']}-week closing high.")
    elif close > prior_high_close:
        reasons_pass.append(f"Close > prior {cfg['lookback_high_weeks']}-week closing high.")
    else:
        reasons_fail.append(f"Close not above prior {cfg['lookback_high_weeks']}-week closing high.")

    # Breakout candle quality
    weekly_move_pct, upper_wick_pct = candle_metrics(curr, prev)
    if weekly_move_pct is None:
        reasons_fail.append("Cannot compute weekly move %.")  # rare
    else:
        if cfg["min_weekly_move_pct"] <= weekly_move_pct <= cfg["max_weekly_move_pct"]:
            reasons_pass.append(f"Weekly move {weekly_move_pct:.2f}% within {cfg['min_weekly_move_pct']}–{cfg['max_weekly_move_pct']}%.")
        else:
            reasons_fail.append(f"Weekly move {weekly_move_pct:.2f}% outside {cfg['min_weekly_move_pct']}–{cfg['max_weekly_move_pct']}%.")

    if upper_wick_pct is None:
        reasons_fail.append("Cannot compute upper wick %.")  # rare
    else:
        if upper_wick_pct <= cfg["max_upper_wick_pct"]:
            reasons_pass.append(f"Upper wick {upper_wick_pct:.2f}% ≤ {cfg['max_upper_wick_pct']}%.")
        else:
            reasons_fail.append(f"Upper wick {upper_wick_pct:.2f}% > {cfg['max_upper_wick_pct']}% (selling pressure).")

    # Close above box high
    if box_high is not None:
        if close > box_high:
            reasons_pass.append(f"Close above box high ({box_high:.2f}).")
        else:
            reasons_fail.append(f"Close not above box high ({box_high:.2f}).")

    # Volume spike (optional)
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

    # Entry / stop (box low)
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

    fail_count = len(reasons_fail)
    pass_count = len(reasons_pass)
    score = int(clamp(round((pass_count / max(1, pass_count + fail_count)) * 100), 0, 100))
    decision_base = "BUY" if fail_count == 0 else "PASS"

    metrics = {
        "close": close,
        "ma": ma,
        "macd": macd_line,
        "macd_signal": macd_sig,
        "natr": natr,
        "weekly_move_pct": weekly_move_pct,
        "upper_wick_pct": upper_wick_pct,
        "prior_high_close": prior_high_close,
        "box_high": box_high,
        "box_low": box_low,
        "vol_spike_pct": vol_spike_pct,
    }
    return ScanResult(symbol, decision_base, score, entry, stop, risk_pct, exit_rule, reasons_pass, reasons_fail, metrics)

# =========================
# Starter universes (sector-first scanning)
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
        # Try CSV
        try:
            df = pd.read_csv(io.BytesIO(raw))
            if "symbol" in df.columns:
                return parse_tickers("\n".join(df["symbol"].astype(str).tolist()))
            return parse_tickers("\n".join(df.iloc[:, 0].astype(str).tolist()))
        except Exception:
            return parse_tickers(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return []

# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Scanner Controls")

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

with st.sidebar.expander("Financial Wisdom Rules (Weekly)", expanded=True):
    cfg = dict(DEFAULT_CFG)
    cfg["consolidation_weeks"] = st.number_input("Consolidation box weeks", 6, 30, cfg["consolidation_weeks"], 1)
    cfg["natr_max"] = st.number_input("NATR max (%)", 1.0, 25.0, float(cfg["natr_max"]), 0.5)
    cfg["ma_weeks"] = st.number_input("MA weeks", 10, 50, cfg["ma_weeks"], 1)
    cfg["lookback_high_weeks"] = st.number_input("Breakout vs prior closing high (weeks)", 5, 30, cfg["lookback_high_weeks"], 1)
    cfg["min_weekly_move_pct"] = st.number_input("Min breakout weekly move (%)", 0.0, 50.0, float(cfg["min_weekly_move_pct"]), 0.5)
    cfg["max_weekly_move_pct"] = st.number_input("Max breakout weekly move (%)", 1.0, 100.0, float(cfg["max_weekly_move_pct"]), 0.5)
    cfg["max_upper_wick_pct"] = st.number_input("Max upper wick (%)", 0.0, 100.0, float(cfg["max_upper_wick_pct"]), 1.0)
    cfg["require_volume_spike"] = st.checkbox("Require volume spike vs prior week", value=cfg["require_volume_spike"])
    cfg["vol_spike_min_pct"] = st.number_input("Volume spike min (%)", -50.0, 500.0, float(cfg["vol_spike_min_pct"]), 5.0)
    cfg["max_stop_pct"] = st.number_input("Max stop risk (%)", 5.0, 50.0, float(cfg["max_stop_pct"]), 0.5)
    cfg["min_weeks_history"] = st.number_input("Min weeks history", 40, 200, cfg["min_weeks_history"], 5)

with st.sidebar.expander("💰 Risk & Position Sizing", expanded=True):
    account_size = st.number_input("Account size ($)", min_value=1000, value=25000, step=1000)
    risk_pct_per_trade = st.selectbox("Risk per trade (%)", [0.25, 0.5, 0.75, 1.0], index=1)
    risk_dollars_per_trade = account_size * (risk_pct_per_trade / 100.0)
    max_portfolio_risk_pct = st.selectbox("Max total open risk (new trades) %", [2.0, 3.0, 5.0, 7.5, 10.0], index=2)
    max_portfolio_risk_dollars = account_size * (max_portfolio_risk_pct / 100.0)
    max_new_positions = st.slider("Max new positions to take", 1, 25, 10, 1)
    st.caption(f"Risk per trade: **${risk_dollars_per_trade:,.2f}** | Portfolio risk cap: **${max_portfolio_risk_dollars:,.2f}**")

with st.sidebar.expander("✅ Execution Timing", expanded=True):
    enforce_weekly_close = st.checkbox("Enforce weekly-close confirmation (FW best practice)", value=True)
    confirmed = weekly_close_confirmed()
    if enforce_weekly_close:
        st.caption(f"Weekly close confirmed right now? **{'YES' if confirmed else 'NO'}** (ET-based)")
        st.caption("If NO, BUY setups will be tagged WATCHLIST (not READY).")
    else:
        st.caption("Weekly-close confirmation disabled: BUY setups can be marked READY anytime.")

with st.sidebar.expander("Performance", expanded=False):
    cfg["max_workers"] = st.slider("Parallel workers", 1, 20, cfg["max_workers"], 1)
    st.caption("More workers = faster but may trigger rate limits/timeouts.")

st.sidebar.divider()
st.sidebar.caption("Pro workflow: Run scans Friday after close → build watchlist → execute Monday.")

# =========================
# Ticker input
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
else:
    uploaded = st.sidebar.file_uploader("Upload .csv or .txt of tickers", type=["csv", "txt"])
    if uploaded:
        tickers = read_uploaded_file(uploaded)

tickers = [t for t in tickers if t]
tickers = list(dict.fromkeys(tickers))  # dedupe stable

# =========================
# Top controls
# =========================
st.write("")
colA, colB, colC = st.columns([1.2, 1.2, 2.0])
with colA:
    run = st.button("🚀 Run Scan", use_container_width=True)
with colB:
    show_only_ready = st.checkbox("Show only READY", value=False)
with colC:
    search = st.text_input("Search results (ticker)", value="").strip().upper()

if not tickers:
    st.warning("Add tickers in the sidebar first.")
    st.stop()

if not run:
    st.info("Click **Run Scan** to scan the tickers.")
    st.stop()

# =========================
# Execute scan
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
            "Exit: Weekly MACD crosses DOWN below signal.",
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
                "Exit: Weekly MACD crosses DOWN below signal.",
                [], [f"Error scanning {sym}: {e}"], {}
            )
        results.append(r)
        done += 1
        progress.progress(done / total)
        status.write(f"Scanning… {done}/{total}")

status.write("✅ Scan complete.")
progress.empty()

# =========================
# Build trade plans (position sizing + portfolio risk cap + READY/WATCHLIST)
# =========================
# Sort by: BUY first, higher score first
results.sort(key=lambda r: (0 if r.decision_base == "BUY" else 1, -r.score, r.symbol))

confirmed_now = weekly_close_confirmed() if enforce_weekly_close else True

trade_plans: List[TradePlan] = []
portfolio_risk_used = 0.0
new_positions_used = 0

for r in results:
    if r.decision_base != "BUY":
        trade_plans.append(TradePlan(
            symbol=r.symbol,
            status="PASS",
            score=r.score,
            entry=r.entry,
            stop=r.stop,
            shares=0,
            position_value=0.0,
            risk_dollars=0.0,
            risk_pct_stop=r.risk_pct,
            reason_status="Failed at least one FW gate."
        ))
        continue

    # Weekly close confirmation gate -> WATCHLIST only (not READY)
    if not confirmed_now:
        trade_plans.append(TradePlan(
            symbol=r.symbol,
            status="WATCHLIST",
            score=r.score,
            entry=r.entry,
            stop=r.stop,
            shares=0,
            position_value=0.0,
            risk_dollars=0.0,
            risk_pct_stop=r.risk_pct,
            reason_status="Weekly close not confirmed yet (use as watchlist, confirm after Friday close)."
        ))
        continue

    # Must have entry/stop for sizing
    if r.entry is None or r.stop is None or r.entry <= r.stop:
        trade_plans.append(TradePlan(
            symbol=r.symbol,
            status="WATCHLIST",
            score=r.score,
            entry=r.entry,
            stop=r.stop,
            shares=0,
            position_value=0.0,
            risk_dollars=0.0,
            risk_pct_stop=r.risk_pct,
            reason_status="Missing valid structured stop for sizing."
        ))
        continue

    # Risk per share
    per_share_risk = r.entry - r.stop
    shares = int(risk_dollars_per_trade // per_share_risk)

    if shares <= 0:
        trade_plans.append(TradePlan(
            symbol=r.symbol,
            status="SKIP_NO_SHARES",
            score=r.score,
            entry=r.entry,
            stop=r.stop,
            shares=0,
            position_value=0.0,
            risk_dollars=0.0,
            risk_pct_stop=r.risk_pct,
            reason_status="Risk per share too large for your risk-per-trade setting."
        ))
        continue

    position_value = shares * r.entry
    trade_risk_dollars = shares * per_share_risk

    # Portfolio risk cap + max positions
    if new_positions_used >= max_new_positions:
        trade_plans.append(TradePlan(
            symbol=r.symbol,
            status="SKIP_RISK_CAP",
            score=r.score,
            entry=r.entry,
            stop=r.stop,
            shares=shares,
            position_value=position_value,
            risk_dollars=trade_risk_dollars,
            risk_pct_stop=r.risk_pct,
            reason_status="Max new positions reached."
        ))
        continue

    if (portfolio_risk_used + trade_risk_dollars) > max_portfolio_risk_dollars:
        trade_plans.append(TradePlan(
            symbol=r.symbol,
            status="SKIP_RISK_CAP",
            score=r.score,
            entry=r.entry,
            stop=r.stop,
            shares=shares,
            position_value=position_value,
            risk_dollars=trade_risk_dollars,
            risk_pct_stop=r.risk_pct,
            reason_status="Adding this trade would exceed portfolio risk cap."
        ))
        continue

    # READY
    portfolio_risk_used += trade_risk_dollars
    new_positions_used += 1
    trade_plans.append(TradePlan(
        symbol=r.symbol,
        status="READY",
        score=r.score,
        entry=r.entry,
        stop=r.stop,
        shares=shares,
        position_value=position_value,
        risk_dollars=trade_risk_dollars,
        risk_pct_stop=r.risk_pct,
        reason_status="Meets all FW gates + sizing + portfolio risk limits."
    ))

# =========================
# Filter/search view
# =========================
def matches_search(sym: str) -> bool:
    return (not search) or (search in sym.upper())

view_plans = [p for p in trade_plans if matches_search(p.symbol)]
if show_only_ready:
    view_plans = [p for p in view_plans if p.status == "READY"]

# =========================
# Summary + TradingView export
# =========================
ready_syms = [p.symbol for p in trade_plans if p.status == "READY"]
watch_syms = [p.symbol for p in trade_plans if p.status == "WATCHLIST"]

tv_ready_comma = ",".join([tv_symbol(s, exchange_prefix) for s in ready_syms])
tv_ready_lines = "\n".join([tv_symbol(s, exchange_prefix) for s in ready_syms])

st.subheader("✅ Output")
c1, c2, c3, c4, c5 = st.columns([1.1, 1.0, 1.0, 1.2, 1.7])
c1.metric("Scanned", len(trade_plans))
c2.metric("READY", sum(1 for p in trade_plans if p.status == "READY"))
c3.metric("WATCHLIST", sum(1 for p in trade_plans if p.status == "WATCHLIST"))
c4.metric("PASS", sum(1 for p in trade_plans if p.status == "PASS"))
c5.metric("New-trade risk used", f"${portfolio_risk_used:,.0f} / ${max_portfolio_risk_dollars:,.0f}")

st.markdown("**TradingView Export (READY list)**")
export_mode = st.radio("Format", ["Comma-separated", "Newline-separated"], horizontal=True)
export_text = tv_ready_comma if export_mode == "Comma-separated" else tv_ready_lines
st.text_area("Copy into TradingView watchlist / notes", value=export_text, height=80)
st.download_button(
    "⬇️ Download READY_watchlist.txt",
    data=export_text.encode("utf-8"),
    file_name="READY_watchlist.txt",
    mime="text/plain",
    use_container_width=True
)

# =========================
# Results table
# =========================
st.write("")
st.subheader("📋 Scan Results (Strategy + Execution)")

table_rows = []
for p in view_plans:
    # Find raw metrics for display
    r = next((x for x in results if x.symbol == p.symbol), None)
    close = r.metrics.get("close") if r else None
    natr = r.metrics.get("natr") if r else None
    box_high = r.metrics.get("box_high") if r else None
    box_low = r.metrics.get("box_low") if r else None

    table_rows.append({
        "Symbol": p.symbol,
        "Status": p.status,
        "Score": p.score,
        "Entry": None if p.entry is None else round(p.entry, 2),
        "Stop": None if p.stop is None else round(p.stop, 2),
        "Stop Risk %": None if p.risk_pct_stop is None else round(p.risk_pct_stop, 2),
        "Shares": p.shares,
        "Position $": round(p.position_value, 2),
        "Risk $": round(p.risk_dollars, 2),
        "Close": None if close is None else round(close, 2),
        "NATR%": None if natr is None else round(natr, 2),
        "Box High": None if box_high is None else round(box_high, 2),
        "Box Low": None if box_low is None else round(box_low, 2),
    })

df_out = pd.DataFrame(table_rows)
st.dataframe(df_out, use_container_width=True, height=380)

# =========================
# Journal export (CSV)
# =========================
st.write("")
st.subheader("🧾 Journal Export (READY trades)")

journal_rows = []
timestamp_et = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")

for p in trade_plans:
    if p.status != "READY":
        continue
    r = next((x for x in results if x.symbol == p.symbol), None)
    journal_rows.append({
        "timestamp": timestamp_et,
        "symbol": p.symbol,
        "status": p.status,
        "score": p.score,
        "entry": p.entry,
        "stop": p.stop,
        "shares": p.shares,
        "position_value": p.position_value,
        "risk_dollars": p.risk_dollars,
        "risk_per_trade_target": risk_dollars_per_trade,
        "portfolio_risk_cap": max_portfolio_risk_dollars,
        "reason": p.reason_status,
        "passed_gates": " | ".join(r.reasons_pass) if r else "",
        "failed_gates": " | ".join(r.reasons_fail) if r else "",
        "exit_rule": r.exit_rule if r else "",
    })

df_journal = pd.DataFrame(journal_rows)
st.dataframe(df_journal, use_container_width=True, height=220)

csv_bytes = df_journal.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download READY_journal.csv",
    data=csv_bytes,
    file_name="READY_journal.csv",
    mime="text/csv",
    use_container_width=True
)

# =========================
# Explainability panel
# =========================
st.write("")
st.subheader("🔎 Why was it picked? (Click a ticker)")

symbols_in_view = [p.symbol for p in view_plans] if view_plans else [p.symbol for p in trade_plans]
selected = st.selectbox("Select ticker", options=symbols_in_view, index=0)

sel_r = next((r for r in results if r.symbol == selected), None)
sel_p = next((p for p in trade_plans if p.symbol == selected), None)

if sel_r and sel_p:
    left, right = st.columns([1.15, 1.35])

    with left:
        icon = "🟢" if sel_p.status == "READY" else ("🟡" if sel_p.status == "WATCHLIST" else "🔴")
        st.markdown(f"### {icon} {sel_p.symbol} — **{sel_p.status}** (Score {sel_p.score})")

        st.write("**Execution Plan**")
        st.write(f"- Entry: **{('—' if sel_p.entry is None else f'${sel_p.entry:.2f}') }**")
        st.write(f"- Stop: **{('—' if sel_p.stop is None else f'${sel_p.stop:.2f}') }**")
        st.write(f"- Shares: **{sel_p.shares}**")
        st.write(f"- Position: **${sel_p.position_value:,.2f}**")
        st.write(f"- Risk (if stopped): **${sel_p.risk_dollars:,.2f}**")

        st.write("**Status Reason**")
        st.info(sel_p.reason_status)

        st.write("**Exit Rule**")
        st.write(f"- {sel_r.exit_rule}")

        st.write("**Key Metrics (Weekly)**")
        m = sel_r.metrics
        def fmt(k, digits=2):
            v = m.get(k)
            if v is None:
                return "—"
            return f"{v:.{digits}f}"

        st.write(f"- Close: {fmt('close')}")
        st.write(f"- MA{cfg['ma_weeks']}: {fmt('ma')}")
        if m.get("macd") is not None and m.get("macd_signal") is not None:
            st.write(f"- MACD: {m.get('macd'):.4f} | Signal: {m.get('macd_signal'):.4f}")
        else:
            st.write("- MACD: —")
        st.write(f"- NATR: {fmt('natr')}%")
        if m.get("box_high") is not None and m.get("box_low") is not None:
            st.write(f"- Box High/Low: {m.get('box_high'):.2f} / {m.get('box_low'):.2f}")
        else:
            st.write("- Box High/Low: —")
        st.write(f"- Weekly Move: {fmt('weekly_move_pct')}%")
        st.write(f"- Upper Wick: {fmt('upper_wick_pct')}%")
        if m.get("vol_spike_pct") is not None:
            st.write(f"- Volume spike vs prior: {m.get('vol_spike_pct'):.1f}%")

    with right:
        st.markdown("### ✅ Passed Gates")
        if sel_r.reasons_pass:
            for x in sel_r.reasons_pass:
                st.success(x)
        else:
            st.info("None.")

        st.markdown("### ❌ Failed Gates (Blockers)")
        if sel_r.reasons_fail:
            for x in sel_r.reasons_fail:
                st.error(x)
        else:
            st.success("None. (Base decision BUY.)")

# =========================
# Beginner help
# =========================
with st.expander("🧠 Beginner Help (FW timing + how to use)", expanded=False):
    st.markdown(
        textwrap.dedent(
            """
            **Correct FW timing**
            - Best: **Friday after close (4:10pm ET+)** → scan → build READY list → execute Monday.
            - Midweek: scan for **WATCHLIST only** (don’t treat signals as confirmed).

            **READY vs WATCHLIST**
            - READY = passes all gates + weekly close confirmed + position sizing fits + portfolio risk cap OK.
            - WATCHLIST = looks good, but not confirmed yet (or sizing/structure not valid).

            **Sizing (professional)**
            - Shares = (Risk $ per trade) ÷ (Entry - Stop)
            - Portfolio cap prevents overexposure across multiple trades.

            **Pro workflow**
            1) Friday close → Run Scan sector-by-sector
            2) Export READY to TradingView
            3) Place orders Monday (or next session)
            4) Manage weekly (do not micromanage daily noise)
            """
        )
    )
