import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import io
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# =========================
# Page
# =========================
st.set_page_config(page_title="Financial Wisdom Scanner (Full Universe)", page_icon="📈", layout="wide")
st.title("📈 Financial Wisdom Scanner — Full Universe (Tech + Fundamentals QC)")
st.caption("Weekly breakouts + consolidation box + risk-defined execution + fundamentals QC. Full S&P 500 / NASDAQ / NYSE / AMEX universes supported.")

# =========================
# HTTP headers (prevents 403/blocks)
# =========================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

# =========================
# Time helpers
# =========================
def now_eastern() -> datetime:
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        pass
    return datetime.now(timezone.utc)

def weekly_close_confirmed() -> bool:
    """
    Conservative: Friday after 4:10pm ET, or Sat/Sun.
    """
    t = now_eastern()
    wd = t.weekday()  # Mon=0..Sun=6
    if wd in (5, 6):
        return True
    if wd == 4 and ((t.hour > 16) or (t.hour == 16 and t.minute >= 10)):
        return True
    return False

# =========================
# Parsing helpers
# =========================
def parse_tickers(text: str) -> List[str]:
    if not text:
        return []
    parts = []
    for chunk in text.replace(",", " ").split():
        t = chunk.strip().upper()
        if t:
            parts.append(t)
    seen = set()
    out = []
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def _dedupe_clean_symbols(symbols: List[str]) -> List[str]:
    out = []
    seen = set()
    for s in symbols:
        s = str(s).strip().upper().replace(".", "-")
        if not s or s in ("N/A", "NA", "NONE", "NAN"):
            continue
        if "^" in s or "/" in s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

# =========================
# Universe loaders (ROBUST)
# =========================
@st.cache_data(show_spinner=False, ttl=60*60*12)
def load_sp500() -> List[str]:
    """
    Robust loader:
    1) GitHub dataset (fast/reliable on Streamlit Cloud)
    2) Wikipedia fallback with requests+UA
    """
    github_urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents_symbols.txt",
    ]
    for url in github_urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            text = r.text
            if url.endswith(".txt"):
                syms = [ln.strip() for ln in text.splitlines() if ln.strip()]
                return _dedupe_clean_symbols(syms)
            df = pd.read_csv(io.StringIO(text))
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            syms = df[col].astype(str).tolist()
            return _dedupe_clean_symbols(syms)
        except Exception:
            pass

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0]
        syms = df["Symbol"].astype(str).tolist()
        return _dedupe_clean_symbols(syms)
    except Exception:
        st.error("Failed to load S&P 500 constituents (network/blocked). Try again later or switch to an Exchange universe.")
        return []

@st.cache_data(show_spinner=False, ttl=60*60*12)
def load_nasdaq100() -> List[str]:
    """
    Robust loader via Wikipedia with requests+UA (graceful failure).
    """
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))

        best = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
                best = t
                break
        if best is None:
            best = tables[0]

        sym_col = None
        for c in best.columns:
            cl = str(c).lower()
            if "ticker" in cl or "symbol" in cl:
                sym_col = c
                break
        if sym_col is None:
            sym_col = best.columns[0]

        syms = best[sym_col].astype(str).tolist()
        syms = [s for s in syms if s.isascii() and len(s) <= 8 and any(ch.isalpha() for ch in s)]
        return _dedupe_clean_symbols(syms)
    except Exception:
        st.error("Failed to load Nasdaq-100 constituents (network/blocked). Try again or use an Exchange universe.")
        return []

@st.cache_data(show_spinner=False, ttl=60*60*12)
def load_russell1000() -> List[str]:
    """
    Russell 1000 via a maintained public list (GitHub raw).
    """
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/russell1000/russell1000.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        col = None
        for c in df.columns:
            if str(c).lower() in ("symbol", "ticker"):
                col = c
                break
        if col is None:
            col = df.columns[0]
        syms = df[col].astype(str).tolist()
        syms = [s for s in syms if s.isascii() and len(s) <= 8 and any(ch.isalpha() for ch in s)]
        return _dedupe_clean_symbols(syms)
    except Exception:
        st.error("Failed to load Russell 1000 list. Try again later or use Exchange universe.")
        return []

@st.cache_data(show_spinner=False, ttl=60*60*12)
def load_exchange_list(exchange: str) -> List[str]:
    """
    Full US symbol directories from NasdaqTrader (pipe-delimited).
    exchange: 'NASDAQ' | 'NYSE' | 'AMEX' | 'ALL'
    """
    listed_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    other_url  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    def fetch_txt(url: str) -> pd.DataFrame:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        lines = r.text.splitlines()
        lines = [ln for ln in lines if ln and not ln.startswith("File Creation Time")]
        data = "\n".join(lines)
        return pd.read_csv(io.StringIO(data), sep="|")

    try:
        df_nas = fetch_txt(listed_url)
        df_oth = fetch_txt(other_url)
    except Exception:
        st.error("Failed to load exchange symbol lists (network/blocked). Try again later.")
        return []

    syms: List[str] = []
    if exchange == "NASDAQ":
        syms = df_nas["Symbol"].astype(str).tolist()
    elif exchange in ("NYSE", "AMEX"):
        if "ACT Symbol" in df_oth.columns and "Exchange" in df_oth.columns:
            syms = df_oth.loc[df_oth["Exchange"].astype(str).str.upper() == exchange, "ACT Symbol"].astype(str).tolist()
        else:
            syms = df_oth.iloc[:, 0].astype(str).tolist()
    elif exchange == "ALL":
        a = df_nas["Symbol"].astype(str).tolist()
        b = df_oth["ACT Symbol"].astype(str).tolist() if "ACT Symbol" in df_oth.columns else df_oth.iloc[:, 0].astype(str).tolist()
        syms = a + b
    else:
        syms = df_nas["Symbol"].astype(str).tolist()

    out = []
    for s in syms:
        s = str(s).strip().upper().replace(".", "-")
        if not s or s in ("N/A", "NA"):
            continue
        if "^" in s or "/" in s:
            continue
        out.append(s)
    out = [s for s in out if s.isascii() and len(s) <= 8 and any(ch.isalpha() for ch in s)]
    return _dedupe_clean_symbols(out)

# =========================
# Technical scan core (FW)
# =========================
@dataclass
class ScanResult:
    symbol: str
    decision_base: str     # BUY or PASS (technical gates only)
    score_tech: int
    entry: Optional[float]
    stop: Optional[float]
    risk_pct: Optional[float]
    reasons_pass: List[str]
    reasons_fail: List[str]
    metrics: Dict[str, Optional[float]]

def compute_sma(close: pd.Series, length: int) -> Optional[float]:
    try:
        s = ta.sma(close, length=length)
        return float(s.iloc[-1]) if s is not None and len(s) else None
    except Exception:
        return None

def compute_macd(close: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    try:
        m = ta.macd(close)
        if m is None or m.empty:
            return None, None
        cols = list(m.columns)
        if len(cols) < 2:
            return None, None
        return float(m[cols[0]].iloc[-1]), float(m[cols[1]].iloc[-1])
    except Exception:
        return None, None

def compute_natr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Optional[float]:
    try:
        n = ta.natr(high, low, close, length=length)
        return float(n.iloc[-1]) if n is not None and len(n) else None
    except Exception:
        return None

def consolidation_box(df: pd.DataFrame, weeks: int) -> Optional[Tuple[float, float]]:
    if df is None or len(df) < weeks + 2:
        return None
    prior = df.iloc[:-1].tail(weeks)
    if prior.empty:
        return None
    return float(prior["High"].max()), float(prior["Low"].min())

def prior_high_close(df: pd.DataFrame, lookback: int) -> Optional[float]:
    if df is None or len(df) < lookback + 1:
        return None
    prior = df["Close"].iloc[:-1].tail(lookback)
    if prior.empty:
        return None
    return float(prior.max())

def candle_quality(curr: pd.Series, prev: Optional[pd.Series]) -> Tuple[Optional[float], Optional[float]]:
    try:
        o, h, l, c = float(curr["Open"]), float(curr["High"]), float(curr["Low"]), float(curr["Close"])
        rng = max(1e-9, h - l)
        upper_wick = h - max(o, c)
        upper_wick_pct = (upper_wick / rng) * 100.0
        move_pct = None
        if prev is not None:
            prev_c = float(prev["Close"])
            if prev_c != 0:
                move_pct = ((c - prev_c) / prev_c) * 100.0
        return move_pct, upper_wick_pct
    except Exception:
        return None, None

def eval_fw_technical(symbol: str, df: pd.DataFrame, cfg: dict) -> ScanResult:
    rp, rf = [], []
    if df is None or df.empty or len(df) < cfg["min_weeks_history"]:
        rf.append(f"Not enough weekly history (need ≥ {cfg['min_weeks_history']}).")
        return ScanResult(symbol, "PASS", 0, None, None, None, rp, rf, {})

    curr = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None
    close = float(curr["Close"])

    ma = compute_sma(df["Close"], cfg["ma_weeks"])
    if ma is None:
        rf.append(f"Cannot compute MA{cfg['ma_weeks']}.")
    elif close >= ma:
        rp.append(f"Close above MA{cfg['ma_weeks']}.")
    else:
        rf.append(f"Close below MA{cfg['ma_weeks']}.")

    macd_line, macd_sig = compute_macd(df["Close"])
    if macd_line is None or macd_sig is None:
        rf.append("Cannot compute MACD.")
    elif macd_line >= macd_sig:
        rp.append("MACD line ≥ signal.")
    else:
        rf.append("MACD line < signal.")

    natr = compute_natr(df["High"], df["Low"], df["Close"], 14)
    if natr is None:
        rf.append("Cannot compute NATR.")
    elif natr < cfg["natr_max"]:
        rp.append(f"NATR {natr:.2f}% < {cfg['natr_max']}%.")
    else:
        rf.append(f"NATR {natr:.2f}% ≥ {cfg['natr_max']}%.")

    box = consolidation_box(df, cfg["consolidation_weeks"])
    box_high = box_low = None
    if box is None:
        rf.append(f"Cannot compute consolidation box ({cfg['consolidation_weeks']}w).")
    else:
        box_high, box_low = box
        rp.append("Consolidation box computed.")

    phc = prior_high_close(df, cfg["lookback_high_weeks"])
    if phc is None:
        rf.append(f"Cannot compute prior {cfg['lookback_high_weeks']}w closing high.")
    elif close > phc:
        rp.append(f"Close > prior {cfg['lookback_high_weeks']}w closing high.")
    else:
        rf.append(f"Close not above prior {cfg['lookback_high_weeks']}w closing high.")

    move_pct, wick_pct = candle_quality(curr, prev)
    if move_pct is None:
        rf.append("Cannot compute weekly move %.")  # rare
    else:
        if cfg["min_weekly_move_pct"] <= move_pct <= cfg["max_weekly_move_pct"]:
            rp.append(f"Weekly move {move_pct:.2f}% within range.")
        else:
            rf.append(f"Weekly move {move_pct:.2f}% outside range.")

    if wick_pct is None:
        rf.append("Cannot compute upper wick %.")  # rare
    else:
        if wick_pct <= cfg["max_upper_wick_pct"]:
            rp.append(f"Upper wick {wick_pct:.2f}% OK.")
        else:
            rf.append(f"Upper wick {wick_pct:.2f}% too high.")

    if box_high is not None:
        if close > box_high:
            rp.append(f"Close above box high ({box_high:.2f}).")
        else:
            rf.append(f"Close not above box high ({box_high:.2f}).")

    vol_spike_pct = None
    if cfg["require_volume_spike"] and prev is not None:
        pv = float(prev["Volume"])
        cv = float(curr["Volume"])
        if pv > 0:
            vol_spike_pct = ((cv - pv) / pv) * 100.0
            if vol_spike_pct >= cfg["vol_spike_min_pct"]:
                rp.append(f"Volume spike {vol_spike_pct:.1f}% ≥ {cfg['vol_spike_min_pct']}%.")
            else:
                rf.append(f"Volume spike {vol_spike_pct:.1f}% < {cfg['vol_spike_min_pct']}%.")

    entry = close
    stop = None
    risk_pct = None
    if box_low is not None:
        stop = float(box_low)
        risk_pct = ((entry - stop) / entry) * 100.0
        if risk_pct <= cfg["max_stop_pct"]:
            rp.append(f"Stop risk {risk_pct:.2f}% ≤ {cfg['max_stop_pct']}%.")
        else:
            rf.append(f"Stop risk {risk_pct:.2f}% > {cfg['max_stop_pct']}%.")

    passes = len(rp)
    fails = len(rf)
    score = int(round((passes / max(1, passes + fails)) * 100))
    decision = "BUY" if fails == 0 else "PASS"

    metrics = {
        "close": close,
        "ma": ma,
        "macd": macd_line,
        "macd_signal": macd_sig,
        "natr": natr,
        "weekly_move_pct": move_pct,
        "upper_wick_pct": wick_pct,
        "prior_high_close": phc,
        "box_high": box_high,
        "box_low": box_low,
        "vol_spike_pct": vol_spike_pct,
    }
    return ScanResult(symbol, decision, score, entry, stop, risk_pct, rp, rf, metrics)

# =========================
# Fundamentals QC (Phase 2)
# =========================
@dataclass
class FundamentalResult:
    symbol: str
    ok: bool
    fund_score: int
    qc_fail_reasons: List[str]
    fields: Dict[str, Optional[float]]

def _safe_num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip()
        if s.lower() in ("nan", "none", "", "n/a"):
            return None
        return float(s)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60*60*6)
def fetch_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    fields = {
        "marketCap": None,
        "averageVolume": None,
        "trailingPE": None,
        "forwardPE": None,
        "profitMargins": None,
        "operatingMargins": None,
        "returnOnEquity": None,
        "totalDebt": None,
        "totalCash": None,
        "debtToEquity": None,
        "revenueGrowth": None,
        "earningsGrowth": None,
        "freeCashflow": None,
    }
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "info", {}) or {}
        for k in list(fields.keys()):
            if k in info:
                fields[k] = _safe_num(info.get(k))
        return fields
    except Exception:
        return fields

def fundamentals_qc(symbol: str, fields: Dict[str, Optional[float]], qc: dict) -> FundamentalResult:
    reasons = []
    mcap = fields.get("marketCap")
    avgv = fields.get("averageVolume")
    opm = fields.get("operatingMargins")
    pm  = fields.get("profitMargins")
    revg = fields.get("revenueGrowth")
    de  = fields.get("debtToEquity")
    fcf = fields.get("freeCashflow")

    if mcap is not None and mcap < qc["min_market_cap"]:
        reasons.append(f"Market cap < ${qc['min_market_cap']/1e9:.1f}B")

    if avgv is not None and avgv < qc["min_avg_volume"]:
        reasons.append(f"Avg volume < {qc['min_avg_volume']:,}/day")

    if qc["require_profitability"]:
        ok_profit = False
        if opm is not None and opm > 0:
            ok_profit = True
        if pm is not None and pm > 0:
            ok_profit = True
        if fcf is not None and fcf > 0:
            ok_profit = True
        if not ok_profit:
            reasons.append("No positive profitability/FCF signal")

    if qc["max_debt_to_equity"] is not None and de is not None and de > qc["max_debt_to_equity"]:
        reasons.append(f"Debt/Equity > {qc['max_debt_to_equity']}")

    ok = len(reasons) == 0

    score = 0
    if mcap is not None:
        score += 20 if mcap >= qc["min_market_cap"] else 5
    if avgv is not None:
        score += 20 if avgv >= qc["min_avg_volume"] else 5
    if revg is not None:
        score += 20 if revg > 0 else 5

    best_margin = None
    if opm is not None:
        best_margin = opm
    if pm is not None:
        best_margin = max(best_margin, pm) if best_margin is not None else pm
    if best_margin is not None:
        score += 20 if best_margin > 0 else 5

    if de is not None:
        score += 20 if de <= (qc["max_debt_to_equity"] if qc["max_debt_to_equity"] is not None else de) else 5

    # if some fields are missing, you still end up ~40–60, not 0
    score = int(max(0, min(100, score)))

    return FundamentalResult(symbol=symbol, ok=ok, fund_score=score, qc_fail_reasons=reasons, fields=fields)

# =========================
# Trading / sizing helpers
# =========================
def tv_symbol(symbol: str, prefix: str) -> str:
    return f"{prefix}:{symbol}"

@dataclass
class FinalRow:
    symbol: str
    status: str
    score_tech: int
    score_fund: int
    score_total: int
    entry: Optional[float]
    stop: Optional[float]
    stop_risk_pct: Optional[float]
    shares: int
    position_value: float
    risk_dollars: float
    close: Optional[float]
    natr: Optional[float]
    box_high: Optional[float]
    box_low: Optional[float]
    reason: str

# =========================
# Sidebar controls
# =========================
st.sidebar.header("🧰 Universe (Full Exchange Supported)")

universe_mode = st.sidebar.selectbox(
    "Universe Source",
    [
        "AUTO: S&P 500",
        "AUTO: Nasdaq-100",
        "AUTO: Russell 1000",
        "AUTO: NASDAQ (All)",
        "AUTO: NYSE (All)",
        "AUTO: AMEX (All)",
        "AUTO: ALL US (NASDAQ+NYSE+AMEX)",
        "Paste tickers",
        "Upload tickers file",
    ],
    index=0
)

exchange_prefix = st.sidebar.selectbox("TradingView prefix for export", ["NASDAQ", "NYSE", "AMEX", "CBOE", "TVC", "FX", "CRYPTO"], index=0)

st.sidebar.divider()
st.sidebar.header("⚙️ FW Technical Rules (Weekly)")

cfg = {}
cfg["min_weeks_history"] = st.sidebar.number_input("Min weeks history", 40, 200, 80, 5)
cfg["ma_weeks"] = st.sidebar.number_input("MA weeks", 10, 50, 20, 1)
cfg["lookback_high_weeks"] = st.sidebar.number_input("Breakout vs prior closing high (weeks)", 5, 30, 10, 1)
cfg["consolidation_weeks"] = st.sidebar.number_input("Consolidation box weeks", 6, 30, 12, 1)
cfg["natr_max"] = st.sidebar.number_input("NATR max (%)", 1.0, 25.0, 8.0, 0.5)
cfg["min_weekly_move_pct"] = st.sidebar.number_input("Min breakout weekly move (%)", 0.0, 50.0, 5.0, 0.5)
cfg["max_weekly_move_pct"] = st.sidebar.number_input("Max breakout weekly move (%)", 1.0, 100.0, 20.0, 0.5)
cfg["max_upper_wick_pct"] = st.sidebar.number_input("Max upper wick (%)", 0.0, 100.0, 50.0, 1.0)
cfg["require_volume_spike"] = st.sidebar.checkbox("Require volume spike vs prior week", value=True)
cfg["vol_spike_min_pct"] = st.sidebar.number_input("Volume spike min (%)", -50.0, 500.0, 0.0, 5.0)
cfg["max_stop_pct"] = st.sidebar.number_input("Max stop risk (%)", 5.0, 50.0, 20.0, 0.5)

st.sidebar.divider()
st.sidebar.header("📊 Fundamentals QC (FW Materials Inspection)")

qc = {}
qc["enable_fundamentals"] = st.sidebar.checkbox("Enable fundamentals QC + scoring (recommended)", value=True)
qc["min_market_cap"] = st.sidebar.selectbox("Min market cap", ["$1B", "$2B", "$5B", "$10B"], index=1)
qc["min_market_cap"] = {"$1B":1e9,"$2B":2e9,"$5B":5e9,"$10B":1e10}[qc["min_market_cap"]]
qc["min_avg_volume"] = st.sidebar.selectbox("Min avg shares/day", ["200k","500k","1M","2M"], index=2)
qc["min_avg_volume"] = {"200k":200_000,"500k":500_000,"1M":1_000_000,"2M":2_000_000}[qc["min_avg_volume"]]
qc["require_profitability"] = st.sidebar.checkbox("Require profitability/FCF signal", value=True)
qc["max_debt_to_equity"] = st.sidebar.selectbox("Max Debt/Equity (if available)", ["No limit", "2.0", "1.5", "1.0"], index=1)
qc["max_debt_to_equity"] = None if qc["max_debt_to_equity"] == "No limit" else float(qc["max_debt_to_equity"])
fund_weight = st.sidebar.slider("Fundamentals weight (Total Score)", 0, 40, 20, 5)
tech_weight = 100 - fund_weight
st.sidebar.caption(f"Total Score = {tech_weight}% Tech + {fund_weight}% Fundamentals")

st.sidebar.divider()
st.sidebar.header("💰 Risk + Execution")

account_size = st.sidebar.number_input("Account size ($)", min_value=1000, value=25000, step=1000)
risk_pct_per_trade = st.sidebar.selectbox("Risk per trade (%)", [0.25, 0.5, 0.75, 1.0], index=1)
risk_dollars_per_trade = account_size * (risk_pct_per_trade / 100.0)
max_portfolio_risk_pct = st.sidebar.selectbox("Max total new-trade risk (%)", [2.0, 3.0, 5.0, 7.5, 10.0], index=2)
max_portfolio_risk_dollars = account_size * (max_portfolio_risk_pct / 100.0)
max_new_positions = st.sidebar.slider("Max new positions", 1, 50, 10, 1)

enforce_weekly_close = st.sidebar.checkbox("Enforce weekly-close confirmation (READY only after Fri close)", value=True)
confirmed_now = weekly_close_confirmed() if enforce_weekly_close else True
st.sidebar.caption(f"Weekly close confirmed now? **{'YES' if confirmed_now else 'NO'}**")

st.sidebar.divider()
st.sidebar.header("⚡ Performance / Scale Controls")

max_tickers = st.sidebar.number_input("Max tickers to scan (safety)", min_value=50, value=800, step=50)
batch_size = st.sidebar.number_input("Batch size (download chunk)", min_value=20, value=80, step=10)
max_workers = st.sidebar.slider("Parallel batches", 1, 12, 4, 1)
fund_top_n = st.sidebar.number_input("Run fundamentals on top N candidates", min_value=50, value=250, step=50)
st.sidebar.caption("Full-exchange scans are heavy. This design scans tech first, then fundamentals only on the shortlist.")

# =========================
# Universe selection
# =========================
tickers: List[str] = []

if universe_mode == "AUTO: S&P 500":
    tickers = load_sp500()
elif universe_mode == "AUTO: Nasdaq-100":
    tickers = load_nasdaq100()
elif universe_mode == "AUTO: Russell 1000":
    tickers = load_russell1000()
elif universe_mode == "AUTO: NASDAQ (All)":
    tickers = load_exchange_list("NASDAQ")
elif universe_mode == "AUTO: NYSE (All)":
    tickers = load_exchange_list("NYSE")
elif universe_mode == "AUTO: AMEX (All)":
    tickers = load_exchange_list("AMEX")
elif universe_mode == "AUTO: ALL US (NASDAQ+NYSE+AMEX)":
    tickers = load_exchange_list("ALL")
elif universe_mode == "Paste tickers":
    pasted = st.sidebar.text_area("Paste tickers", value="AAPL MSFT NVDA AMD GOOGL META", height=120)
    tickers = parse_tickers(pasted)
else:
    up = st.sidebar.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
    if up is not None:
        raw = up.read()
        try:
            dfu = pd.read_csv(io.BytesIO(raw))
            col = dfu.columns[0]
            if "symbol" in [c.lower() for c in dfu.columns.astype(str).tolist()]:
                for c in dfu.columns:
                    if str(c).lower() == "symbol":
                        col = c
                        break
            tickers = parse_tickers("\n".join(dfu[col].astype(str).tolist()))
        except Exception:
            tickers = parse_tickers(raw.decode("utf-8", errors="ignore"))

tickers = [t for t in tickers if t]
tickers = list(dict.fromkeys(tickers))  # stable dedupe

if len(tickers) > int(max_tickers):
    tickers = tickers[: int(max_tickers)]

# =========================
# Top UI
# =========================
st.write("")
topA, topB, topC = st.columns([1.2, 1.0, 2.2])
with topA:
    run = st.button("🚀 Run Full Scan", use_container_width=True)
with topB:
    show_ready_only = st.checkbox("Show only READY", value=False)
with topC:
    search = st.text_input("Search ticker", value="").strip().upper()

st.write(f"**Universe loaded:** {len(tickers)} tickers")

if not run:
    st.info("Click **Run Full Scan** to scan the universe.")
    st.stop()

# =========================
# Phase 1: Batch download weekly OHLCV
# =========================
st.subheader("1) Technical Scan (Weekly FW Breakouts)")
progress = st.progress(0)
status = st.empty()

results: List[ScanResult] = []

def download_batch(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not symbols:
        return out
    s = " ".join(symbols)
    try:
        data = yf.download(
            tickers=s,
            period="5y",
            interval="1wk",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False
        )
        if data is None or data.empty:
            return out

        if isinstance(data.columns, pd.MultiIndex):
            for sym in symbols:
                if sym in data.columns.get_level_values(0):
                    dfw = data[sym].dropna()
                    if not dfw.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(dfw.columns):
                        out[sym] = dfw.copy()
        else:
            dfw = data.dropna()
            if not dfw.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(dfw.columns):
                out[symbols[0]] = dfw.copy()
    except Exception:
        return out
    return out

batches = chunked(tickers, int(batch_size))
total_batches = len(batches)

def process_batch(symbols: List[str]) -> List[ScanResult]:
    dfs = download_batch(symbols)
    local: List[ScanResult] = []
    for sym in symbols:
        dfw = dfs.get(sym)
        if dfw is None or dfw.empty:
            local.append(ScanResult(sym, "PASS", 0, None, None, None, [], [f"No weekly data for {sym}"], {}))
        else:
            local.append(eval_fw_technical(sym, dfw, cfg))
    return local

done = 0
with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
    futs = {ex.submit(process_batch, b): b for b in batches}
    for fut in as_completed(futs):
        try:
            batch_res = fut.result()
        except Exception:
            batch_res = []
        results.extend(batch_res)
        done += 1
        progress.progress(done / max(1, total_batches))
        status.write(f"Scanning batches… {done}/{total_batches}")

status.write("✅ Technical scan complete.")
progress.empty()

results.sort(key=lambda r: (0 if r.decision_base == "BUY" else 1, -r.score_tech, r.symbol))

shortlist_syms = [r.symbol for r in results if r.decision_base == "BUY"]
if len(shortlist_syms) < int(fund_top_n):
    add = [r.symbol for r in results if r.symbol not in shortlist_syms][: (int(fund_top_n) - len(shortlist_syms))]
    shortlist_syms.extend(add)
shortlist_syms = shortlist_syms[: int(fund_top_n)]

# =========================
# Phase 2: Fundamentals QC (Shortlist only)
# =========================
st.subheader("2) Fundamentals QC + Scoring (Applied to Shortlist)")
fund_map: Dict[str, FundamentalResult] = {}

if qc["enable_fundamentals"]:
    fprog = st.progress(0)
    fstatus = st.empty()

    def one_f(sym: str) -> FundamentalResult:
        fields = fetch_fundamentals(sym)
        return fundamentals_qc(sym, fields, qc)

    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(one_f, s): s for s in shortlist_syms}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                fr = fut.result()
            except Exception:
                fr = FundamentalResult(sym, ok=True, fund_score=50, qc_fail_reasons=["Fundamentals fetch failed"], fields={})
            fund_map[sym] = fr
            done += 1
            fprog.progress(done / max(1, len(shortlist_syms)))
            fstatus.write(f"Fetching fundamentals… {done}/{len(shortlist_syms)}")

    fstatus.write("✅ Fundamentals QC complete.")
    fprog.empty()
else:
    st.info("Fundamentals QC is OFF. (You can enable it in the sidebar.)")

# =========================
# Final decision + position sizing + portfolio risk cap
# =========================
st.subheader("3) Execution Output (READY / WATCHLIST / PASS / QC_FAIL)")

portfolio_risk_used = 0.0
new_positions_used = 0
rows: List[FinalRow] = []

for r in results:
    if search and search not in r.symbol.upper():
        continue

    fr = fund_map.get(r.symbol)
    fund_score = 50
    qc_reason = ""
    if qc["enable_fundamentals"]:
        if fr is not None:
            fund_score = fr.fund_score
            if not fr.ok:
                qc_reason = " | ".join(fr.qc_fail_reasons)

    total_score = int(round((r.score_tech * (tech_weight / 100.0)) + (fund_score * (fund_weight / 100.0))))

    if r.decision_base != "BUY":
        status_lbl = "PASS"
        reason = "Failed one or more technical FW gates."
    else:
        if not confirmed_now:
            status_lbl = "WATCHLIST"
            reason = "Weekly close not confirmed yet. Confirm after Friday close."
        else:
            status_lbl = "WATCHLIST"
            reason = "Meets technical gates; pending sizing/QC."

    if r.decision_base == "BUY" and qc["enable_fundamentals"] and (fr is not None) and (not fr.ok) and confirmed_now:
        status_lbl = "QC_FAIL"
        reason = f"Fundamentals QC fail: {qc_reason}"

    shares = 0
    position_value = 0.0
    risk_dollars = 0.0

    if r.decision_base == "BUY" and confirmed_now and status_lbl not in ("QC_FAIL", "PASS"):
        if r.entry is not None and r.stop is not None and r.entry > r.stop:
            per_share_risk = r.entry - r.stop
            shares = int(risk_dollars_per_trade // per_share_risk)
            if shares <= 0:
                status_lbl = "SKIP_NO_SHARES"
                reason = "Per-share risk too large for your risk-per-trade."
            else:
                position_value = shares * r.entry
                risk_dollars = shares * per_share_risk

                if new_positions_used >= int(max_new_positions):
                    status_lbl = "SKIP_RISK_CAP"
                    reason = "Max new positions reached."
                elif (portfolio_risk_used + risk_dollars) > max_portfolio_risk_dollars:
                    status_lbl = "SKIP_RISK_CAP"
                    reason = "Would exceed portfolio risk cap."
                else:
                    status_lbl = "READY"
                    reason = "Meets FW tech + timing + (optional) fundamentals + sizing + risk cap."
                    portfolio_risk_used += risk_dollars
                    new_positions_used += 1
        else:
            status_lbl = "WATCHLIST"
            reason = "No valid structured stop for sizing."

    if show_ready_only and status_lbl != "READY":
        continue

    m = r.metrics or {}
    rows.append(FinalRow(
        symbol=r.symbol,
        status=status_lbl,
        score_tech=r.score_tech,
        score_fund=fund_score,
        score_total=total_score,
        entry=r.entry,
        stop=r.stop,
        stop_risk_pct=r.risk_pct,
        shares=shares,
        position_value=position_value,
        risk_dollars=risk_dollars,
        close=m.get("close"),
        natr=m.get("natr"),
        box_high=m.get("box_high"),
        box_low=m.get("box_low"),
        reason=reason
    ))

order = {"READY": 0, "WATCHLIST": 1, "QC_FAIL": 2, "SKIP_RISK_CAP": 3, "SKIP_NO_SHARES": 4, "PASS": 5}
rows.sort(key=lambda x: (order.get(x.status, 9), -x.score_total, x.symbol))

# =========================
# TradingView export
# =========================
ready_syms = [r.symbol for r in rows if r.status == "READY"]
tv_lines = "\n".join(tv_symbol(s, exchange_prefix) for s in ready_syms)
tv_commas = ",".join(tv_symbol(s, exchange_prefix) for s in ready_syms)

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.0])
c1.metric("Universe scanned", len(tickers))
c2.metric("READY", len(ready_syms))
c3.metric("WATCHLIST", sum(1 for r in rows if r.status == "WATCHLIST"))
c4.metric("New-trade risk used", f"${portfolio_risk_used:,.0f} / ${max_portfolio_risk_dollars:,.0f}")

fmt = st.radio("TradingView export format", ["Newline-separated", "Comma-separated"], horizontal=True)
export_text = tv_lines if fmt == "Newline-separated" else tv_commas
st.text_area("Copy into TradingView watchlist / notes", export_text, height=90)
st.download_button("⬇️ Download READY_watchlist.txt", data=export_text.encode("utf-8"), file_name="READY_watchlist.txt", mime="text/plain", use_container_width=True)

# =========================
# Results table
# =========================
df = pd.DataFrame([{
    "Symbol": r.symbol,
    "Status": r.status,
    "Tech": r.score_tech,
    "Fund": r.score_fund,
    "Total": r.score_total,
    "Entry": None if r.entry is None else round(r.entry, 2),
    "Stop": None if r.stop is None else round(r.stop, 2),
    "Stop Risk %": None if r.stop_risk_pct is None else round(r.stop_risk_pct, 2),
    "Shares": r.shares,
    "Position $": round(r.position_value, 2),
    "Risk $": round(r.risk_dollars, 2),
    "Close": None if r.close is None else round(r.close, 2),
    "NATR%": None if r.natr is None else round(r.natr, 2),
    "Box High": None if r.box_high is None else round(r.box_high, 2),
    "Box Low": None if r.box_low is None else round(r.box_low, 2),
    "Reason": r.reason
} for r in rows])

st.dataframe(df, use_container_width=True, height=520)

# =========================
# Explainability panel
# =========================
st.subheader("🔎 Why picked? (Explainability)")

if len(rows) > 0:
    pick = st.selectbox("Select ticker", options=[r.symbol for r in rows], index=0)
    rr = next((x for x in results if x.symbol == pick), None)
    fr = fund_map.get(pick)

    L, R = st.columns([1.1, 1.4])
    with L:
        row = next((x for x in rows if x.symbol == pick), None)
        icon = "🟢" if row.status == "READY" else ("🟡" if row.status == "WATCHLIST" else "🔴")
        st.markdown(f"### {icon} {pick} — **{row.status}** (Total {row.score_total})")
        st.write("**Execution**")
        st.write(f"- Entry: {('—' if row.entry is None else f'${row.entry:.2f}')}")
        st.write(f"- Stop: {('—' if row.stop is None else f'${row.stop:.2f}')}")
        st.write(f"- Shares: **{row.shares}**")
        st.write(f"- Position: **${row.position_value:,.2f}**")
        st.write(f"- Risk if stopped: **${row.risk_dollars:,.2f}**")
        st.write("**Reason**")
        st.info(row.reason)

        if fr is not None and qc["enable_fundamentals"]:
            st.write("**Fundamentals QC**")
            if fr.ok:
                st.success(f"QC PASS | Fund score: {fr.fund_score}")
            else:
                st.error(f"QC FAIL | Fund score: {fr.fund_score} | " + " | ".join(fr.qc_fail_reasons))
            st.write(fr.fields)

    with R:
        st.markdown("### ✅ Passed Technical Gates")
        if rr and rr.reasons_pass:
            for x in rr.reasons_pass:
                st.success(x)
        else:
            st.info("None")

        st.markdown("### ❌ Failed Technical Gates")
        if rr and rr.reasons_fail:
            for x in rr.reasons_fail:
                st.error(x)
        else:
            st.success("None (Technical BUY).")

# =========================
# Journal export
# =========================
st.subheader("🧾 Journal Export (READY)")
timestamp = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
journal = df[df["Status"] == "READY"].copy()
journal.insert(0, "Timestamp", timestamp)
csv = journal.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download READY_journal.csv", data=csv, file_name="READY_journal.csv", mime="text/csv", use_container_width=True)

with st.expander("🧠 How to run FW correctly (timing)", expanded=False):
    st.markdown("""
- **Friday after close (4:10pm ET+)**: run scans → focus on **READY** list.
- **Weekend**: review charts in TradingView → plan orders.
- **Monday (next session)**: execute entries (limit orders around breakout level).
- **Midweek scans**: treat as **WATCHLIST**, not confirmed signals.
""")
