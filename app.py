import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Wisdom Exact Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Wisdom Exact Breakout Scanner")
st.caption(
    "Pure FW weekly breakout scanner: 20W MA + MACD + NATR + consolidation breakout + volume + candle quality + quality fundamentals."
)


# ============================================================
# CONSTANTS — TRUE FW RULES
# ============================================================

FW_MA_WEEKS = 20
FW_MIN_CONSOLIDATION_WEEKS = 6
FW_HIGH_CLOSE_LOOKBACK = 10
FW_NATR_MAX = 8.0
FW_MIN_VOLUME_SPIKE_PCT = 30.0
FW_MIN_BREAKOUT_PCT = 5.0
FW_MAX_BREAKOUT_PCT = 20.0
FW_MAX_UPPER_WICK_PCT = 50.0
FW_MAX_STOP_RISK_PCT = 20.0

FW_MIN_ROC = 0.10
FW_MIN_ROE = 0.10
FW_MIN_OPERATING_MARGIN = 0.10

FW_POSITION_SIZE_PCT = 0.16  # FW example uses 16% position size from fractional Kelly


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120 Safari/537.36"
    )
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TechnicalResult:
    symbol: str
    technical_pass: bool
    entry: Optional[float]
    stop: Optional[float]
    stop_risk_pct: Optional[float]
    metrics: Dict[str, Optional[float]]
    passed: List[str]
    failed: List[str]


@dataclass
class FundamentalResult:
    symbol: str
    fundamental_pass: bool
    metrics: Dict[str, Optional[float]]
    passed: List[str]
    failed: List[str]


@dataclass
class FinalResult:
    symbol: str
    status: str
    entry: Optional[float]
    stop: Optional[float]
    stop_risk_pct: Optional[float]
    position_value: float
    shares: int
    risk_dollars: float
    risk_on_equity_pct: float
    technical: TechnicalResult
    fundamental: FundamentalResult


# ============================================================
# TIME HELPERS
# ============================================================

def now_eastern() -> datetime:
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        pass
    return datetime.now(timezone.utc)


def weekly_close_confirmed() -> bool:
    """
    Official FW signal should be confirmed after the weekly candle closes.
    Conservative rule:
    - Friday after 4:10 PM ET
    - Saturday
    - Sunday
    """
    t = now_eastern()
    wd = t.weekday()  # Monday=0, Sunday=6

    if wd in (5, 6):
        return True

    if wd == 4 and ((t.hour > 16) or (t.hour == 16 and t.minute >= 10)):
        return True

    return False


# ============================================================
# SYMBOL / UNIVERSE HELPERS
# ============================================================

def clean_symbols(symbols: List[str]) -> List[str]:
    cleaned = []
    seen = set()

    for s in symbols:
        sym = str(s).strip().upper().replace(".", "-")

        if not sym:
            continue

        if sym in ("N/A", "NA", "NONE", "NAN"):
            continue

        if "^" in sym or "/" in sym:
            continue

        if len(sym) > 8:
            continue

        if not any(ch.isalpha() for ch in sym):
            continue

        if sym not in seen:
            cleaned.append(sym)
            seen.add(sym)

    return cleaned


def parse_tickers(text: str) -> List[str]:
    if not text:
        return []

    raw = text.replace(",", " ").split()
    return clean_symbols(raw)


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_sp500() -> List[str]:
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents_symbols.txt",
    ]

    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()

            if url.endswith(".txt"):
                return clean_symbols(r.text.splitlines())

            df = pd.read_csv(io.StringIO(r.text))
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            return clean_symbols(df[col].astype(str).tolist())

        except Exception:
            pass

    st.error("Could not load S&P 500 list. Try another universe or paste tickers manually.")
    return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_nasdaq100() -> List[str]:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()

        tables = pd.read_html(io.StringIO(r.text))

        selected = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                selected = t
                break

        if selected is None:
            selected = tables[0]

        sym_col = None
        for c in selected.columns:
            cl = str(c).lower()
            if "ticker" in cl or "symbol" in cl:
                sym_col = c
                break

        if sym_col is None:
            sym_col = selected.columns[0]

        return clean_symbols(selected[sym_col].astype(str).tolist())

    except Exception:
        st.error("Could not load Nasdaq-100 list. Try another universe or paste tickers manually.")
        return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_exchange_list(exchange: str) -> List[str]:
    """
    Loads NASDAQ / NYSE / AMEX / ALL from NasdaqTrader symbol directories.
    """
    listed_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    def fetch_pipe_file(url: str) -> pd.DataFrame:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()

        lines = [
            line for line in r.text.splitlines()
            if line and not line.startswith("File Creation Time")
        ]

        return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")

    try:
        nasdaq_df = fetch_pipe_file(listed_url)
        other_df = fetch_pipe_file(other_url)
    except Exception:
        st.error("Could not load exchange symbol lists. Try again later or paste tickers manually.")
        return []

    symbols = []

    if exchange == "NASDAQ":
        symbols = nasdaq_df["Symbol"].astype(str).tolist()

    elif exchange in ("NYSE", "AMEX"):
        if "ACT Symbol" in other_df.columns and "Exchange" in other_df.columns:
            symbols = other_df.loc[
                other_df["Exchange"].astype(str).str.upper() == exchange,
                "ACT Symbol"
            ].astype(str).tolist()

    elif exchange == "ALL":
        nasdaq_symbols = nasdaq_df["Symbol"].astype(str).tolist()
        other_symbols = (
            other_df["ACT Symbol"].astype(str).tolist()
            if "ACT Symbol" in other_df.columns
            else []
        )
        symbols = nasdaq_symbols + other_symbols

    return clean_symbols(symbols)


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def macd_line_and_signal(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    return macd_line, signal_line


def true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()

    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).rolling(length).mean()


def natr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return (atr(df, length) / df["Close"]) * 100.0


# ============================================================
# MARKET DATA
# ============================================================

def download_weekly_batch(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Downloads weekly OHLCV for a list of symbols using yfinance.
    """
    result = {}

    if not symbols:
        return result

    try:
        data = yf.download(
            tickers=" ".join(symbols),
            period="5y",
            interval="1wk",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False
        )

        if data is None or data.empty:
            return result

        required = {"Open", "High", "Low", "Close", "Volume"}

        if isinstance(data.columns, pd.MultiIndex):
            available_symbols = set(data.columns.get_level_values(0))

            for sym in symbols:
                if sym not in available_symbols:
                    continue

                df = data[sym].dropna().copy()

                if required.issubset(set(df.columns)) and not df.empty:
                    result[sym] = df

        else:
            df = data.dropna().copy()

            if required.issubset(set(df.columns)) and not df.empty:
                result[symbols[0]] = df

    except Exception:
        pass

    return result


# ============================================================
# TRUE FW TECHNICAL SCAN
# ============================================================

def evaluate_technical(symbol: str, df: Optional[pd.DataFrame]) -> TechnicalResult:
    passed = []
    failed = []
    metrics: Dict[str, Optional[float]] = {}

    if df is None or df.empty or len(df) < 60:
        failed.append("Not enough weekly data.")
        return TechnicalResult(symbol, False, None, None, None, metrics, passed, failed)

    df = df.copy().dropna()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    current = df.iloc[-1]
    previous = df.iloc[-2]

    current_close = float(current["Close"])
    previous_close = float(previous["Close"])

    ma20 = sma(close, FW_MA_WEEKS).iloc[-1]

    macd_line, signal_line = macd_line_and_signal(close)
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]

    current_natr = natr(df, 14).iloc[-1]

    # Consolidation box uses prior 6 weeks, excluding breakout candle.
    prior_box = df.iloc[:-1].tail(FW_MIN_CONSOLIDATION_WEEKS)

    box_high = float(prior_box["High"].max())
    box_low = float(prior_box["Low"].min())
    box_range = box_high - box_low

    # Stop placement: lower boundary of middle third.
    stop = box_low + (box_range / 3.0)

    prior_10w_high_close = float(close.iloc[:-1].tail(FW_HIGH_CLOSE_LOOKBACK).max())

    breakout_pct = ((current_close - previous_close) / previous_close) * 100.0

    upper_wick = float(current["High"]) - max(float(current["Open"]), float(current["Close"]))
    candle_range = max(float(current["High"]) - float(current["Low"]), 0.000001)
    upper_wick_pct = (upper_wick / candle_range) * 100.0

    previous_volume = float(previous["Volume"])
    current_volume = float(current["Volume"])

    if previous_volume > 0:
        volume_spike_pct = ((current_volume - previous_volume) / previous_volume) * 100.0
    else:
        volume_spike_pct = None

    if stop < current_close:
        stop_risk_pct = ((current_close - stop) / current_close) * 100.0
    else:
        stop_risk_pct = None

    metrics.update({
        "close": current_close,
        "ma20": float(ma20) if not pd.isna(ma20) else None,
        "macd": float(current_macd) if not pd.isna(current_macd) else None,
        "macd_signal": float(current_signal) if not pd.isna(current_signal) else None,
        "natr": float(current_natr) if not pd.isna(current_natr) else None,
        "box_high": box_high,
        "box_low": box_low,
        "stop_middle_third": stop,
        "prior_10w_high_close": prior_10w_high_close,
        "breakout_pct": breakout_pct,
        "upper_wick_pct": upper_wick_pct,
        "volume_spike_pct": volume_spike_pct,
        "stop_risk_pct": stop_risk_pct,
    })

    # Rule 1 — price above 20-week MA
    if ma20 is not None and not pd.isna(ma20) and current_close > ma20:
        passed.append("Price above 20-week MA.")
    else:
        failed.append("Price not above 20-week MA.")

    # Rule 2 — MACD bullish
    if not pd.isna(current_macd) and not pd.isna(current_signal) and current_macd > current_signal:
        passed.append("Weekly MACD line above signal line.")
    else:
        failed.append("Weekly MACD not bullish.")

    # Rule 3 — NATR < 8
    if not pd.isna(current_natr) and current_natr < FW_NATR_MAX:
        passed.append("Weekly NATR under 8.")
    else:
        failed.append("Weekly NATR not under 8.")

    # Rule 4 — minimum 6-week consolidation
    if len(prior_box) >= FW_MIN_CONSOLIDATION_WEEKS and box_range > 0:
        passed.append("Minimum 6-week consolidation box exists.")
    else:
        failed.append("No valid 6-week consolidation box.")

    # Rule 5 — breakout above resistance and close above resistance
    if current_close > box_high:
        passed.append("Weekly close above consolidation resistance.")
    else:
        failed.append("Weekly close not above consolidation resistance.")

    # Rule 6 — 10-week closing high
    if current_close > prior_10w_high_close:
        passed.append("Breakout candle is a 10-week closing high.")
    else:
        failed.append("Breakout candle is not a 10-week closing high.")

    # Rule 7 — volume spike >= 30%
    if volume_spike_pct is not None and volume_spike_pct >= FW_MIN_VOLUME_SPIKE_PCT:
        passed.append("Volume spike at least 30% above prior week.")
    else:
        failed.append("Volume spike less than 30% above prior week.")

    # Rule 8 — breakout candle size 5% to 20%
    if FW_MIN_BREAKOUT_PCT < breakout_pct < FW_MAX_BREAKOUT_PCT:
        passed.append("Breakout size between 5% and 20%.")
    else:
        failed.append("Breakout size not between 5% and 20%.")

    # Rule 9 — upper wick <= 50%
    if upper_wick_pct <= FW_MAX_UPPER_WICK_PCT:
        passed.append("Upper wick is 50% or less.")
    else:
        failed.append("Upper wick greater than 50%.")

    # Rule 10 — stop loss less than 20%
    if stop_risk_pct is not None and stop_risk_pct < FW_MAX_STOP_RISK_PCT:
        passed.append("Stop risk under 20%.")
    else:
        failed.append("Stop risk not under 20%.")

    technical_pass = len(failed) == 0

    return TechnicalResult(
        symbol=symbol,
        technical_pass=technical_pass,
        entry=current_close,
        stop=stop,
        stop_risk_pct=stop_risk_pct,
        metrics=metrics,
        passed=passed,
        failed=failed
    )


# ============================================================
# FUNDAMENTALS
# ============================================================

def safe_num(value) -> Optional[float]:
    try:
        if value is None:
            return None

        if isinstance(value, (int, float, np.integer, np.floating)):
            if pd.isna(value):
                return None
            return float(value)

        text = str(value).strip()

        if text.lower() in ("", "none", "nan", "n/a"):
            return None

        return float(text)

    except Exception:
        return None


def get_row_value(df: pd.DataFrame, possible_names: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None

    index_map = {str(idx).lower(): idx for idx in df.index}

    for name in possible_names:
        key = name.lower()

        if key in index_map:
            try:
                row = df.loc[index_map[key]]
                if len(row) > 0:
                    return safe_num(row.iloc[0])
            except Exception:
                pass

    return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_fundamentals(symbol: str) -> FundamentalResult:
    passed = []
    failed = []
    metrics: Dict[str, Optional[float]] = {}

    try:
        ticker = yf.Ticker(symbol)

        info = getattr(ticker, "info", {}) or {}

        roe = safe_num(info.get("returnOnEquity"))
        operating_margin = safe_num(info.get("operatingMargins"))

        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet

        revenue = get_row_value(income_stmt, [
            "Total Revenue",
            "TotalRevenue"
        ])

        net_income = get_row_value(income_stmt, [
            "Net Income",
            "NetIncome"
        ])

        operating_income = get_row_value(income_stmt, [
            "Operating Income",
            "OperatingIncome"
        ])

        total_debt = get_row_value(balance_sheet, [
            "Total Debt",
            "TotalDebt"
        ])

        total_equity = get_row_value(balance_sheet, [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest"
        ])

        cash = get_row_value(balance_sheet, [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash"
        ])

        roc = None
        if operating_income is not None and total_equity is not None:
            debt = total_debt if total_debt is not None else 0.0
            cash_value = cash if cash is not None else 0.0

            invested_capital = total_equity + debt - cash_value

            if invested_capital > 0:
                roc = operating_income / invested_capital

        metrics.update({
            "roc": roc,
            "roe": roe,
            "operating_margin": operating_margin,
            "revenue": revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "total_debt": total_debt,
            "total_equity": total_equity,
            "cash": cash,
        })

        # FW fundamental rule 1 — ROC > 10%
        if roc is not None and roc > FW_MIN_ROC:
            passed.append("ROC above 10%.")
        else:
            failed.append("ROC not above 10% or unavailable.")

        # FW fundamental rule 2 — ROE > 10%
        if roe is not None and roe > FW_MIN_ROE:
            passed.append("ROE above 10%.")
        else:
            failed.append("ROE not above 10% or unavailable.")

        # FW fundamental rule 3 — Operating margin > 10%
        if operating_margin is not None and operating_margin > FW_MIN_OPERATING_MARGIN:
            passed.append("Operating margin above 10%.")
        else:
            failed.append("Operating margin not above 10% or unavailable.")

        # FW fundamental rule 4 — Revenue positive
        if revenue is not None and revenue > 0:
            passed.append("Revenue positive.")
        else:
            failed.append("Revenue not positive or unavailable.")

        # FW fundamental rule 5 — Net profit positive
        if net_income is not None and net_income > 0:
            passed.append("Net income positive.")
        else:
            failed.append("Net income not positive or unavailable.")

        fundamental_pass = len(failed) == 0

        return FundamentalResult(
            symbol=symbol,
            fundamental_pass=fundamental_pass,
            metrics=metrics,
            passed=passed,
            failed=failed
        )

    except Exception as e:
        failed.append(f"Fundamental data error: {e}")

        return FundamentalResult(
            symbol=symbol,
            fundamental_pass=False,
            metrics=metrics,
            passed=passed,
            failed=failed
        )


# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.header("Universe")

universe = st.sidebar.selectbox(
    "Choose universe",
    [
        "S&P 500",
        "Nasdaq-100",
        "NASDAQ All",
        "NYSE All",
        "AMEX All",
        "ALL US",
        "Paste tickers",
        "Upload tickers file"
    ],
    index=0
)

tv_prefix = st.sidebar.selectbox(
    "TradingView export prefix",
    ["NASDAQ", "NYSE", "AMEX", "NYSEARCA", "CBOE"],
    index=0
)

st.sidebar.header("Performance safety")

max_tickers = st.sidebar.number_input(
    "Max tickers to scan",
    min_value=25,
    max_value=10000,
    value=500,
    step=25
)

batch_size = st.sidebar.number_input(
    "Download batch size",
    min_value=10,
    max_value=250,
    value=80,
    step=10
)

parallel_batches = st.sidebar.slider(
    "Parallel batches",
    min_value=1,
    max_value=8,
    value=3,
    step=1
)

st.sidebar.header("Position sizing")

account_equity = st.sidebar.number_input(
    "Account equity ($)",
    min_value=1000,
    value=25000,
    step=1000
)

st.sidebar.caption(
    f"FW position size is fixed at 16% of equity: ${account_equity * FW_POSITION_SIZE_PCT:,.2f}"
)


# ============================================================
# LOAD UNIVERSE
# ============================================================

symbols: List[str] = []

if universe == "S&P 500":
    symbols = load_sp500()

elif universe == "Nasdaq-100":
    symbols = load_nasdaq100()

elif universe == "NASDAQ All":
    symbols = load_exchange_list("NASDAQ")

elif universe == "NYSE All":
    symbols = load_exchange_list("NYSE")

elif universe == "AMEX All":
    symbols = load_exchange_list("AMEX")

elif universe == "ALL US":
    symbols = load_exchange_list("ALL")

elif universe == "Paste tickers":
    pasted = st.sidebar.text_area(
        "Paste tickers",
        value="AAPL MSFT NVDA AMD GOOGL META",
        height=120
    )
    symbols = parse_tickers(pasted)

elif universe == "Upload tickers file":
    upload = st.sidebar.file_uploader("Upload .txt or .csv", type=["txt", "csv"])

    if upload is not None:
        raw = upload.read()

        try:
            df_upload = pd.read_csv(io.BytesIO(raw))
            col = df_upload.columns[0]

            for c in df_upload.columns:
                if str(c).lower() == "symbol":
                    col = c
                    break

            symbols = parse_tickers("\n".join(df_upload[col].astype(str).tolist()))

        except Exception:
            symbols = parse_tickers(raw.decode("utf-8", errors="ignore"))


symbols = clean_symbols(symbols)

if len(symbols) > max_tickers:
    symbols = symbols[:max_tickers]


# ============================================================
# MAIN UI
# ============================================================

confirmed = weekly_close_confirmed()

c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    run_scan = st.button("Run Exact FW Scan", use_container_width=True)

with c2:
    show_only_ready = st.checkbox("Show only READY", value=False)

with c3:
    search = st.text_input("Search ticker").strip().upper()

st.write(f"**Universe loaded:** {len(symbols)} symbols")

if confirmed:
    st.success("Weekly close is confirmed. READY signals are official.")
else:
    st.warning("Weekly close is not confirmed. Passing setups will be marked WATCHLIST until Friday close.")

if not run_scan:
    st.info("Click **Run Exact FW Scan** to begin.")
    st.stop()


# ============================================================
# RUN TECHNICAL SCAN
# ============================================================

st.subheader("1. Weekly Technical Scan")

progress = st.progress(0)
status_text = st.empty()

technical_results: Dict[str, TechnicalResult] = {}

symbol_batches = chunked(symbols, int(batch_size))

def process_batch(batch: List[str]) -> List[TechnicalResult]:
    weekly_data = download_weekly_batch(batch)
    results = []

    for sym in batch:
        df = weekly_data.get(sym)
        results.append(evaluate_technical(sym, df))

    return results


completed = 0

with ThreadPoolExecutor(max_workers=int(parallel_batches)) as executor:
    futures = {
        executor.submit(process_batch, batch): batch
        for batch in symbol_batches
    }

    for future in as_completed(futures):
        try:
            batch_results = future.result()
        except Exception:
            batch_results = []

        for result in batch_results:
            technical_results[result.symbol] = result

        completed += 1
        progress.progress(completed / max(1, len(symbol_batches)))
        status_text.write(f"Technical scan progress: {completed}/{len(symbol_batches)} batches")

progress.empty()
status_text.success("Technical scan complete.")


# ============================================================
# RUN FUNDAMENTAL SCAN ONLY ON TECHNICAL PASS NAMES
# ============================================================

technical_pass_symbols = [
    sym for sym, result in technical_results.items()
    if result.technical_pass
]

st.subheader("2. Fundamental Quality Scan")
st.write(f"Technical pass candidates sent to fundamentals: **{len(technical_pass_symbols)}**")

fundamental_results: Dict[str, FundamentalResult] = {}

fund_progress = st.progress(0)
fund_status = st.empty()

completed_fund = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(fetch_fundamentals, sym): sym
        for sym in technical_pass_symbols
    }

    for future in as_completed(futures):
        sym = futures[future]

        try:
            fundamental_results[sym] = future.result()
        except Exception:
            fundamental_results[sym] = FundamentalResult(
                symbol=sym,
                fundamental_pass=False,
                metrics={},
                passed=[],
                failed=["Fundamental scan failed."]
            )

        completed_fund += 1

        if len(technical_pass_symbols) > 0:
            fund_progress.progress(completed_fund / len(technical_pass_symbols))

        fund_status.write(
            f"Fundamental scan progress: {completed_fund}/{len(technical_pass_symbols)}"
        )

fund_progress.empty()
fund_status.success("Fundamental scan complete.")


# ============================================================
# BUILD FINAL RESULTS
# ============================================================

final_results: List[FinalResult] = []

position_value_default = account_equity * FW_POSITION_SIZE_PCT

for sym, tech in technical_results.items():
    fund = fundamental_results.get(sym)

    if fund is None:
        fund = FundamentalResult(
            symbol=sym,
            fundamental_pass=False,
            metrics={},
            passed=[],
            failed=["Fundamentals not checked because technical scan failed."]
        )

    if tech.technical_pass and fund.fundamental_pass:
        status = "READY" if confirmed else "WATCHLIST"
    elif tech.technical_pass and not fund.fundamental_pass:
        status = "FUND_FAIL"
    else:
        status = "TECH_FAIL"

    shares = 0
    position_value = 0.0
    risk_dollars = 0.0
    risk_on_equity_pct = 0.0

    if tech.entry is not None and tech.stop is not None and tech.entry > 0:
        shares = int(position_value_default // tech.entry)
        position_value = shares * tech.entry

        if tech.stop_risk_pct is not None:
            risk_dollars = position_value * (tech.stop_risk_pct / 100.0)
            risk_on_equity_pct = (risk_dollars / account_equity) * 100.0

    final_results.append(
        FinalResult(
            symbol=sym,
            status=status,
            entry=tech.entry,
            stop=tech.stop,
            stop_risk_pct=tech.stop_risk_pct,
            position_value=position_value,
            shares=shares,
            risk_dollars=risk_dollars,
            risk_on_equity_pct=risk_on_equity_pct,
            technical=tech,
            fundamental=fund
        )
    )


order = {
    "READY": 0,
    "WATCHLIST": 1,
    "FUND_FAIL": 2,
    "TECH_FAIL": 3,
}

final_results.sort(
    key=lambda r: (
        order.get(r.status, 9),
        r.symbol
    )
)

if search:
    final_results = [
        r for r in final_results
        if search in r.symbol
    ]

if show_only_ready:
    final_results = [
        r for r in final_results
        if r.status == "READY"
    ]


# ============================================================
# OUTPUT SUMMARY
# ============================================================

ready_symbols = [r.symbol for r in final_results if r.status == "READY"]
watchlist_symbols = [r.symbol for r in final_results if r.status == "WATCHLIST"]

s1, s2, s3, s4 = st.columns(4)

s1.metric("Scanned", len(final_results))
s2.metric("READY", len(ready_symbols))
s3.metric("WATCHLIST", len(watchlist_symbols))
s4.metric("Technical Pass", len(technical_pass_symbols))


# ============================================================
# TRADINGVIEW EXPORT
# ============================================================

st.subheader("TradingView Export")

export_symbols = ready_symbols if confirmed else watchlist_symbols

export_text = "\n".join([f"{tv_prefix}:{sym}" for sym in export_symbols])

st.text_area(
    "Copy this into TradingView",
    value=export_text,
    height=120
)

st.download_button(
    "Download FW_watchlist.txt",
    data=export_text.encode("utf-8"),
    file_name="FW_watchlist.txt",
    mime="text/plain",
    use_container_width=True
)


# ============================================================
# RESULTS TABLE
# ============================================================

st.subheader("Exact FW Scan Results")

table_rows = []

for r in final_results:
    tm = r.technical.metrics
    fm = r.fundamental.metrics

    table_rows.append({
        "Symbol": r.symbol,
        "Status": r.status,
        "Entry": None if r.entry is None else round(r.entry, 2),
        "Stop": None if r.stop is None else round(r.stop, 2),
        "Stop Risk %": None if r.stop_risk_pct is None else round(r.stop_risk_pct, 2),
        "Shares @ 16%": r.shares,
        "Position $": round(r.position_value, 2),
        "Risk $": round(r.risk_dollars, 2),
        "Risk on Equity %": round(r.risk_on_equity_pct, 2),
        "Close": None if tm.get("close") is None else round(tm.get("close"), 2),
        "20W MA": None if tm.get("ma20") is None else round(tm.get("ma20"), 2),
        "NATR": None if tm.get("natr") is None else round(tm.get("natr"), 2),
        "Volume Spike %": None if tm.get("volume_spike_pct") is None else round(tm.get("volume_spike_pct"), 2),
        "Breakout %": None if tm.get("breakout_pct") is None else round(tm.get("breakout_pct"), 2),
        "Upper Wick %": None if tm.get("upper_wick_pct") is None else round(tm.get("upper_wick_pct"), 2),
        "Box High": None if tm.get("box_high") is None else round(tm.get("box_high"), 2),
        "Box Low": None if tm.get("box_low") is None else round(tm.get("box_low"), 2),
        "ROC": None if fm.get("roc") is None else round(fm.get("roc") * 100, 2),
        "ROE": None if fm.get("roe") is None else round(fm.get("roe") * 100, 2),
        "Operating Margin": None if fm.get("operating_margin") is None else round(fm.get("operating_margin") * 100, 2),
    })

results_df = pd.DataFrame(table_rows)

st.dataframe(
    results_df,
    use_container_width=True,
    height=520
)


# ============================================================
# EXPLAINABILITY
# ============================================================

st.subheader("Why did it pass or fail?")

if len(final_results) > 0:
    selected_symbol = st.selectbox(
        "Select ticker",
        options=[r.symbol for r in final_results],
        index=0
    )

    selected = next((r for r in final_results if r.symbol == selected_symbol), None)

    if selected is not None:
        left, right = st.columns(2)

        with left:
            st.markdown(f"### {selected.symbol} — {selected.status}")

            st.write("**Execution**")
            st.write(f"- Entry: {selected.entry if selected.entry is not None else '—'}")
            st.write(f"- Stop: {selected.stop if selected.stop is not None else '—'}")
            st.write(f"- Stop Risk %: {selected.stop_risk_pct if selected.stop_risk_pct is not None else '—'}")
            st.write(f"- Shares using 16% FW position size: {selected.shares}")
            st.write(f"- Risk on equity: {selected.risk_on_equity_pct:.2f}%")

            st.write("**Technical Metrics**")
            st.write(selected.technical.metrics)

            st.write("**Fundamental Metrics**")
            st.write(selected.fundamental.metrics)

        with right:
            st.markdown("### Technical Passed")
            if selected.technical.passed:
                for item in selected.technical.passed:
                    st.success(item)
            else:
                st.info("None")

            st.markdown("### Technical Failed")
            if selected.technical.failed:
                for item in selected.technical.failed:
                    st.error(item)
            else:
                st.success("None")

            st.markdown("### Fundamental Passed")
            if selected.fundamental.passed:
                for item in selected.fundamental.passed:
                    st.success(item)
            else:
                st.info("None")

            st.markdown("### Fundamental Failed")
            if selected.fundamental.failed:
                for item in selected.fundamental.failed:
                    st.error(item)
            else:
                st.success("None")


# ============================================================
# JOURNAL EXPORT
# ============================================================

st.subheader("Journal Export")

journal_df = results_df[
    results_df["Status"].isin(["READY", "WATCHLIST"])
].copy()

journal_df.insert(
    0,
    "Scan Timestamp",
    now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
)

csv_bytes = journal_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download FW_journal.csv",
    data=csv_bytes,
    file_name="FW_journal.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

with st.expander("How this scanner matches Financial Wisdom"):
    st.markdown(
        f"""
        This scanner uses fixed FW criteria only:

        **Technical**
        - Weekly chart
        - Price above 20-week MA
        - Weekly MACD line above signal
        - NATR below 8
        - Minimum 6-week consolidation box
        - Weekly close above resistance
        - 10-week closing high
        - Volume spike at least 30%
        - Breakout size greater than 5% and less than 20%
        - Upper wick 50% or less
        - Stop risk under 20%

        **Stop**
        - Consolidation box is split into thirds.
        - Stop is placed at the lower boundary of the middle third.

        **Fundamentals**
        - ROC above 10%
        - ROE above 10%
        - Operating margin above 10%
        - Revenue positive
        - Net income positive

        **Position sizing**
        - Uses FW example position size of 16% of equity.
        """
    )
