
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FW Protocol Scanner", page_icon="🦅", layout="wide")

# --- HEADER & SIDEBAR ---
st.title("Financial Wisdom Protocol Scanner 🦅")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    universe_selection = st.selectbox(
        "1. Select Market Universe:",
        ("S&P 500 (Large Cap)", "S&P 400 (Mid Cap)", "S&P 600 (Small Cap)", "S&P 1500 (All Quality)", "Nasdaq 100 (Tech)")
    )

with col2:
    strategy_mode = st.selectbox(
        "2. Select Strategy Mode:",
        ("Standard Scan (Layer A - Setups)", "Sniper Scan (Green Label - Breakouts)")
    )

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=86400)
def get_tickers(selection):
    """
    Scrapes Wikipedia for ticker lists based on selection.
    Handles symbol cleaning (dots to hyphens).
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    
    try:
        # Nasdaq 100
        if "Nasdaq" in selection:
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            dfs = pd.read_html(requests.get(url, headers=headers).text)
            for df in dfs:
                if 'Ticker' in df.columns:
                    tickers.extend(df['Ticker'].tolist())
                    break
                elif 'Symbol' in df.columns:
                    tickers.extend(df['Symbol'].tolist())
                    break
        
        # S&P Indices
        else:
            if "500" in selection or "1500" in selection:
                url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
                tickers.extend(pd.read_html(requests.get(url, headers=headers).text)[0]['Symbol'].tolist())
            
            if "400" in selection or "1500" in selection:
                url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
                tickers.extend(pd.read_html(requests.get(url, headers=headers).text)[0]['Symbol'].tolist())

            if "600" in selection or "1500" in selection:
                url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
                tickers.extend(pd.read_html(requests.get(url, headers=headers).text)[0]['Symbol'].tolist())

        # Clean Tickers (BRK.B -> BRK-B for Yahoo Finance)
        clean_tickers = list(set([t.replace('.', '-') for t in tickers]))
        return clean_tickers

    except Exception as e:
        st.error(f"Data Source Error: {e}")
        return []

def analyze_stock(ticker, mode):
    """
    Fetches 2y Weekly Data via yfinance.
    Applies strict FW Protocol Rules.
    Returns dictionary with Pass/Fail and reasons.
    """
    try:
        # 1. FETCH DATA (Yahoo Finance Source)
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1wk")
        
        if len(df) < 52: 
            return None

        # 2. CALCULATE INDICATORS
        # Trend
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        
        # Volatility (NATR)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['NATR'] = (df['ATR'] / df['Close']) * 100
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is None: return None
        df['MACD_Line'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']

        # Box / Breakout Levels
        # Highest High of the last 12 weeks (Shifted 1 to exclude current week)
        df['Box_High'] = df['High'].rolling(12).max().shift(1)
        df['Box_Low'] = df['Low'].rolling(12).min().shift(1)

        # Current Candle
        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # 3. APPLY RULES (The Verdict Logic)
        reasons = []
        is_valid = True

        # Rule A: Trend
        if curr['Close'] > curr['SMA_20']:
            # reasons.append("Trend OK") # Too verbose, implied
            pass
        else:
            is_valid = False
            reasons.append("Price below 20 SMA")

        # Rule B: Tightness (Consolidation)
        # Average NATR of last 4 closed weeks
        avg_natr = df['NATR'].iloc[-5:-1].mean()
        if avg_natr < 8:
            pass
        else:
            is_valid = False
            reasons.append(f"Base too loose (NATR {round(avg_natr, 1)}%)")

        # Rule C: Momentum (MACD)
        if curr['MACD_Line'] > curr['MACD_Signal']:
            pass
        else:
            # We allow entering if MACD is just turning, but prefer bullish
            # is_valid = False # Strict rule? Let's keep it soft for Standard scan
            reasons.append("MACD Bearish")

        # --- MODE SPECIFIC RULES ---
        
        if "Sniper" in mode:
            # 1. Breakout Check
            if curr['Close'] > curr['Box_High']:
                reasons.append("BREAKOUT")
            else:
                is_valid = False
                # No reason needed, simply not a result
            
            # 2. Volume Spike
            vol_change = ((curr['Volume'] / prev['Volume']) - 1) * 100
            if vol_change > 30:
                reasons.append(f"Vol Spike +{int(vol_change)}%")
            else:
                is_valid = False
                # reasons.append(f"Low Vol (+{int(vol_change)}%)")

            # 3. Wick Check
            range_len = curr['High'] - curr['Low']
            upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
            if range_len > 0 and (upper_wick / range_len) > 0.5:
                is_valid = False
                # reasons.append("Wick > 50%")

        # Final Formatting
        if is_valid:
            # Calculate Risk
            risk_pct = ((curr['Close'] - curr['Box_Low']) / curr['Close']) * 100
            
            return {
                "Ticker": ticker,
                "Price": round(curr['Close'], 2),
                "Risk": f"{round(risk_pct, 1)}%",
                "NATR": round(curr['NATR'], 2),
                "Vol Spike": f"+{int(((curr['Volume']/prev['Volume'])-1)*100)}%" if "Sniper" in mode else "N/A",
                "Notes": ", ".join(reasons) if reasons else "Clean Setup"
            }
        
        return None

    except Exception:
        return None

# --- EXECUTION LOGIC ---

if st.button("🚀 INITIATE SCAN"):
    
    # 1. Get Universe
    with st.status("Initializing Data Protocol...", expanded=True) as status:
        st.write(f"Connecting to Wikipedia source for {universe_selection}...")
        tickers = get_tickers(universe_selection)
        
        if not tickers:
            st.error("Failed to retrieve ticker list.")
            st.stop()
            
        st.write(f"Successfully retrieved {len(tickers)} symbols.")
        st.write("Cleaning symbols for Yahoo Finance compatibility...")
        
        st.write(f"Beginning Analysis ({strategy_mode})...")
        
        # 2. Run Scan
        results = []
        progress_bar = st.progress(0)
        log_placeholder = st.empty()
        
        start_time = time.time()
        
        for i, ticker in enumerate(tickers):
            # Update UI every 5 ticks to save resources
            if i % 5 == 0:
                progress_bar.progress((i + 1) / len(tickers))
                log_placeholder.code(f"Scanning: {ticker} | Found: {len(results)}")
            
            data = analyze_stock(ticker, strategy_mode)
            if data:
                results.append(data)
                
        progress_bar.empty()
        log_placeholder.empty()
        status.update(label="Scan Complete!", state="complete", expanded=False)

    # 3. Display Results
    if results:
        st.success(f"✅ SCAN COMPLETE. Found {len(results)} Candidates.")
        
        df = pd.DataFrame(results)
        
        # Sort by best criteria
        if "Sniper" in strategy_mode:
            # For Sniper, we usually want to see them all, maybe sort by Volume
            pass 
        else:
            # For Standard, sort by Tightness (NATR)
            df = df.sort_values(by="NATR")

        st.dataframe(
            df,
            column_config={
                "Ticker": st.column_config.TextColumn("Symbol", help="Stock Ticker"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Risk": st.column_config.TextColumn("Risk %", help="Distance to Stop Loss"),
                "NATR": st.column_config.NumberColumn("Tightness", help="Lower is Better (<8)", format="%.2f%%"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # 4. Copy Paste Area
        st.markdown("### 📋 Copy for TradingView")
        st.code(",".join(df['Ticker'].tolist()))
        
    else:
        st.warning("No stocks matched the strict Financial Wisdom criteria this week.")
