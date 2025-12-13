import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FW Protocol Scanner", page_icon="🦅", layout="wide")

st.title("Financial Wisdom Market Scanner 🦅")

# --- DROPDOWN MENUS ---
col1, col2 = st.columns(2)

with col1:
    universe = st.selectbox(
        "1. Select Market Universe:",
        ("S&P 500 (Large Cap)", "S&P 400 (Mid Cap)", "S&P 600 (Small Cap)", "S&P 1500 (All Quality US Stocks)")
    )

with col2:
    scan_mode = st.selectbox(
        "2. Select Strategy:",
        ("Standard Scan (Layer A - Setups)", "Sniper Scan (Green Label - Breakouts)")
    )

st.info(f"Ready to scan **{universe}** using **{scan_mode}** logic.")

# --- DATA FUNCTIONS ---
@st.cache_data(ttl=86400)
def get_tickers(selection):
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    
    try:
        # 1. S&P 500 (Large)
        if "500" in selection or "1500" in selection:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tickers.extend(pd.read_html(requests.get(url, headers=headers).text)[0]['Symbol'].tolist())
            
        # 2. S&P 400 (Mid)
        if "400" in selection or "1500" in selection:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
            tickers.extend(pd.read_html(requests.get(url, headers=headers).text)[0]['Symbol'].tolist())

        # 3. S&P 600 (Small)
        if "600" in selection or "1500" in selection:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
            tickers.extend(pd.read_html(requests.get(url, headers=headers).text)[0]['Symbol'].tolist())

        # Clean tickers (Change BRK.B to BRK-B and remove duplicates)
        tickers = list(set([t.replace('.', '-') for t in tickers]))
        return tickers
        
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return []

# --- LOGIC 1: STANDARD (LAYER A - SETUP) ---
def check_layer_a(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1wk")
        if len(df) < 52: return None

        # Indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['NATR'] = (df['ATR'] / df['Close']) * 100
        
        # MACD
        macd = ta.macd(df['Close'])
        df['MACD_Line'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']

        curr = df.iloc[-1]

        # RULES
        trend = (curr['Close'] > curr['SMA_20']) and (curr['Close'] > curr['SMA_50'])
        alignment = curr['SMA_20'] > curr['SMA_50']
        
        # Consolidation (Avg of last 4 weeks < 8%)
        if len(df) >= 4:
            consolidation = df['NATR'].iloc[-5:-1].mean() < 8 
        else:
            consolidation = False
            
        momentum = curr['MACD_Line'] > curr['MACD_Signal']

        if trend and alignment and consolidation and momentum:
            return {
                'Ticker': ticker,
                'Price': round(curr['Close'], 2),
                'NATR %': round(curr['NATR'], 2),
                'Trend': "Strong"
            }
        return None
    except:
        return None

# --- LOGIC 2: SNIPER (GREEN LABEL - BREAKOUT) ---
def check_sniper(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1wk")
        if len(df) < 52: return None

        # Indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['NATR'] = (df['ATR'] / df['Close']) * 100
        
        # Box High (Max of previous 12 weeks)
        df['Box_High'] = df['High'].rolling(12).max().shift(1)
        
        # MACD
        macd = ta.macd(df['Close'])
        df['MACD_Line'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # RULES
        trend = (curr['Close'] > curr['SMA_20']) and (curr['SMA_20'] > curr['SMA_50'])
        tight = df['NATR'].iloc[-5:-1].mean() < 8 
        
        if not (trend and tight): return None

        # Breakout Details
        breakout = curr['Close'] > curr['Box_High']
        volume = curr['Volume'] > (prev['Volume'] * 1.3)
        
        rng = curr['High'] - curr['Low']
        wick = (curr['High'] - max(curr['Open'], curr['Close']))
        clean_candle = (wick / rng) < 0.50 if rng > 0 else False

        macd_bull = curr['MACD_Line'] > curr['MACD_Signal']

        if breakout and volume and clean_candle and macd_bull:
            return {
                'Ticker': ticker,
                'Price': round(curr['Close'], 2),
                'Vol Spike': f"{round(((curr['Volume']/prev['Volume'])-1)*100, 1)}%",
                'NATR': round(curr['NATR'], 2)
            }
        return None
    except:
        return None

# --- EXECUTION ---
if st.button("🚀 RUN SCAN"):
    tickers = get_tickers(universe)
    
    if not tickers: 
        st.stop()
    
    st.write(f"Loading {len(tickers)} stocks... (This may take a few minutes)")
    
    progress = st.progress(0)
    status = st.empty()
    results = []
    
    for i, ticker in enumerate(tickers):
        # Update progress bar every 10 stocks
        if i % 10 == 0: 
            progress.progress((i+1)/len(tickers))
            status.text(f"Scanning: {ticker}")
            
        if "Standard" in scan_mode:
            data = check_layer_a(ticker)
        else:
            data = check_sniper(ticker)
            
        if data: results.append(data)
            
    progress.empty()
    status.empty()
    
    # --- OUTPUT ---
    if results:
        df_results = pd.DataFrame(results)
        
        if "Standard" in scan_mode:
            st.success(f"✅ Found {len(results)} Candidates")
            df_results = df_results.sort_values(by="NATR %")
        else:
            st.balloons()
            st.success(f"🔥 FOUND {len(results)} BREAKOUTS")
            
        st.dataframe(df_results, use_container_width=True)
        
        st.subheader("📋 Copy List for TradingView")
        st.code(",".join(df_results['Ticker'].tolist()))
    else:
        st.warning("No stocks matched the criteria.")
