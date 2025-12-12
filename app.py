import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FW Protocol Scanner", page_icon="📈")

st.title("Financial Wisdom S&P 500 Scanner")
st.write("This app scans the full S&P 500 for 'Layer A' candidates: Strong Trend + Tight Base.")

# --- FUNCTIONS ---
@st.cache_data(ttl=86400) # Cache the list for 24 hours to speed up
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(response.text)
        df = dfs[0]
        return df['Symbol'].tolist()
    except Exception as e:
        st.error(f"Error fetching list: {e}")
        return []

def check_protocol(ticker):
    try:
        # Fetch data (Silent mode)
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y", interval="1wk")
        
        if df.empty or len(df) < 52:
            return None

        # Indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['NATR'] = (df['ATR'] / df['Close']) * 100
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is None: return None
        df['MACD_Line'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']

        curr = df.iloc[-1]

        # LOGIC
        trend = (curr['Close'] > curr['SMA_20']) and (curr['Close'] > curr['SMA_50'])
        alignment = curr['SMA_20'] > curr['SMA_50']
        
        if len(df) >= 4:
            consolidation = df['NATR'].iloc[-4:].mean() < 8 # Tight Base Rule
        else:
            consolidation = False
            
        momentum = curr['MACD_Line'] > curr['MACD_Signal']

        if trend and alignment and consolidation and momentum:
            return {
                'Ticker': ticker,
                'Price': round(curr['Close'], 2),
                'NATR %': round(curr['NATR'], 2),
            }
        return None

    except Exception:
        return None

# --- APP LOGIC ---
if st.button("🚀 START WEEKLY SCAN"):
    tickers = get_sp500_tickers()
    
    if not tickers:
        st.stop()
        
    st.info(f"Scanning {len(tickers)} stocks... Please wait (approx 2-3 mins).")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    # Run the Loop
    for i, ticker in enumerate(tickers):
        # Update progress every 10 stocks to save resources
        if i % 10 == 0:
            progress_bar.progress((i + 1) / len(tickers))
            status_text.text(f"Scanning: {ticker}")
            
        # Fix Ticker format (BRK.B -> BRK-B)
        clean_ticker = ticker.replace('.', '-')
        
        data = check_protocol(clean_ticker)
        if data:
            results.append(data)
            
    progress_bar.empty()
    status_text.empty()
    
    # --- DISPLAY RESULTS ---
    if results:
        st.success(f"✅ Scan Complete! Found {len(results)} Candidates.")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="NATR %") # Best tight bases first
        
        # 1. Nice Interactive Table
        st.dataframe(df_results, use_container_width=True)
        
        # 2. Copy/Paste List for TradingView
        st.subheader("📋 Copy for Watchlist")
        ticker_list = ",".join(df_results['Ticker'].tolist())
        st.code(ticker_list, language=None)
        st.caption("Copy the text above and paste it into TradingView's 'Import List' feature.")
        
    else:
        st.warning("No stocks matched the strict criteria this week.")
