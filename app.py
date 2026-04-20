import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from data_provider import fetch_and_analyze, TiingoProvider, IndicatorCalculator
import requests

st.set_page_config(page_title="Mean Reversion Algoritmo", layout="wide")

st.title("📊 Mean Reversion Intraday Analyzer")
st.markdown("**Versione 1.0** — Real-time analysis with Tiingo IEX data")

# ==================== SIDEBAR: INPUT ====================
st.sidebar.header("⚙️ Configuration")

# API Key from Streamlit Secrets
try:
    api_key = st.secrets["tiingo_api_key"]
except KeyError:
    st.error("❌ API key not found in Streamlit Secrets. Please configure it in the app settings.")
    st.stop()

# User inputs
st.sidebar.subheader("Market Context")
market_time_str = st.sidebar.text_input("Current time (HH:MM CET)", "09:30")
market_status = st.sidebar.radio("Market Status", ["Pre-market", "RTH (Regular Trading)", "Post-market"], index=1)
small_cap_mode = st.sidebar.checkbox("SMALL CAP MODE (< $300M)", value=False)

st.sidebar.subheader("Tickers to Analyze")
tickers_input = st.sidebar.text_area(
    "Enter ticker symbols (one per line, max 5)",
    "AAPL\nMSFT\nTSLA",
    height=120
)

tickers = [t.strip().upper() for t in tickers_input.strip().split('\n') if t.strip()]
if len(tickers) > 5:
    st.sidebar.warning("⚠️ Max 5 tickers allowed. Using first 5.")
    tickers = tickers[:5]

# ==================== ANALYSIS BUTTON ====================
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    st.session_state.run_analysis = True
else:
    st.session_state.run_analysis = False

# ==================== MAIN ANALYSIS ====================
if st.session_state.get("run_analysis", False):
    st.markdown("---")
    st.subheader(f"📈 Analysis Results — {market_status} — {market_time_str}")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    errors = []
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Fetching {ticker}... ({idx + 1}/{len(tickers)})")
        progress_bar.progress((idx + 1) / len(tickers))
        
        try:
            analysis = fetch_and_analyze(ticker, api_key)
            if analysis is None:
                errors.append(f"{ticker}: Failed to fetch data")
                continue
            
            # Calculate technical score and MRPS
            rsi = analysis.get('rsi', np.nan)
            rsi_slope = analysis.get('rsi_slope', np.nan)
            vwap = analysis.get('vwap', np.nan)
            bb_zscore = analysis.get('bb_zscore', np.nan)
            atr = analysis.get('atr', np.nan)
            volume_last = analysis.get('volume_last', np.nan)
            adv20 = analysis.get('adv20', np.nan)
            intraday_bars = analysis.get('intraday_bars_count', 0)
            last_price = analysis.get('last_price', np.nan)
            timestamp = analysis.get('timestamp', 'N/A')
            
            # Volume ratio
            vol_ratio = volume_last / adv20 if not np.isnan(adv20) and adv20 > 0 else np.nan
            
            # Simple technical score (weighted sum of normalized indicators)
            # RSI: 0-100, weight 17%
            rsi_score = (30 - rsi) / 30 * 100 if not np.isnan(rsi) else 50  # Oversold favors MR
            
            # RSI slope: positive slope means RSI reversing up, favorable
            rsi_slope_score = np.clip((rsi_slope + 5) / 10 * 100, 0, 100) if not np.isnan(rsi_slope) else 50
            
            # VWAP distance: below VWAP favors MR
            bb_score = np.clip((2 - bb_zscore) / 4 * 100, 0, 100) if not np.isnan(bb_zscore) else 50
            
            # Volume: high volume panic confirms MR opportunity
            vol_score = np.clip(vol_ratio * 50, 0, 100) if not np.isnan(vol_ratio) else 50
            
            # Technical Score (weighted)
            technical_score = (
                rsi_score * 0.17 +
                rsi_slope_score * 0.11 +
                bb_score * 0.11 +
                vol_score * 0.11 +
                60 * 0.50  # Placeholder for other indicators
            )
            
            # Regime multiplier (placeholder based on market status)
            if market_status == "Pre-market":
                regime_mult = 0.7
            elif market_status == "RTH (Regular Trading)":
                regime_mult = 1.0
            else:  # Post-market
                regime_mult = 0.6
            
            # Liquidity quality (simplified)
            if not np.isnan(adv20):
                if adv20 > 5_000_000:
                    liquidity_quality = 1.0
                elif adv20 > 1_000_000:
                    liquidity_quality = 0.95
                else:
                    liquidity_quality = 0.85
            else:
                liquidity_quality = 0.9
            
            # Data quality multiplier
            if intraday_bars > 30:
                data_quality = 1.0
            elif intraday_bars > 10:
                data_quality = 0.95
            else:
                data_quality = 0.85
            
            # FRS penalty (Fundamental Risk Score) — placeholder, assume low FRS
            frs_penalty = 5  # Assume minor news risk
            
            # MRPS calculation
            mrps = (technical_score * regime_mult * liquidity_quality * data_quality) - frs_penalty
            mrps = np.clip(mrps, 0, 100)
            
            # Verdict
            if mrps < 55:
                verdict = "❌ No Trade"
                color = "🔴"
            elif mrps < 70:
                verdict = "⚠️ Watch"
                color = "🟡"
            elif mrps < 85:
                verdict = "✅ Buy Zone"
                color = "🟢"
            else:
                verdict = "🔥 High Conviction"
                color = "🟢"
            
            results.append({
                'Ticker': ticker,
                'Last Price': f"${last_price:.2f}" if not np.isnan(last_price) else "N/A",
                'Timestamp': timestamp,
                'RSI': f"{rsi:.1f}" if not np.isnan(rsi) else "N/A",
                'RSI Slope': f"{rsi_slope:.2f}" if not np.isnan(rsi_slope) else "N/A",
                'BB Z-Score': f"{bb_zscore:.2f}" if not np.isnan(bb_zscore) else "N/A",
                'Vol Ratio': f"{vol_ratio:.2f}x" if not np.isnan(vol_ratio) else "N/A",
                'ATR': f"${atr:.2f}" if not np.isnan(atr) else "N/A",
                'ADV20 ($M)': f"{adv20/1e6:.2f}" if not np.isnan(adv20) else "N/A",
                'Bars (1-min)': intraday_bars,
                'Tech Score': f"{technical_score:.1f}",
                'MRPS': f"{mrps:.1f}",
                'Verdict': verdict,
                'Signal': color,
            })
        
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    # ==================== DISPLAY RESULTS ====================
    
    # Main results table
    st.subheader("📊 Summary Table")
    if results:
        df_results = pd.DataFrame(results)
        
        # Display with custom styling
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Verdict": st.column_config.TextColumn(),
                "Signal": st.column_config.TextColumn(),
                "MRPS": st.column_config.NumberColumn(format="%.1f"),
            }
        )
    else:
        st.warning("No successful analyses. Check tickers and try again.")
    
    # Errors
    if errors:
        st.subheader("⚠️ Errors")
        for err in errors:
            st.error(err)
    
    # ==================== DETAILS & REASONING ====================
    st.markdown("---")
    st.subheader("📌 Detailed Reasoning")
    
    for result in results:
        with st.expander(f"**{result['Ticker']}** — MRPS {result['MRPS']} — {result['Verdict']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Price & Volume**")
                st.markdown(f"- Last Price: {result['Last Price']}")
                st.markdown(f"- Timestamp: {result['Timestamp']}")
                st.markdown(f"- ADV20: {result['ADV20 ($M)']}")
                st.markdown(f"- Vol Ratio: {result['Vol Ratio']}")
                st.markdown(f"- Intraday bars: {result['Bars (1-min)']}")
            
            with col2:
                st.markdown("**Technical Indicators**")
                st.markdown(f"- RSI (14): {result['RSI']}")
                st.markdown(f"- RSI Slope (5): {result['RSI Slope']}")
                st.markdown(f"- Bollinger Z: {result['BB Z-Score']}")
                st.markdown(f"- ATR (14): {result['ATR']}")
                st.markdown(f"- Tech Score: {result['Tech Score']}")
            
            st.markdown(f"**Mean Reversion Probability Score (MRPS): `{result['MRPS']}`**")
            
            if float(result['MRPS'].strip()) < 55:
                st.info("🔴 **No Trade** — MRPS too low. Wait for better setup.")
            elif float(result['MRPS'].strip()) < 70:
                st.warning("🟡 **Watch** — Entry only on reversal confirmation. Monitor for signal.")
            elif float(result['MRPS'].strip()) < 85:
                st.success("🟢 **Buy Zone** — Favorable mean reversion setup. Monitor entry levels.")
            else:
                st.success("🔥 **High Conviction** — Strong mean reversion opportunity. Consider entry.")
    
    # ==================== DATA QUALITY REPORT ====================
    st.markdown("---")
    st.subheader("📡 Data Quality Report")
    
    for result in results:
        st.markdown(f"**{result['Ticker']}**")
        st.markdown(f"- Data source: Tiingo IEX real-time")
        st.markdown(f"- Bars collected (1-min): {result['Bars (1-min)']}")
        st.markdown(f"- Last update: {result['Timestamp']}")
        st.markdown(f"- Data quality: ✓ OK" if int(result['Bars (1-min)']) > 10 else "⚠️ Limited bars")
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This analysis is for educational purposes only. Not investment advice. 
    Always conduct your own due diligence and consult a financial advisor before trading.
    """)

else:
    st.info("👈 Configure inputs in the sidebar and click **Run Analysis** to start.")
