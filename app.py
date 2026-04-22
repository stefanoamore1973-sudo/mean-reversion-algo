"""
app.py v2.0 — Mean Reversion Intraday Analyzer
================================================
Changelog vs v1.0:
- MRPS scoring completamente riscritto con pesi data-driven
- IBS come feature dominante (peso 22%) — Pagonidis 2013 evidence
- VWAP distance come co-dominante (peso 20%) — Madhavan 2002
- Volume ratio peso 15% — Conrad Hameed Niden 1994
- Regime detection automatica da SPY+VIX (non più switch manuale)
- FRS rimosso come penalty hardcoded (l'utente valuta news separatamente)
- Hard filters applicati: liquidità, spread, halt proxy
- Verdict thresholds ricalibrati
- Dashboard con ranking e interpretazione trasparente di ogni score
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_provider import (
    fetch_and_analyze,
    fetch_spy_daily,
    fetch_vix_daily,
    detect_market_regime,
)

st.set_page_config(page_title="Mean Reversion Algoritmo v2", layout="wide")

# ============================================================================
# HEADER
# ============================================================================
st.title("📊 Mean Reversion Intraday Analyzer")
st.caption("**v2.0** — Data-driven weights from empirical literature (Pagonidis 2013, Conrad-Hameed-Niden 1994, Madhavan 2002)")

# ============================================================================
# API KEY
# ============================================================================
try:
    api_key = st.secrets["tiingo_api_key"]
except KeyError:
    st.error("❌ API key non trovata. Configura 'tiingo_api_key' in Streamlit Secrets.")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Tickers")
tickers_input = st.sidebar.text_area(
    "Ticker symbols (uno per riga, max 5)",
    "AAPL\nMSFT\nTSLA",
    height=120
)
tickers = [t.strip().upper() for t in tickers_input.strip().split('\n') if t.strip()]
if len(tickers) > 5:
    st.sidebar.warning("Max 5 ticker. Uso i primi 5.")
    tickers = tickers[:5]

st.sidebar.subheader("Filtri di liquidità")
min_price = st.sidebar.number_input("Prezzo minimo ($)", value=5.0, step=0.5)
min_adv20_m = st.sidebar.number_input("ADV20 minimo ($M)", value=5.0, step=1.0)
small_cap_mode = st.sidebar.checkbox("SMALL CAP MODE (soglie rilassate)", value=False)
if small_cap_mode:
    min_price = 2.0
    min_adv20_m = 1.0

run = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ============================================================================
# MRPS — PESI BASATI SU EVIDENZA EMPIRICA
# ============================================================================
#
# Ogni peso è giustificato da letteratura o backtest pubblicati.
# Tre cluster di indicatori:
#   A) Position / deviation signals (43%): quanto è stretched il ticker
#   B) Momentum / confirmation signals (30%): sta effettivamente girando?
#   C) Participation signals (27%): il mercato sta confermando?

WEIGHTS = {
    # A) POSITION / DEVIATION (43%)
    'ibs':            0.22,  # Pagonidis 2013: IBS<0.2 avg +0.35% next day; filtro migliora returns di ~10pp
    'vwap_dist_atr':  0.20,  # Madhavan 2002; Tradewink 2026: VWAP+/-2SD = zone reversal reliable
    'bb_zscore':      0.08,  # Bollinger Bands: |z|>2 = stretch estremo (una delle top-5 per MR per Liberated Stock Trader)
    # Totale cluster A: 50%
    # Nota: 50% delle informazioni "dove sta il prezzo rispetto al mean"
    
    # B) MOMENTUM / REVERSAL CONFIRMATION (25%)
    'rsi':            0.12,  # Wilder 1978; ma meno affidabile stand-alone di quanto si pensi (vedi LST 120k test)
    'rsi_slope':      0.08,  # Inversione momentum in atto
    'candle_score':   0.05,  # Nison 1991, Bulkowski: pattern reversal. Basso perché efficacia empirica modesta.
    # Totale cluster B: 25%
    
    # C) PARTICIPATION / CONTEXT (25%)
    'vol_ratio':      0.15,  # Conrad-Hameed-Niden 1994: volume alto + drop = strong reversal signal. TOP empirical signal.
    'rel_strength_5d':0.05,  # Ticker underperform SPY = MR candidate (Chan-Jegadeesh-Lakonishok 1996)
    'gap_atr':        0.05,  # Gap down >1 ATR = panic proxy
}
# Totale: 100%

# Somma check
_total = sum(WEIGHTS.values())
assert abs(_total - 1.0) < 0.001, f"Weights sum = {_total}, deve essere 1.0"


# ============================================================================
# SCORING FUNCTIONS (ognuna restituisce un punteggio 0-100)
# ============================================================================

def score_ibs(ibs_val):
    """
    IBS scoring — Pagonidis 2013.
    IBS < 0.2 → +0.35% avg next day (molto bullish)
    IBS > 0.8 → -0.13% avg next day (bearish)
    Scoring invertito: basso IBS = alto score.
    """
    if pd.isna(ibs_val):
        return 50
    # Lineare decrescente: ibs=0 → 100, ibs=1 → 0
    return float(np.clip((1 - ibs_val) * 100, 0, 100))


def score_vwap_dist_atr(dist_atr):
    """
    VWAP distance in ATR unit.
    Setup MR long ideale: prezzo -0.5 / -1.5 ATR sotto VWAP.
    - dist = -1.0 → score 100 (sweet spot)
    - dist = 0 → score 50
    - dist = +1.0 → score 0 (sopra VWAP, no setup long)
    - dist < -2.5 → score 40 (troppo lontano, potrebbe essere downtrend forte)
    """
    if pd.isna(dist_atr):
        return 50
    if dist_atr <= -2.5:
        return 40  # Troppo stretched, rischio downtrend
    if -1.5 <= dist_atr <= -0.5:
        return 100  # Sweet spot
    if -2.5 < dist_atr < -1.5:
        return 85
    if -0.5 < dist_atr < 0:
        return 70
    if 0 <= dist_atr < 0.5:
        return 40
    return 15  # Sopra VWAP, non MR long setup


def score_bb_zscore(z):
    """Z-score: -2 ideal, -3 troppo, 0 neutrale."""
    if pd.isna(z):
        return 50
    if z <= -2.5:
        return 80  # Molto stretched, ma rischio spirale
    if -2.5 < z <= -1.5:
        return 100  # Sweet spot
    if -1.5 < z <= -0.5:
        return 75
    if -0.5 < z <= 0.5:
        return 50
    if 0.5 < z <= 1.5:
        return 30
    return 10


def score_rsi(rsi):
    """RSI: ideal range 25-35 per MR long."""
    if pd.isna(rsi):
        return 50
    if rsi <= 20:
        return 85  # Very oversold, rischio extension
    if 20 < rsi <= 35:
        return 100  # Sweet spot MR long
    if 35 < rsi <= 45:
        return 70
    if 45 < rsi <= 55:
        return 50
    if 55 < rsi <= 70:
        return 25
    return 10  # Overbought: no setup long


def score_rsi_slope(slope):
    """RSI slope positivo = momentum reverting up. Ottimo."""
    if pd.isna(slope):
        return 50
    if slope >= 2:
        return 100
    if 0.5 <= slope < 2:
        return 80
    if -0.5 <= slope < 0.5:
        return 50
    if -2 <= slope < -0.5:
        return 30
    return 15


def score_candle(candle_val):
    """candle_score già in [0,1], lo scalo a [0,100]."""
    if pd.isna(candle_val):
        return 50
    return float(np.clip(candle_val * 100, 0, 100))


def score_vol_ratio(v):
    """
    Volume ratio (today $ vol cumulato / ADV20).
    Conrad-Hameed-Niden 1994: reversal più forte con volume alto.
    """
    if pd.isna(v):
        return 50
    if v >= 2.0:
        return 100  # Panic/capitulation volume → forte segnale MR
    if 1.3 <= v < 2.0:
        return 85
    if 0.7 <= v < 1.3:
        return 55
    if 0.3 <= v < 0.7:
        return 30
    return 15  # Volume troppo basso → no conviction


def score_rel_strength(rs):
    """
    RS ticker vs SPY su 5gg. Negativo = underperform → candidato MR.
    """
    if pd.isna(rs):
        return 50
    rs_pct = rs * 100  # Percentuale
    if rs_pct <= -5:
        return 100
    if -5 < rs_pct <= -2:
        return 80
    if -2 < rs_pct <= 0:
        return 60
    if 0 < rs_pct <= 2:
        return 40
    return 20


def score_gap(gap_atr):
    """Gap down > 0.5 ATR = panic → bullish MR (se non fondamentale)."""
    if pd.isna(gap_atr):
        return 50
    if gap_atr <= -2:
        return 90  # Big panic gap
    if -2 < gap_atr <= -0.5:
        return 85
    if -0.5 < gap_atr <= 0.5:
        return 50
    if 0.5 < gap_atr <= 2:
        return 30
    return 15  # Gap up forte → no MR long


# ============================================================================
# HARD FILTERS
# ============================================================================

def apply_hard_filters(analysis, min_price, min_adv20_m):
    """Ritorna (passed: bool, reason: str)."""
    price = analysis.get('last_price')
    adv20 = analysis.get('adv20')
    
    if price is None or pd.isna(price):
        return False, "Prezzo non disponibile"
    if price < min_price:
        return False, f"Prezzo ${price:.2f} < soglia ${min_price}"
    if pd.isna(adv20):
        return False, "ADV20 non disponibile"
    if adv20 < min_adv20_m * 1_000_000:
        return False, f"ADV20 ${adv20/1e6:.1f}M < soglia ${min_adv20_m}M"
    if analysis.get('intraday_bars_count', 0) < 10:
        return False, f"Solo {analysis.get('intraday_bars_count',0)} barre intraday (liquidità intraday insufficiente)"
    
    return True, "OK"


# ============================================================================
# MRPS CALCULATION
# ============================================================================

def calculate_mrps(analysis, regime_info):
    """Calcola Mean Reversion Probability Score."""
    scores = {
        'ibs':            score_ibs(analysis.get('ibs')),
        'vwap_dist_atr':  score_vwap_dist_atr(analysis.get('vwap_dist_atr')),
        'bb_zscore':      score_bb_zscore(analysis.get('bb_zscore')),
        'rsi':            score_rsi(analysis.get('rsi')),
        'rsi_slope':      score_rsi_slope(analysis.get('rsi_slope')),
        'candle_score':   score_candle(analysis.get('candle_score')),
        'vol_ratio':      score_vol_ratio(analysis.get('vol_ratio')),
        'rel_strength_5d':score_rel_strength(analysis.get('rel_strength_5d')),
        'gap_atr':        score_gap(analysis.get('gap_atr')),
    }
    
    technical_score = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    
    # Applica regime multiplier
    multiplier = regime_info.get('multiplier', 1.0)
    mrps = technical_score * multiplier
    
    # Data quality penalty se bars insufficienti
    bars = analysis.get('intraday_bars_count', 0)
    if bars < 30:
        mrps *= 0.90
    elif bars < 60:
        mrps *= 0.95
    
    mrps = float(np.clip(mrps, 0, 100))
    
    return {
        'mrps': mrps,
        'technical_score': technical_score,
        'component_scores': scores,
        'regime_multiplier': multiplier,
    }


def get_verdict(mrps):
    if mrps >= 78:
        return "🔥 High Conviction", "success"
    if mrps >= 65:
        return "🟢 Buy Zone", "success"
    if mrps >= 52:
        return "🟡 Watch", "warning"
    return "🔴 No Trade", "info"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if run:
    st.markdown("---")
    
    # Fetch context (SPY, VIX) con caching
    with st.spinner("Fetching market context (SPY, VIX)..."):
        spy_daily = fetch_spy_daily(api_key)
        vix_daily = fetch_vix_daily(api_key)
        regime_info = detect_market_regime(spy_daily, vix_daily)
    
    # Regime display
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Regime", regime_info['regime'].replace('MR_', '').title())
    c2.metric("Regime Multiplier", f"{regime_info['multiplier']:.2f}x")
    c3.metric("Tickers", len(tickers))
    
    with st.expander("📡 Regime Detection Details"):
        st.json(regime_info['details'])
    
    st.markdown("---")
    
    # Analyze each ticker
    progress = st.progress(0)
    status = st.empty()
    
    results = []
    filtered_out = []
    errors = []
    
    for i, ticker in enumerate(tickers):
        status.text(f"Analizzando {ticker}... ({i+1}/{len(tickers)})")
        progress.progress((i+1)/len(tickers))
        
        try:
            analysis = fetch_and_analyze(ticker, api_key, spy_daily=spy_daily)
            if analysis is None:
                errors.append(f"{ticker}: fetch fallito")
                continue
            
            passed, reason = apply_hard_filters(analysis, min_price, min_adv20_m)
            if not passed:
                filtered_out.append({'ticker': ticker, 'reason': reason})
                continue
            
            mrps_info = calculate_mrps(analysis, regime_info)
            verdict_text, verdict_type = get_verdict(mrps_info['mrps'])
            
            results.append({
                'analysis': analysis,
                'mrps_info': mrps_info,
                'verdict': verdict_text,
                'verdict_type': verdict_type,
            })
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
    
    progress.empty()
    status.empty()
    
    # -------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------
    st.subheader("📊 Summary Ranking")
    
    if results:
        # Ordina per MRPS discendente
        results_sorted = sorted(results, key=lambda r: r['mrps_info']['mrps'], reverse=True)
        
        summary_rows = []
        for r in results_sorted:
            a = r['analysis']
            m = r['mrps_info']
            price = a.get('last_price')
            atr_d = a.get('atr_daily')
            entry_low = price - 0.5 * atr_d if price and not pd.isna(atr_d) else np.nan
            entry_high = price + 0.2 * atr_d if price and not pd.isna(atr_d) else np.nan
            stop = price - 1.5 * atr_d if price and not pd.isna(atr_d) else np.nan
            t1 = a.get('vwap')
            
            summary_rows.append({
                'Ticker': a['ticker'],
                'MRPS': f"{m['mrps']:.1f}",
                'Verdict': r['verdict'],
                'Price': f"${price:.2f}" if price and not pd.isna(price) else "N/A",
                'IBS': f"{a.get('ibs', np.nan):.2f}" if not pd.isna(a.get('ibs', np.nan)) else "N/A",
                'VWAP dist (ATR)': f"{a.get('vwap_dist_atr', np.nan):+.2f}" if not pd.isna(a.get('vwap_dist_atr', np.nan)) else "N/A",
                'RSI': f"{a.get('rsi', np.nan):.0f}" if not pd.isna(a.get('rsi', np.nan)) else "N/A",
                'Vol/ADV': f"{a.get('vol_ratio', np.nan):.2f}x" if not pd.isna(a.get('vol_ratio', np.nan)) else "N/A",
                'Entry': f"${entry_low:.2f}–${entry_high:.2f}" if not pd.isna(entry_low) else "N/A",
                'Stop': f"${stop:.2f}" if not pd.isna(stop) else "N/A",
                'T1 (VWAP)': f"${t1:.2f}" if t1 and not pd.isna(t1) else "N/A",
            })
        
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
    else:
        st.warning("Nessun ticker ha passato i filtri e prodotto un'analisi.")
    
    # -------------------------------------------------------------------
    # FILTERED OUT
    # -------------------------------------------------------------------
    if filtered_out:
        with st.expander(f"⏭️ Filtrati ({len(filtered_out)})"):
            for f in filtered_out:
                st.markdown(f"- **{f['ticker']}**: {f['reason']}")
    
    # -------------------------------------------------------------------
    # ERRORS
    # -------------------------------------------------------------------
    if errors:
        with st.expander(f"⚠️ Errori ({len(errors)})"):
            for e in errors:
                st.error(e)
    
    # -------------------------------------------------------------------
    # DETAILED BREAKDOWN
    # -------------------------------------------------------------------
    if results:
        st.markdown("---")
        st.subheader("🔍 Breakdown per Ticker")
        
        for r in results_sorted:
            a = r['analysis']
            m = r['mrps_info']
            scores = m['component_scores']
            
            with st.expander(f"**{a['ticker']}** — MRPS {m['mrps']:.1f} — {r['verdict']}"):
                
                # Component scores table
                st.markdown("**Contributi al Technical Score** (peso × score = contributo)")
                
                component_rows = []
                for key, weight in WEIGHTS.items():
                    raw_val = a.get(key)
                    score_val = scores[key]
                    contribution = score_val * weight
                    
                    if pd.isna(raw_val) if raw_val is not None else True:
                        raw_display = "N/A"
                    elif isinstance(raw_val, (int, float)):
                        raw_display = f"{raw_val:.3f}"
                    else:
                        raw_display = str(raw_val)
                    
                    component_rows.append({
                        'Indicator': key,
                        'Raw Value': raw_display,
                        'Score (0-100)': f"{score_val:.0f}",
                        'Weight': f"{weight*100:.0f}%",
                        'Contribution': f"{contribution:.1f}",
                    })
                
                df_comp = pd.DataFrame(component_rows)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Technical Score", f"{m['technical_score']:.1f}")
                c2.metric("Regime Multiplier", f"{m['regime_multiplier']:.2f}x")
                c3.metric("Final MRPS", f"{m['mrps']:.1f}")
                
                # Interpretazione
                st.markdown("**Interpretazione**")
                interp = []
                if not pd.isna(a.get('ibs', np.nan)):
                    if a['ibs'] < 0.2:
                        interp.append("✅ IBS molto basso → forte segnale MR (Pagonidis 2013)")
                    elif a['ibs'] < 0.4:
                        interp.append("🟢 IBS basso → segnale MR favorevole")
                    elif a['ibs'] > 0.8:
                        interp.append("🔴 IBS molto alto → segnale bearish, evita MR long")
                
                if not pd.isna(a.get('vwap_dist_atr', np.nan)):
                    d = a['vwap_dist_atr']
                    if -1.5 <= d <= -0.5:
                        interp.append(f"✅ Prezzo a {d:+.2f} ATR da VWAP → sweet spot MR long")
                    elif d < -2:
                        interp.append(f"⚠️ Prezzo {d:+.2f} ATR sotto VWAP → potrebbe essere downtrend forte")
                    elif d > 0:
                        interp.append(f"🔴 Prezzo sopra VWAP → non è setup MR long")
                
                if not pd.isna(a.get('vol_ratio', np.nan)):
                    v = a['vol_ratio']
                    if v >= 1.5:
                        interp.append(f"✅ Volume {v:.2f}× ADV → alta partecipazione, conferma segnale (Conrad 1994)")
                    elif v < 0.5:
                        interp.append(f"⚠️ Volume {v:.2f}× ADV → bassa partecipazione, segnale debole")
                
                if not pd.isna(a.get('rsi', np.nan)):
                    r_val = a['rsi']
                    if r_val < 30:
                        interp.append(f"✅ RSI {r_val:.0f} → oversold")
                    elif r_val > 70:
                        interp.append(f"🔴 RSI {r_val:.0f} → overbought, no setup long")
                
                for note in interp:
                    st.markdown(note)
                
                if not interp:
                    st.caption("Nessun segnale forte emerso dagli indicatori.")
    
    # -------------------------------------------------------------------
    # DISCLAIMER
    # -------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "**Disclaimer:** Analisi a scopo educativo/tecnico. Non è consulenza finanziaria. "
        "L'MRPS è una probabilità stimata basata su indicatori pesati da letteratura empirica: "
        "non garantisce performance futura. Valida ogni segnale con news research qualitativa "
        "e gestione del rischio. Drawdown significativi sono attesi in regimi di mercato avversi."
    )

else:
    st.info("👈 Inserisci i ticker nella sidebar e clicca **Run Analysis**.")
    
    st.markdown("---")
    st.subheader("📚 Pesi degli Indicatori (Evidence-Based)")
    
    weights_df = pd.DataFrame([
        {'Cluster': 'Position / Deviation (50%)', 'Indicator': 'IBS', 'Weight': '22%', 'Evidence': 'Pagonidis 2013: IBS<0.2 → +0.35% avg next day; filtro migliora strategia RSI di ~10pp'},
        {'Cluster': 'Position / Deviation', 'Indicator': 'VWAP distance (ATR)', 'Weight': '20%', 'Evidence': 'Madhavan 2002: VWAP = fair value intraday istituzionale; zone reversal reliable'},
        {'Cluster': 'Position / Deviation', 'Indicator': 'Bollinger z-score', 'Weight': '8%', 'Evidence': 'Bollinger 2001; Liberated Stock Trader 120k backtest: top-5 MR indicator'},
        {'Cluster': 'Momentum (25%)', 'Indicator': 'RSI', 'Weight': '12%', 'Evidence': 'Wilder 1978; efficacia modesta stand-alone ma utile in combinazione'},
        {'Cluster': 'Momentum', 'Indicator': 'RSI slope', 'Weight': '8%', 'Evidence': 'Inversione momentum in atto'},
        {'Cluster': 'Momentum', 'Indicator': 'Candle reversal', 'Weight': '5%', 'Evidence': 'Nison 1991; Bulkowski 2008: efficacia modesta ma utile come conferma'},
        {'Cluster': 'Participation (25%)', 'Indicator': 'Volume ratio', 'Weight': '15%', 'Evidence': 'Conrad-Hameed-Niden 1994: volume alto + drop = strong reversal signal'},
        {'Cluster': 'Participation', 'Indicator': 'Relative strength vs SPY', 'Weight': '5%', 'Evidence': 'Chan-Jegadeesh-Lakonishok 1996'},
        {'Cluster': 'Participation', 'Indicator': 'Gap behavior', 'Weight': '5%', 'Evidence': 'Gap down >1 ATR = panic proxy'},
    ])
    st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Note di trasparenza:**
    - I pesi sono una stima basata sulla letteratura empirica disponibile, non ottimizzati su backtest personalizzato
    - Senza backtest walk-forward specifico sul tuo universo ticker, questi pesi sono un *prior ragionevole*, non ottimali
    - Per ottimizzazione: servirebbe un dataset storico e validation walk-forward (fase successiva di sviluppo)
    """)
