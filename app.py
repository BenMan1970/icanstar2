import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import time

# Configuration de la page
st.set_page_config(
    page_title="Scanner Confluence Forex Premium",
    page_icon="⭐",
    layout="wide"
)

st.title("🔍 Scanner Confluence Forex Premium")
st.markdown("*Filtrage automatique 5-6 étoiles sur les paires forex et XAU/USD*")

# Liste des paires forex principales + XAU/USD
FOREX_PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X',
    'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'AUDJPY=X', 'EURAUD=X',
    'EURCHF=X', 'AUDNZD=X', 'NZDJPY=X', 'GBPAUD=X', 'GBPCAD=X', 'EURNZD=X',
    'AUDCAD=X', 'GBPCHF=X', 'CADCHF=X', 'EURCAD=X', 'AUDCHF=X', 'NZDCAD=X',
    'NZDCHF=X', 'GC=F'  # XAU/USD (Gold)
]

# Paramètres par défaut (identiques au script TradingView)
DEFAULT_PARAMS = {
    'hma_length': 20,
    'adx_threshold': 20,
    'rsi_length': 10,
    'adx_length': 14,
    'ichimoku_length': 9,
    'smoothed_ha_len1': 10,
    'smoothed_ha_len2': 10
}

def get_forex_data(symbol, period='1d', interval='1h'):
    """Récupère les données forex avec gestion d'erreurs"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Erreur pour {symbol}: {str(e)}")
        return None

def calculate_hma(data, length):
    """Calcule Hull Moving Average"""
    try:
        close = data['Close'].values
        wma_half = talib.WMA(close, int(length/2))
        wma_full = talib.WMA(close, length)
        diff = 2 * wma_half - wma_full
        hma = talib.WMA(diff, int(np.sqrt(length)))
        return hma
    except:
        return np.full(len(data), np.nan)

def calculate_heiken_ashi(data):
    """Calcule les chandelles Heiken Ashi"""
    ha_close = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    ha_open = np.zeros(len(data))
    ha_open[0] = (data['Open'].iloc[0] + data['Close'].iloc[0]) / 2
    
    for i in range(1, len(data)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    
    return ha_close, ha_open

def calculate_smoothed_heiken_ashi(data, len1, len2):
    """Calcule les chandelles Heiken Ashi lissées"""
    try:
        # Première étape de lissage
        o = data['Open'].ewm(span=len1).mean()
        c = data['Close'].ewm(span=len1).mean()
        h = data['High'].ewm(span=len1).mean()
        l = data['Low'].ewm(span=len1).mean()
        
        ha_close = (o + h + l + c) / 4
        ha_open = np.zeros(len(data))
        ha_open[0] = (o.iloc[0] + c.iloc[0]) / 2
        
        for i in range(1, len(data)):
            ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
        
        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))
        
        # Deuxième étape de lissage
        o2 = pd.Series(ha_open).ewm(span=len2).mean()
        c2 = ha_close.ewm(span=len2).mean()
        
        return o2.values, c2.values
    except:
        return np.full(len(data), np.nan), np.full(len(data), np.nan)

def calculate_ichimoku(data, length=9):
    """Calcule les nuages Ichimoku"""
    try:
        # Tenkan-sen
        tenkan = (data['High'].rolling(length).max() + data['Low'].rolling(length).min()) / 2
        
        # Kijun-sen
        kijun = (data['High'].rolling(26).max() + data['Low'].rolling(26).min()) / 2
        
        # Senkou Span A
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B
        senkou_b = (data['High'].rolling(52).max() + data['Low'].rolling(52).min()) / 2
        
        cloud_top = np.maximum(senkou_a, senkou_b)
        cloud_bottom = np.minimum(senkou_a, senkou_b)
        
        return cloud_top.values, cloud_bottom.values
    except:
        return np.full(len(data), np.nan), np.full(len(data), np.nan)

def calculate_confluence_signals(data, params):
    """Calcule tous les signaux de confluence"""
    if data is None or len(data) < 60:
        return None
    
    try:
        # 1. HMA Signal
        hma = calculate_hma(data, params['hma_length'])
        hma_slope = 1 if hma[-1] > hma[-2] else -1
        
        # 2. Heiken Ashi Signal
        ha_close, ha_open = calculate_heiken_ashi(data)
        ha_signal = 1 if ha_close.iloc[-1] > ha_open[-1] else -1
        
        # 3. Smoothed Heiken Ashi Signal
        sha_open, sha_close = calculate_smoothed_heiken_ashi(data, params['smoothed_ha_len1'], params['smoothed_ha_len2'])
        sha_signal = 1 if sha_close[-1] > sha_open[-1] else -1
        
        # 4. RSI Signal
        rsi_source = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        rsi = talib.RSI(rsi_source.values, timeperiod=params['rsi_length'])
        rsi_signal = 1 if rsi[-1] > 50 else -1
        
        # 5. ADX Signal
        adx = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=params['adx_length'])
        adx_has_momentum = adx[-1] >= params['adx_threshold']
        
        # 6. Ichimoku Signal
        cloud_top, cloud_bottom = calculate_ichimoku(data, params['ichimoku_length'])
        close_price = data['Close'].iloc[-1]
        if close_price > cloud_top[-1]:
            ichimoku_signal = 1
        elif close_price < cloud_bottom[-1]:
            ichimoku_signal = -1
        else:
            ichimoku_signal = 0
        
        # Calcul des confluences
        bull_confluences = 0
        bear_confluences = 0
        
        bull_confluences += 1 if hma_slope == 1 else 0
        bull_confluences += 1 if ha_signal == 1 else 0
        bull_confluences += 1 if sha_signal == 1 else 0
        bull_confluences += 1 if rsi_signal == 1 else 0
        bull_confluences += 1 if adx_has_momentum else 0
        bull_confluences += 1 if ichimoku_signal == 1 else 0
        
        bear_confluences += 1 if hma_slope == -1 else 0
        bear_confluences += 1 if ha_signal == -1 else 0
        bear_confluences += 1 if sha_signal == -1 else 0
        bear_confluences += 1 if rsi_signal == -1 else 0
        bear_confluences += 1 if adx_has_momentum else 0
        bear_confluences += 1 if ichimoku_signal == -1 else 0
        
        confluence = max(bull_confluences, bear_confluences)
        direction = "HAUSSIER" if bull_confluences > bear_confluences else "BAISSIER"
        
        return {
            'confluence': confluence,
            'direction': direction,
            'bull_confluences': bull_confluences,
            'bear_confluences': bear_confluences,
            'rsi': rsi[-1],
            'adx': adx[-1],
            'adx_momentum': adx_has_momentum,
            'signals': {
                'HMA': "▲" if hma_slope == 1 else "▼",
                'HA': "▲" if ha_signal == 1 else "▼",
                'SHA': "▲" if sha_signal == 1 else "▼",
                'RSI': "▲" if rsi_signal == 1 else "▼",
                'ADX': "✔" if adx_has_momentum else "✖",
                'Ichimoku': "▲" if ichimoku_signal == 1 else "▼" if ichimoku_signal == -1 else "─"
            }
        }
    except Exception as e:
        st.error(f"Erreur calcul confluence: {str(e)}")
        return None

def get_rating_stars(confluence):
    """Retourne le nombre d'étoiles selon la confluence"""
    if confluence == 6:
        return "⭐⭐⭐⭐⭐⭐"
    elif confluence == 5:
        return "⭐⭐⭐⭐⭐"
    elif confluence == 4:
        return "⭐⭐⭐⭐"
    elif confluence == 3:
        return "⭐⭐⭐"
    elif confluence == 2:
        return "⭐⭐"
    elif confluence == 1:
        return "⭐"
    else:
        return "WAIT"

def main():
    # Sidebar pour les paramètres
    st.sidebar.header("⚙️ Paramètres")
    
    # Sélection de la timeframe
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1h", "4h", "1d"],
        index=0
    )
    
    # Filtrage minimum
    min_stars = st.sidebar.selectbox(
        "Filtrage minimum",
        [5, 6],
        index=0
    )
    
    # Bouton de scan
    if st.sidebar.button("🔍 Scanner les paires", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, symbol in enumerate(FOREX_PAIRS):
            progress = (i + 1) / len(FOREX_PAIRS)
            progress_bar.progress(progress)
            status_text.text(f"Analyse de {symbol}... ({i+1}/{len(FOREX_PAIRS)})")
            
            # Récupération des données
            data = get_forex_data(symbol, period='5d', interval=timeframe)
            
            if data is not None:
                # Calcul des signaux
                signals = calculate_confluence_signals(data, DEFAULT_PARAMS)
                
                if signals and signals['confluence'] >= min_stars:
                    pair_name = symbol.replace('=X', '').replace('=F', ' (Gold)')
                    results.append({
                        'Paire': pair_name,
                        'Direction': signals['direction'],
                        'Confluence': signals['confluence'],
                        'Étoiles': get_rating_stars(signals['confluence']),
                        'RSI': f"{signals['rsi']:.1f}",
                        'ADX': f"{signals['adx']:.1f}",
                        'HMA': signals['signals']['HMA'],
                        'Heiken Ashi': signals['signals']['HA'],
                        'HA Lissé': signals['signals']['SHA'],
                        'RSI Signal': signals['signals']['RSI'],
                        'ADX Momentum': signals['signals']['ADX'],
                        'Ichimoku': signals['signals']['Ichimoku']
                    })
            
            time.sleep(0.1)  # Éviter les limites de taux
        
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats
        if results:
            st.success(f"🎯 {len(results)} paire(s) trouvée(s) avec {min_stars}+ étoiles!")
            
            # Tri par confluence descendante
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Confluence', ascending=False)
            
            # Affichage avec style
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Section détaillée
            st.subheader("📊 Détails des signaux")
            for _, row in results_df.iterrows():
                with st.expander(f"🔹 {row['Paire']} - {row['Étoiles']} ({row['Direction']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("RSI", row['RSI'])
                        st.metric("ADX", row['ADX'])
                    
                    with col2:
                        st.write("**Signaux techniques:**")
                        st.write(f"• HMA: {row['HMA']}")
                        st.write(f"• Heiken Ashi: {row['Heiken Ashi']}")
                        st.write(f"• HA Lissé: {row['HA Lissé']}")
                    
                    with col3:
                        st.write("**Confirmations:**")
                        st.write(f"• RSI: {row['RSI Signal']}")
                        st.write(f"• ADX: {row['ADX Momentum']}")
                        st.write(f"• Ichimoku: {row['Ichimoku']}")
        else:
            st.warning(f"❌ Aucune paire trouvée avec {min_stars}+ étoiles pour le moment.")
            st.info("💡 Essayez avec un filtrage plus bas ou une timeframe différente.")
    
    # Informations sur les indicateurs
    with st.expander("ℹ️ Informations sur les indicateurs"):
        st.markdown("""
        **Indicateurs utilisés (identiques au script TradingView):**
        
        • **HMA (Hull Moving Average)** - Longueur: 20
        • **Heiken Ashi** - Chandelles lissées
        • **Heiken Ashi Smoothed** - Double lissage (10/10)
        • **RSI** - Longueur: 10 (sur OHLC/4)
        • **ADX** - Longueur: 14, Seuil: 20
        • **Ichimoku** - Longueur: 9
        
        **Système de notation:**
        - ⭐⭐⭐⭐⭐⭐ (6/6) = Signal très fort
        - ⭐⭐⭐⭐⭐ (5/6) = Signal fort
        - Filtrage automatique sur 5-6 étoiles uniquement
        """)

if __name__ == "__main__":
    main()
   
