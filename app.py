# --- Section 1: Fonction de Calcul de l'Indicateur Canadian Confluence ---
def calculate_canadian_confluence(df, hmaLength, adxThreshold, rsiLength, adxLength, ichimokuLength, len1_smooth_ha, len2_smooth_ha):
    min_rows_needed = max(hmaLength, adxLength, rsiLength, ichimokuLength, 26, 52, len1_smooth_ha, len2_smooth_ha) + 50
    current_price_fallback = 0
    if not df.empty and 'Close' in df.columns and not df['Close'].empty:
        current_price_fallback = df['Close'].iloc[-1]
    
    if len(df) < min_rows_needed:
        return "Erreur", f"Pas assez de données ({len(df)}/{min_rows_needed})", current_price_fallback

    df_calc = df.copy()
    # Initialisation des signaux individuels
    hma_slope_signal = 0      # -1, 0, or 1
    ha_signal = 0             # -1, 0, or 1
    smoothed_ha_signal = 0    # -1, 0, or 1
    rsi_signal = 0            # -1, 0, or 1
    adx_has_momentum_signal = 0 # 0 or 1
    ichimoku_signal = 0       # -1, 0, or 1
    current_price = current_price_fallback

    # --- HMA ---
    try:
        hma_series = df_calc.ta.hma(length=hmaLength, append=False)
        if hma_series is not None and not hma_series.isna().all() and len(hma_series) >= 2:
            if hma_series.iloc[-1] > hma_series.iloc[-2]: hma_slope_signal = 1
            elif hma_series.iloc[-1] < hma_series.iloc[-2]: hma_slope_signal = -1
        else: return "Erreur HMA", "Calcul HMA impossible", current_price
    except Exception as e: return "Erreur HMA", str(e), current_price

    # --- Heiken Ashi Standard ---
    try:
        ha_df_calc = df_calc.copy() # Utiliser une copie pour éviter les modifications de df_calc par .ta.ha(append=True)
        ha_df = ha_df_calc.ta.ha(append=False) # Important: append=False
        if ha_df is not None and 'HA_close' in ha_df.columns and 'HA_open' in ha_df.columns and \
           not ha_df['HA_close'].isna().all() and not ha_df['HA_open'].isna().all() and len(ha_df) > 0:
            if ha_df['HA_close'].iloc[-1] > ha_df['HA_open'].iloc[-1]: ha_signal = 1
            elif ha_df['HA_close'].iloc[-1] < ha_df['HA_open'].iloc[-1]: ha_signal = -1
        else: return "Erreur HA", "Calcul HA impossible", current_price
    except Exception as e: return "Erreur HA", str(e), current_price

    # --- Smoothed Heiken Ashi ---
    try:
        ohlc_ema = pd.DataFrame(index=df_calc.index)
        ohlc_ema['Open'] = df_calc['Open'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['High'] = df_calc['High'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['Low'] = df_calc['Low'].ewm(span=len1_smooth_ha, adjust=False).mean()
        ohlc_ema['Close'] = df_calc['Close'].ewm(span=len1_smooth_ha, adjust=False).mean()

        ha_on_ema = pd.DataFrame(index=ohlc_ema.index)
        ha_on_ema['haclose_s1'] = (ohlc_ema['Open'] + ohlc_ema['High'] + ohlc_ema['Low'] + ohlc_ema['Close']) / 4
        ha_on_ema['haopen_s1'] = np.nan
        first_valid_idx = ohlc_ema.first_valid_index()
        if first_valid_idx is not None and not ohlc_ema.loc[first_valid_idx, ['Open', 'Close']].isna().any():
            ha_on_ema.loc[first_valid_idx, 'haopen_s1'] = (ohlc_ema.loc[first_valid_idx, 'Open'] + ohlc_ema.loc[first_valid_idx, 'Close']) / 2
        start_loop_idx = ha_on_ema['haopen_s1'].first_valid_index()
        if start_loop_idx is not None:
            start_loop_iloc = ha_on_ema.index.get_loc(start_loop_idx)
            for i_loop in range(start_loop_iloc + 1, len(ha_on_ema)): # Renommé 'i' pour éviter conflit
                prev_actual_idx, curr_actual_idx = ha_on_ema.index[i_loop-1], ha_on_ema.index[i_loop]
                if not pd.isna(ha_on_ema.loc[prev_actual_idx, 'haopen_s1']) and not pd.isna(ha_on_ema.loc[prev_actual_idx, 'haclose_s1']):
                    ha_on_ema.loc[curr_actual_idx, 'haopen_s1'] = (ha_on_ema.loc[prev_actual_idx, 'haopen_s1'] + ha_on_ema.loc[prev_actual_idx, 'haclose_s1']) / 2
                elif not ohlc_ema.loc[curr_actual_idx, ['Open', 'Close']].isna().any():
                    ha_on_ema.loc[curr_actual_idx, 'haopen_s1'] = (ohlc_ema.loc[curr_actual_idx, 'Open'] + ohlc_ema.loc[curr_actual_idx, 'Close']) / 2
        ha_on_ema.dropna(subset=['haopen_s1', 'haclose_s1'], inplace=True)
        if ha_on_ema.empty: return "Erreur HA Lissé", "Données HA_on_EMA vides", current_price
        
        smooth_ha_open = ha_on_ema['haopen_s1'].ewm(span=len2_smooth_ha, adjust=False).mean()
        smooth_ha_close = ha_on_ema['haclose_s1'].ewm(span=len2_smooth_ha, adjust=False).mean()
        
        if not smooth_ha_open.empty and not smooth_ha_close.empty and \
           not smooth_ha_open.isna().all() and not smooth_ha_close.isna().all() and \
           len(smooth_ha_open) > 0 and len(smooth_ha_close) > 0: # Ajout de vérification de longueur
            # Pine: smoothedHaSignal = o2 > c2 ? -1 : 1  (o2=smooth_ha_open, c2=smooth_ha_close)
            # Si open > close (o2 > c2) -> baissier (-1)
            # Si open < close (o2 < c2) -> haussier (1)
            if smooth_ha_open.iloc[-1] > smooth_ha_close.iloc[-1]: smoothed_ha_signal = -1 # Baissier
            elif smooth_ha_open.iloc[-1] < smooth_ha_close.iloc[-1]: smoothed_ha_signal = 1 # Haussier
        else: return "Erreur HA Lissé", "Calcul EMA HA Lissé impossible", current_price
    except Exception as e: return "Erreur HA Lissé", str(e), current_price

    # --- RSI ---
    try:
        hlc4 = (df_calc['Open'] + df_calc['High'] + df_calc['Low'] + df_calc['Close']) / 4
        rsi_series = pd_ta.rsi(close=hlc4, length=rsiLength, append=False)
        if rsi_series is not None and not rsi_series.isna().all() and len(rsi_series) > 0:
            if rsi_series.iloc[-1] > 50: rsi_signal = 1
            elif rsi_series.iloc[-1] < 50: rsi_signal = -1
        else: return "Erreur RSI", "Calcul RSI impossible", current_price
    except Exception as e: return "Erreur RSI", str(e), current_price

    # --- ADX ---
    try:
        adx_df_calc = df_calc.copy() # Utiliser une copie
        adx_df = adx_df_calc.ta.adx(length=adxLength, append=False)
        adx_col_name = f'ADX_{adxLength}'
        if adx_df is not None and adx_col_name in adx_df.columns and \
           not adx_df[adx_col_name].isna().all() and len(adx_df[adx_col_name]) > 0: # Vérifier la colonne spécifique
            adx_val = adx_df[adx_col_name].iloc[-1]
            if adx_val >= adxThreshold: adx_has_momentum_signal = 1
        else: return "Erreur ADX", f"Calcul ADX impossible (col {adx_col_name})", current_price
    except Exception as e: return "Erreur ADX", str(e), current_price

    # --- Ichimoku ---
    try:
        tenkan_period, kijun_period, senkou_b_period = ichimokuLength, 26, 52
        tenkan_sen = (df_calc['High'].rolling(window=tenkan_period).max() + df_calc['Low'].rolling(window=tenkan_period).min()) / 2
        kijun_sen = (df_calc['High'].rolling(window=kijun_period).max() + df_calc['Low'].rolling(window=kijun_period).min()) / 2
        senkou_a_current = (tenkan_sen + kijun_sen) / 2
        senkou_b_current = (df_calc['High'].rolling(window=senkou_b_period).max() + df_calc['Low'].rolling(window=senkou_b_period).min()) / 2
        
        if tenkan_sen.empty or kijun_sen.empty or senkou_a_current.empty or senkou_b_current.empty or \
           tenkan_sen.isna().all() or kijun_sen.isna().all() or senkou_a_current.isna().all() or senkou_b_current.isna().all() or \
           len(df_calc['Close']) == 0 or len(cloud_top_current) == 0 or len(cloud_bottom_current) == 0: # Ajout de vérifications
             return "Erreur Ichimoku", "Calcul lignes/cloud Ichimoku impossible", current_price

        cloud_top_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).max(axis=1)
        cloud_bottom_current = pd.concat([senkou_a_current, senkou_b_current], axis=1).min(axis=1)

        # S'assurer que les index correspondent pour iloc[-1]
        if not df_calc.index.equals(cloud_top_current.index) or not df_calc.index.equals(cloud_bottom_current.index):
            # Réindexer si nécessaire, ou gérer l'erreur. Pour l'instant, on suppose qu'ils devraient correspondre.
            # Si ce n'est pas le cas, il y a un problème de données en amont.
             pass # Pour l'instant, on laisse, mais c'est un point d'attention

        current_close_val = df_calc['Close'].iloc[-1]
        current_cloud_top_val = cloud_top_current.iloc[-1]
        current_cloud_bottom_val = cloud_bottom_current.iloc[-1]
        
        if pd.isna(current_close_val) or pd.isna(current_cloud_top_val) or pd.isna(current_cloud_bottom_val):
             return "Erreur Ichimoku", "Données cloud Ichimoku manquantes (NaN)", current_price
        if current_close_val > current_cloud_top_val: ichimoku_signal = 1
        elif current_close_val < current_cloud_bottom_val: ichimoku_signal = -1
    except Exception as e: return "Erreur Ichimoku", str(e), current_price
    
    # --- Calcul des Confluences (Aligné sur Pine Script) ---
    bullConfluences, bearConfluences = 0,0

    if hma_slope_signal == 1: bullConfluences += 1
    if ha_signal == 1: bullConfluences += 1
    if smoothed_ha_signal == 1: bullConfluences += 1 # Correspond à o2 < c2 dans Pine
    if rsi_signal == 1: bullConfluences += 1
    if adx_has_momentum_signal == 1: bullConfluences += 1 # Ajoute si momentum
    if ichimoku_signal == 1: bullConfluences +=1

    if hma_slope_signal == -1: bearConfluences += 1
    if ha_signal == -1: bearConfluences += 1
    if smoothed_ha_signal == -1: bearConfluences += 1 # Correspond à o2 > c2 dans Pine
    if rsi_signal == -1: bearConfluences += 1
    if adx_has_momentum_signal == 1: bearConfluences += 1 # Ajoute si momentum
    if ichimoku_signal == -1: bearConfluences +=1
    
    return bullConfluences, bearConfluences, current_price
  
