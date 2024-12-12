# indicators.py
import pandas as pd
import logging
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def calculate_rsi(data, period=14):
    """
    Calcule le Relative Strength Index (RSI) pour les données de clôture fournies.

    Args:
        data (pd.DataFrame): DataFrame contenant au moins une colonne 'close'.
        period (int): Période sur laquelle calculer le RSI. Par défaut, 14.

    Returns:
        pd.Series: Série contenant les valeurs RSI.
    """
    if 'close' not in data.columns:
        logging.error("La colonne 'close' est manquante dans les données fournies.")
        raise ValueError("La colonne 'close' est requise pour calculer le RSI.")

    try:
        rsi_indicator = RSIIndicator(close=data['close'], window=period)
        rsi = rsi_indicator.rsi()
        logging.info(f"RSI calculé avec succès sur une période de {period}.")
        return rsi
    except Exception as e:
        logging.error(f"Erreur lors du calcul du RSI : {e}")
        raise

def calculate_stochastic_rsi(data, window=14, smooth1=3, smooth2=3):
    """
    Calcule le Stochastic RSI (stochastic_k et stochastic_d) pour les données de clôture fournies.

    Args:
        data (pd.DataFrame): DataFrame contenant au moins une colonne 'close'.
        window (int): Période de calcul du RSI. Par défaut, 14.
        smooth1 (int): Période de lissage stochastic_k. Par défaut, 3.
        smooth2 (int): Période de lissage stochastic_d. Par défaut, 3.

    Returns:
        pd.DataFrame: DataFrame contenant les colonnes 'stochastic_k' et 'stochastic_d'.
    """
    if 'close' not in data.columns:
        logging.error("La colonne 'close' est manquante dans les données fournies.")
        raise ValueError("La colonne 'close' est requise pour calculer le Stochastic RSI.")

    try:
        # Calcul du RSI
        rsi = calculate_rsi(data, period=window)

        # Calcul du Stochastic RSI
        stoch_rsi = (rsi - rsi.rolling(window=window).min()) / (rsi.rolling(window=window).max() - rsi.rolling(window=window).min())
        stoch_rsi = stoch_rsi * 100  # Convertir en pourcentage

        # Calcul des moyennes mobiles pour stochastic_k et stochastic_d
        stochastic_k = stoch_rsi.rolling(window=smooth1).mean()
        stochastic_d = stochastic_k.rolling(window=smooth2).mean()

        # Ajout des colonnes au DataFrame
        data['stochastic_k'] = stochastic_k
        data['stochastic_d'] = stochastic_d

        logging.info(f"Stochastic RSI calculé avec succès avec window={window}, smooth1={smooth1}, smooth2={smooth2}.")
        return data[['stochastic_k', 'stochastic_d']]
    except Exception as e:
        logging.error(f"Erreur lors du calcul du Stochastic RSI : {e}")
        raise

def calculate_atr(data, window=14):
    """
    Calcule l'Average True Range (ATR) pour les données de marché fournies.

    Args:
        data (pd.DataFrame): DataFrame contenant au moins les colonnes 'high', 'low', et 'close'.
        window (int): Période sur laquelle calculer l'ATR. Par défaut, 14.

    Returns:
        pd.Series: Série contenant les valeurs ATR.
    """
    required_columns = ['high', 'low', 'close']
    for col in required_columns:
        if col not in data.columns:
            logging.error(f"La colonne '{col}' est manquante dans les données fournies.")
            raise ValueError(f"La colonne '{col}' est requise pour calculer l'ATR.")

    try:
        atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=window)
        atr = atr_indicator.average_true_range()
        data['atr'] = atr
        logging.info(f"ATR calculé avec succès sur une période de {window}.")
        return atr
    except Exception as e:
        logging.error(f"Erreur lors du calcul de l'ATR : {e}")
        raise