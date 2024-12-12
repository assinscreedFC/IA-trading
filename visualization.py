# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import logging

def plot_close_price(df, symbol):
    """
    Trace le prix de clôture.

    Args:
        df (pd.DataFrame): DataFrame contenant au moins les colonnes 'timestamp' et 'close'.
        symbol (str): Symbole de la paire de trading (ex. 'BTCUSDT').
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['close'], label='Clôture', color='blue')
        plt.title(f'{symbol} - Prix de Clôture')
        plt.xlabel('Timestamp')
        plt.ylabel('Prix de Clôture (USDT)')
        plt.legend()
        plt.grid(True)
        plt.show()
        logging.info("Graphique des prix de clôture affiché avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation des prix de clôture : {e}")

def plot_rsi(df, symbol):
    """
    Trace le RSI.

    Args:
        df (pd.DataFrame): DataFrame contenant au moins les colonnes 'timestamp' et 'rsi'.
        symbol (str): Symbole de la paire de trading (ex. 'BTCUSDT').
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['rsi'], label='RSI', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Suracheté (70)')
        plt.axhline(30, color='green', linestyle='--', label='Survendu (30)')
        plt.title(f'{symbol} - Relative Strength Index (RSI)')
        plt.xlabel('Timestamp')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.show()
        logging.info("Graphique du RSI affiché avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation du RSI : {e}")

def plot_stochastic_rsi_k_d(df, symbol):
    """
    Trace le Stochastic RSI (stochastic_k et stochastic_d) uniquement.

    Args:
        df (pd.DataFrame): DataFrame contenant au moins les colonnes 'timestamp', 'stochastic_k', et 'stochastic_d'.
        symbol (str): Symbole de la paire de trading (ex. 'BTCUSDT').
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['stochastic_k'], label='stochastic_k', color='blue')
        plt.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['stochastic_d'], label='stochastic_d', color='red')
        plt.axhline(80, color='red', linestyle='--', label='Suracheté (80)')
        plt.axhline(20, color='green', linestyle='--', label='Survendu (20)')
        plt.title(f'{symbol} - Stochastic RSI (stochastic_k & stochastic_d)')
        plt.xlabel('Timestamp')
        plt.ylabel('Stochastic RSI (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        logging.info("Graphique du Stochastic RSI (stochastic_k & stochastic_d) affiché avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation du Stochastic RSI (stochastic_k & stochastic_d) : {e}")

def plot_atr(df, symbol):
    """
    Trace l'Average True Range (ATR).

    Args:
        df (pd.DataFrame): DataFrame contenant au moins les colonnes 'timestamp' et 'atr'.
        symbol (str): Symbole de la paire de trading (ex. 'BTCUSDT').
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['atr'], label='ATR', color='magenta')
        plt.title(f'{symbol} - Average True Range (ATR)')
        plt.xlabel('Timestamp')
        plt.ylabel('ATR')
        plt.legend()
        plt.grid(True)
        plt.show()
        logging.info("Graphique de l'ATR affiché avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation de l'ATR : {e}")

def plot_all_indicators(df, symbol):
    """
    Trace le prix de clôture, le RSI, le Stochastic RSI et l'ATR sur une seule figure avec quatre sous-graphiques alignés verticalement.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'timestamp', 'close', 'rsi', 'stochastic_k', 'stochastic_d', et 'atr'.
        symbol (str): Symbole de la paire de trading (ex. 'BTCUSDT').
    """
    try:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 24), sharex=True)
        
        # Tracer le prix de clôture
        ax1.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['close'], label='Clôture', color='blue')
        ax1.set_title(f'{symbol} - Prix de Clôture')
        ax1.set_ylabel('Prix de Clôture (USDT)')
        ax1.legend()
        ax1.grid(True)
        
        # Tracer le RSI
        ax2.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['rsi'], label='RSI', color='orange')
        ax2.axhline(70, color='red', linestyle='--', label='Suracheté (70)')
        ax2.axhline(30, color='green', linestyle='--', label='Survendu (30)')
        ax2.set_title(f'{symbol} - Relative Strength Index (RSI)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
        
        # Tracer le Stochastic RSI (stochastic_k et stochastic_d)
        ax3.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['stochastic_k'], label='stochastic_k', color='blue')
        ax3.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['stochastic_d'], label='stochastic_d', color='red')
        ax3.axhline(80, color='red', linestyle='--', label='Suracheté (80)')
        ax3.axhline(20, color='green', linestyle='--', label='Survendu (20)')
        ax3.set_title(f'{symbol} - Stochastic RSI (stochastic_k & stochastic_d)')
        ax3.set_ylabel('Stochastic RSI (%)')
        ax3.legend()
        ax3.grid(True)
        
        # Tracer l'ATR
        ax4.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['atr'], label='ATR', color='magenta')
        ax4.set_title(f'{symbol} - Average True Range (ATR)')
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('ATR')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        logging.info("Graphique combiné des indicateurs affiché avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation combinée des indicateurs : {e}")