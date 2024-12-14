import pandas as pd
import matplotlib.pyplot as plt
import logging

def plot_all_indicators(df, symbol):
    try:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 24), sharex=True)
        
        ax1.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['close'], label='Clôture', color='blue')
        ax1.set_title(f'{symbol} - Prix de Clôture')
        ax1.set_ylabel('Prix de Clôture (USDT)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['rsi'], label='RSI', color='orange')
        ax2.axhline(70, color='red', linestyle='--', label='Suracheté (70)')
        ax2.axhline(30, color='green', linestyle='--', label='Survendu (30)')
        ax2.set_title(f'{symbol} - Relative Strength Index (RSI)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['stochastic_k'], label='stochastic_k', color='blue')
        ax3.plot(pd.to_datetime(df['timestamp'], unit='ms'), df['stochastic_d'], label='stochastic_d', color='red')
        ax3.axhline(80, color='red', linestyle='--', label='Suracheté (80)')
        ax3.axhline(20, color='green', linestyle='--', label='Survendu (20)')
        ax3.set_title(f'{symbol} - Stochastic RSI (stochastic_k & stochastic_d)')
        ax3.set_ylabel('Stochastic RSI (%)')
        ax3.legend()
        ax3.grid(True)
        
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
