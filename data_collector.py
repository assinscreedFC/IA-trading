# data_collector.py
from binance.client import Client
import logging

def get_market_data(client: Client, symbol: str, interval: str, limit: int):
    """
    Récupère les données historiques OHLCV d'une paire de trading à partir de l'API Binance.

    Args:
        client (binance.client.Client): Instance de l'API Binance.
        symbol (str): La paire de trading (ex. 'BTCUSDT').
        interval (str): L'intervalle des données (ex. '1m', '5m', '1h').
        limit (int): Le nombre de points de données à récupérer.

    Returns:
        list of dict: Liste des données OHLCV contenant 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        market_data = []
        for kline in klines:
            data_point = {
                'timestamp': kline[0],         # Open time in milliseconds
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            }
            market_data.append(data_point)
        logging.info(f"Récupération des données de marché pour {symbol} avec intervalle {interval} et limite {limit}.")
        return market_data
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données de marché : {e}")
        return []
