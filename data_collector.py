# data_collector.py

import time
from binance.client import Client
import logging
from datetime import datetime

def get_market_data(client: Client, symbol: str, interval: str, limit: int, start_time: int = None) -> list:
    """
    Récupère les données historiques OHLCV d'une paire de trading à partir de l'API Binance.
    Gère la pagination pour récupérer jusqu'à la limite spécifiée.

    Args:
        client (binance.client.Client): Instance de l'API Binance.
        symbol (str): La paire de trading (ex. 'BTCUSDT').
        interval (str): L'intervalle des données (ex. '1m', '5m', '1h').
        limit (int): Le nombre total de points de données à récupérer.
        start_time (int, optional): Timestamp en millisecondes pour démarrer la récupération. Par défaut à None.

    Returns:
        list of dict: Liste des données OHLCV contenant 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    try:
        max_limit = 1000  # Limite maximale par requête
        klines = []
        remaining = limit
        current_start_time = start_time

        # Si aucun start_time n'est fourni, définir une date de départ antérieure
        if current_start_time is None or current_start_time == 0:
            # Binance supporte les données historiques pour BTCUSDT depuis le 17 août 2017
            current_start_time = int(datetime(2017, 8, 17).timestamp() * 1000)  # Timestamp en millisecondes

        while remaining > 0:
            fetch_limit = min(max_limit, remaining)
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': fetch_limit,
                'startTime': current_start_time
            }
            fetched = client.get_klines(**params)
            if not fetched:
                logging.info("Aucune bougie récupérée, arrêt de la récupération.")
                break
            klines += fetched
            last_kline = fetched[-1]
            current_start_time = last_kline[0] + 1  # Avancer d'une milliseconde pour éviter les duplications
            remaining -= len(fetched)
            logging.info(f"{len(fetched)} bougies récupérées, {remaining} restantes.")
            # Pause pour respecter les limites de taux de l'API
            time.sleep(0.5)
        
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
        logging.info(f"Récupération des données de marché pour {symbol} avec intervalle {interval} et limite {limit}. Total récupéré: {len(market_data)}")
        return market_data
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données de marché : {e}")
        return []

def get_historical_market_data(client: Client, symbol: str, interval: str, limit: int, end_time: int = None) -> list:
    """
    Récupère les données historiques OHLCV d'une paire de trading en remontant dans le temps à partir d'un end_time.
    Gère la pagination pour récupérer jusqu'à la limite spécifiée.

    Args:
        client (binance.client.Client): Instance de l'API Binance.
        symbol (str): La paire de trading (ex. 'BTCUSDT').
        interval (str): L'intervalle des données (ex. '1m', '5m', '1h').
        limit (int): Le nombre total de points de données à récupérer.
        end_time (int, optional): Timestamp en millisecondes pour terminer la récupération. Par défaut à None.

    Returns:
        list of dict: Liste des données OHLCV contenant 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    try:
        max_limit = 1000  # Limite maximale par requête
        klines = []
        remaining = limit
        current_end_time = end_time

        # Si aucun end_time n'est fourni, définir une date de fin à l'instant actuel
        if current_end_time is None:
            current_end_time = int(time.time() * 1000)

        while remaining > 0:
            fetch_limit = min(max_limit, remaining)
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': fetch_limit,
                'endTime': current_end_time
            }
            fetched = client.get_klines(**params)
            if not fetched:
                logging.info("Aucune bougie récupérée, arrêt de la récupération.")
                break
            klines = fetched + klines  # Ajouter au début pour maintenir l'ordre chronologique
            first_kline = fetched[0]
            current_end_time = first_kline[0] - 1  # Reculer d'une milliseconde pour éviter les duplications
            remaining -= len(fetched)
            logging.info(f"{len(fetched)} bougies récupérées, {remaining} restantes.")
            # Pause pour respecter les limites de taux de l'API
            time.sleep(0.5)
        
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
        logging.info(f"Récupération des données historiques de marché pour {symbol} avec intervalle {interval} et limite {limit}. Total récupéré: {len(market_data)}")
        return market_data
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données historiques de marché : {e}")
        return []