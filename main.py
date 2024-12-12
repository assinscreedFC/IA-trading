# main.py
from data_collector import get_market_data
from utils import setup_logging, load_config, setup_binance_api
from database import (
    store_market_data_to_db, 
    store_indicators_to_db, 
    create_tables, 
    get_engine
)
from indicators import (
    calculate_rsi, 
    calculate_stochastic_rsi, 
    calculate_atr
)
from visualization import plot_all_indicators
import logging
import pandas as pd

def main():
    # Initialiser le logging
    setup_logging()
    config = load_config()
    
    logging.info("Démarrage du bot de trading.")
    
    # Récupérer le chemin de la base de données depuis la configuration
    db_path = config.get('db_path', 'trading_bot.db')
    
    # Créer l'engine et les tables (market_data et indicators)
    engine = get_engine(db_path)
    create_tables(engine)
    
    # Initialisation de l'API Binance
    try:
        binance_api = setup_binance_api(config)
        if binance_api:
            logging.info("API Binance initialisée avec succès.")
        else:
            logging.info("API Binance non initialisée.")
    except Exception as e:
        logging.error(f"Échec de l'initialisation de l'API Binance : {e}")
        return
    
    # Récupération des données de marché
    symbol = config.get('symbol', 'BTCUSDT')
    interval = config.get('interval', '1m')
    limit = 100  # Vous pouvez ajuster la limite selon vos besoins
    
    logging.info(f"Récupération des données de marché pour {symbol} avec intervalle {interval} et limite {limit}.")
    market_data = get_market_data(binance_api, symbol, interval, limit)
    
    # Conversion des données en DataFrame pandas
    if market_data:
        print(f"Nombre de données récupérées : {len(market_data)}")
        df = pd.DataFrame(market_data)
        print(df.head(10))  # Affiche les 10 premières lignes pour un aperçu
        
        # Stocker les données dans la base de données
        store_market_data_to_db(market_data, symbol, db_path)
        
        # Calcul des indicateurs
        try:
            df['rsi'] = calculate_rsi(df, period=14)
            logging.info("RSI calculé avec succès.")
            print(df[['timestamp', 'close', 'rsi']].tail(10))  # Affiche les 10 dernières valeurs RSI
        except Exception as e:
            logging.error(f"Erreur lors du calcul du RSI : {e}")
            df['rsi'] = None  # Assignation d'une valeur par défaut en cas d'erreur
        
        try:
            stoch_rsi = calculate_stochastic_rsi(df, window=14, smooth1=3, smooth2=3)
            df['stochastic_k'] = stoch_rsi['stochastic_k']
            df['stochastic_d'] = stoch_rsi['stochastic_d']
            logging.info("Stochastic RSI calculé avec succès.")
            print(df[['timestamp', 'close', 'stochastic_k', 'stochastic_d']].tail(10))  # Affiche les 10 dernières valeurs StochRSI
        except Exception as e:
            logging.error(f"Erreur lors du calcul du Stochastic RSI : {e}")
            df['stochastic_k'] = None
            df['stochastic_d'] = None
        
        try:
            df['atr'] = calculate_atr(df, window=14)
            logging.info("ATR calculé avec succès.")
            print(df[['timestamp', 'close', 'atr']].tail(10))  # Affiche les 10 dernières valeurs ATR
        except Exception as e:
            logging.error(f"Erreur lors du calcul de l'ATR : {e}")
            df['atr'] = None
        
        # Préparer les données à stocker (exclure les lignes avec des valeurs manquantes)
        indicators_to_store = df[['timestamp', 'rsi', 'stochastic_k', 'stochastic_d', 'atr']].dropna()
        
        # Stocker les indicateurs dans la base de données
        store_indicators_to_db(indicators_to_store, symbol, db_path)
        
        # Visualisation des données
        try:
            plot_all_indicators(df, symbol)
        except Exception as e:
            logging.error(f"Erreur lors de la visualisation des données : {e}")
    else:
        logging.warning("Aucune donnée de marché récupérée.")
    
    # Suite de votre logique de trading...
    # Par exemple, implémenter des indicateurs techniques supplémentaires, générer des signaux, etc.

if __name__ == '__main__':
    main()
