
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf  # Pour vérifier les GPU

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
from preprocessing import (
    prepare_data_for_lstm
)
from model import build_lstm_model, train_lstm_model  # Import de la fonction build_lstm_model et train_lstm_model

def plot_predictions(model, X_test, y_test, scaler_target, df_cleaned, n_steps):
    """
    Effectue des prédictions sur l'ensemble de test et les visualise par rapport aux valeurs réelles.

    Args:
        model (tf.keras.Model): Modèle entraîné.
        X_test (np.ndarray): Séquences de test.
        y_test (np.ndarray): Cibles de test.
        scaler_target (MinMaxScaler): Scaler utilisé pour normaliser la cible.
        df_cleaned (pd.DataFrame): DataFrame nettoyé contenant les indicateurs.
        n_steps (int): Nombre de pas de temps utilisés dans les séquences.
    """
    # Effectuer les prédictions
    predictions = model.predict(X_test)
    
    # Inverser la normalisation
    predictions = scaler_target.inverse_transform(predictions)
    y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    
    # Créer un DataFrame pour la visualisation
    test_indices = df_cleaned.index[-len(y_test):]
    test_dates = df_cleaned['timestamp'].iloc[test_indices].values
    df_plot = pd.DataFrame({
        'Timestamp': test_dates,
        'Valeur Réelle': y_test.flatten(),
        'Prédiction': predictions.flatten()
    })
    
    # Convertir le timestamp en datetime
    df_plot['Timestamp'] = pd.to_datetime(df_plot['Timestamp'], unit='ms')
    
    # Visualiser
    plt.figure(figsize=(14, 7))
    plt.plot(df_plot['Timestamp'], df_plot['Valeur Réelle'], label='Valeur Réelle')
    plt.plot(df_plot['Timestamp'], df_plot['Prédiction'], label='Prédiction')
    plt.title('Prédictions du Modèle LSTM vs Valeurs Réelles')
    plt.xlabel('Date')
    plt.ylabel('Prix de Clôture')
    plt.legend()
    plt.show()

def main():
    # Initialiser le logging
    setup_logging()
    config = load_config()
    
    logging.info("Démarrage du bot de trading.")
    
    # Vérifier les GPU disponibles
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"{len(gpus)} GPU(s) détecté(s) : {gpus}")
    else:
        logging.warning("Aucun GPU détecté. L'entraînement se fera sur le CPU.")
    
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
    limit = 10000  # Augmenter la limite pour plus de données
    
    logging.info(f"Récupération des données de marché pour {symbol} avec intervalle {interval} et limite {limit}.")
    market_data = get_market_data(binance_api, symbol, interval, limit)
    
    # Conversion des données en DataFrame pandas
    if market_data:
        print(f"Nombre de données récupérées : {len(market_data)}")
        df = pd.DataFrame(market_data)
        print(df.head(10))  # Affiche les 10 premières lignes pour un aperçu
        
        # Préparation des données pour LSTM
        try:
            n_steps = 30  # Réduire le nombre de pas de temps pour plus de séquences
            X_train, y_train, X_val, y_val, X_test, y_test, scaler_features, scaler_target, df_cleaned = prepare_data_for_lstm(df, n_steps)
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            logging.info(f"Colonnes disponibles dans df_cleaned : {df_cleaned.columns.tolist()}")
        except Exception as e:
            logging.error(f"Erreur lors de la préparation des données pour LSTM : {e}")
            return
        
        # Sauvegarder les scalers pour une utilisation future
        try:
            joblib.dump(scaler_features, 'scaler_features.save')
            joblib.dump(scaler_target, 'scaler_target.save')
            logging.info("Scalers sauvegardés sous 'scaler_features.save' et 'scaler_target.save'.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des scalers : {e}")
            return
        
        # Construction du modèle LSTM
        try:
            n_features = X_train.shape[2]  # Nombre de caractéristiques
            model = build_lstm_model(n_steps, n_features)
            logging.info("Modèle LSTM construit et compilé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la construction du modèle LSTM : {e}")
            return
        
        # Entraînement du modèle avec callbacks
        try:
            epochs = 100  # Nombre d'époques d'entraînement, ajustez selon vos besoins
            batch_size = 128  # Taille du batch, ajustez selon vos besoins
            
            model = train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
            logging.info("Modèle LSTM entraîné avec succès avec callbacks.")
        except Exception as e:
            logging.error(f"Erreur lors de l'entraînement du modèle LSTM : {e}")
            return
        
        # Évaluation du modèle sur l'ensemble de test
        try:
            loss = model.evaluate(X_test, y_test, verbose=0)
            logging.info(f"Performance du modèle sur l'ensemble de test : {loss}")
            print(f"Performance du modèle sur l'ensemble de test : {loss}")
        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation du modèle LSTM : {e}")
            return
        
        # Visualiser les prédictions
        try:
            plot_predictions(model, X_test, y_test, scaler_target, df_cleaned, n_steps)
            logging.info("Prédictions visualisées avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la visualisation des prédictions : {e}")
        
        # Stocker les indicateurs dans la base de données
        try:
            # Vérifier les types de données des indicateurs
            for col in ['rsi', 'stochastic_k', 'stochastic_d', 'atr']:
                if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    logging.error(f"La colonne {col} n'est pas de type numérique.")
                    raise TypeError(f"La colonne {col} n'est pas de type numérique.")
            
            # Sélectionner les indicateurs à stocker
            indicators_to_store = df_cleaned[['timestamp', 'rsi', 'stochastic_k', 'stochastic_d', 'atr']].dropna()
            store_indicators_to_db(indicators_to_store, symbol, db_path)
            logging.info("Indicateurs stockés dans la base de données avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors du stockage des indicateurs dans la base de données : {e}")
            return
        
        # Visualisation des données et des indicateurs
        try:
            plot_all_indicators(df_cleaned, symbol)
            logging.info("Graphique combiné des indicateurs affiché avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la visualisation des données : {e}")

if __name__ == '__main__':
    main()