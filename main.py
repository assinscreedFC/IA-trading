# main.py
import os
import numpy as np
import pandas as pd
import logging
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from data_collector import get_market_data, get_historical_market_data
from utils import setup_logging, load_config, setup_binance_api
from database import (
    store_market_data_to_db, create_tables, get_engine, get_latest_timestamp, 
    MarketData, get_session, get_earliest_timestamp, delete_old_candles,store_indicators_to_db, 
    store_predictions_to_db
)
from indicators import calculate_rsi, calculate_stochastic_rsi, calculate_atr
from visualization import plot_all_indicators, plot_predictions, plot_reward_history
from preprocessing import prepare_data_for_lstm
from model import build_improved_model
from rewards import (
    define_reward_function_multi_step, update_model_with_rewards_multi_step, 
    log_rewards_multi_step, analyze_rewards
)

def train_with_mse(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray, epochs: int = 200, batch_size: int = 512) -> tf.keras.callbacks.History:
    """
    Entraîne le modèle avec la perte MSE en utilisant des callbacks pour l'arrêt précoce et la réduction du taux d'apprentissage.

    Args:
        model (tf.keras.Model): Modèle à entraîner.
        X_train (np.ndarray): Données d'entraînement.
        y_train (np.ndarray): Cibles d'entraînement.
        X_val (np.ndarray): Données de validation.
        y_val (np.ndarray): Cibles de validation.
        epochs (int, optional): Nombre d'époques. Par défaut à 200.
        batch_size (int, optional): Taille du lot. Par défaut à 512.

    Returns:
        tf.keras.callbacks.History: Historique de l'entraînement.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        verbose=1, 
        min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )
    return history

def main():
    # Initialisation des logs
    setup_logging()
    config = load_config()

    logging.info("Démarrage du bot de trading.")
    db_path = config.get('trading', {}).get('db_path', 'trading_bot.db')

    # Connexion à la base de données et création des tables si elles n'existent pas
    engine = get_engine(db_path)
    create_tables(engine)

    # Paramètres de trading
    symbol = config.get('trading', {}).get('symbol', 'BTCUSDT')
    interval = config.get('trading', {}).get('interval', '1m')
    limit = config.get('trading', {}).get('limit', 5000)  # Mis à jour selon config.yaml
    n_steps = config.get('trading', {}).get('n_steps', 45)
    train_ratio = config.get('trading', {}).get('train_ratio', 0.75)
    val_ratio = config.get('trading', {}).get('val_ratio', 0.2)
    n_out = config.get('trading', {}).get('n_out', 1)  # Nombre de bougies à prédire

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

    # Obtenir le nombre de bougies actuellement dans la DB
    try:
        session = get_session(engine)
        current_count = session.query(MarketData).filter(MarketData.symbol == symbol).count()
        earliest_ts = session.query(MarketData.timestamp).filter(MarketData.symbol == symbol).order_by(MarketData.timestamp.asc()).first()
        if earliest_ts:
            earliest_timestamp = earliest_ts[0]
        else:
            earliest_timestamp = None
        session.close()
        logging.info(f"Nombre actuel de bougies dans la DB pour {symbol}: {current_count}")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du nombre de bougies dans la DB : {e}")
        return

    # Calcul du nombre de bougies manquantes pour atteindre le limit
    missing_candles = limit - current_count
    if missing_candles > 0:
        logging.info(f"Il manque {missing_candles} bougies pour atteindre le limit de {limit} bougies.")
        # Définir un end_time pour la récupération historique
        if earliest_timestamp:
            end_time = earliest_timestamp - 1  # Récupérer avant la première bougie existante
        else:
            end_time = None  # Récupérer depuis la date de départ par défaut dans data_collector.py

        # Récupérer des bougies historiques pour combler le déficit
        historical_data = get_historical_market_data(
            client=binance_api,
            symbol=symbol,
            interval=interval,
            limit=missing_candles,
            end_time=end_time
        )

        if historical_data:
            print(f"Nombre de bougies historiques récupérées : {len(historical_data)}")
            df_historical = pd.DataFrame(historical_data)
            print(df_historical.head(10))

            # Stockage des bougies historiques dans la base de données
            store_market_data_to_db(historical_data, symbol, db_path)
        else:
            logging.info("Aucune bougie historique récupérée.")
    else:
        logging.info(f"La DB contient déjà au moins {limit} bougies pour {symbol}.")

    # Récupérer les nouvelles bougies pour mettre à jour la DB
    try:
        # Obtenir le dernier timestamp dans la DB
        latest_timestamp = get_latest_timestamp(symbol, db_path)
        if latest_timestamp:
            # Définir le start_time pour récupérer les bougies après le dernier timestamp
            start_time = latest_timestamp
        else:
            start_time = None  # Si la DB est vide, récupérer depuis la date de départ par défaut

        # Définir combien de bougies nous voulons récupérer pour les nouvelles données
        new_limit = max_limit = 1000  # Vous pouvez ajuster ce nombre selon vos besoins

        logging.info(f"Récupération des nouvelles données de marché pour {symbol} avec intervalle {interval} et limite {new_limit}.")
        new_market_data = get_market_data(
            client=binance_api,
            symbol=symbol,
            interval=interval,
            limit=new_limit,
            start_time=start_time
        )

        if not new_market_data:
            logging.info("Aucune nouvelle donnée récupérée.")
        else:
            print(f"Nombre de nouvelles bougies récupérées : {len(new_market_data)}")
            df_new = pd.DataFrame(new_market_data)
            print(df_new.head(10))

            # Stockage des nouvelles bougies dans la base de données
            store_market_data_to_db(new_market_data, symbol, db_path)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des nouvelles données de marché : {e}")
        return

    # Assurer que la DB ne dépasse pas le limit en supprimant les bougies les plus anciennes
    try:
        session = get_session(engine)
        total_candles = session.query(MarketData).filter(MarketData.symbol == symbol).count()
        if total_candles > limit:
            candles_to_delete = total_candles - limit
            logging.info(f"Suppression des {candles_to_delete} bougies les plus anciennes pour maintenir le limit de {limit}.")
            delete_old_candles(session, symbol, candles_to_delete)
            session.commit()
            logging.info(f"{candles_to_delete} bougies supprimées avec succès.")
        session.close()
    except Exception as e:
        logging.error(f"Erreur lors de la suppression des bougies anciennes : {e}")
        return

    try:
        # Récupérer toutes les données de marché pour le prétraitement
        session = get_session(engine)
        all_market_data = session.query(MarketData).filter(MarketData.symbol == symbol).order_by(MarketData.timestamp).all()
        session.close()

        # Convertir les données en DataFrame
        data_records = [{
            'timestamp': entry.timestamp,
            'open': entry.open,
            'high': entry.high,
            'low': entry.low,
            'close': entry.close,
            'volume': entry.volume
        } for entry in all_market_data]
        df_all = pd.DataFrame(data_records)
        logging.info(f"Total de bougies disponibles pour {symbol} : {len(df_all)}")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données de marché depuis la base de données : {e}")
        return

    # Calcul des indicateurs techniques
    try:
        df_all['rsi'] = calculate_rsi(df_all, period=14)
        stochastic_rsi = calculate_stochastic_rsi(df_all, window=14, smooth1=3, smooth2=3)
        df_all['stochastic_k'] = stochastic_rsi['stochastic_k']
        df_all['stochastic_d'] = stochastic_rsi['stochastic_d']
        df_all['atr'] = calculate_atr(df_all, window=14)
        logging.info("Indicateurs calculés.")

        # Stockage des indicateurs dans la base de données
        store_indicators_to_db(df_all[['timestamp', 'rsi', 'stochastic_k', 'stochastic_d', 'atr']], symbol, db_path)
    except Exception as e:
        logging.error(f"Erreur lors du calcul des indicateurs : {e}")
        return

    # Calcul des moyennes mobiles
    try:
        df_all['ma_court'] = df_all['close'].rolling(window=20).mean()
        df_all['ma_long'] = df_all['close'].rolling(window=50).mean()
        logging.info("Moyennes mobiles calculées (ma_court=20, ma_long=50).")
    except Exception as e:
        logging.error(f"Erreur lors du calcul des moyennes mobiles : {e}")
        return

    # Nettoyage des indicateurs et préparation des données pour le modèle LSTM
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler_features, scaler_target, df_cleaned = prepare_data_for_lstm(df_all, n_steps, n_out)
        logging.info("Données préparées pour le modèle LSTM multi-step.")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données : {e}")
        return

    # Sauvegarde des scalers
    try:
        joblib.dump(scaler_features, 'scaler_features.save')
        joblib.dump(scaler_target, 'scaler_target.save')
        logging.info("Scalers sauvegardés avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des scalers : {e}")
        return

    # Construction du modèle LSTM amélioré
    try:
        n_features = X_train.shape[2]
        model = build_improved_model(n_steps, n_features, n_out)
        logging.info(f"Modèle LSTM amélioré construit avec {n_steps} pas de temps, {n_features} caractéristiques et {n_out} sorties.")
    except Exception as e:
        logging.error(f"Erreur lors de la construction du modèle LSTM : {e}")
        return

    # PHASE 1 : Entraînement initial avec MSE
    try:
        logging.info("Phase 1: Entraînement initial avec MSE.")
        train_with_mse(
            model, 
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            epochs=config.get('model', {}).get('epochs_mse', 60), 
            batch_size=config.get('model', {}).get('batch_size_mse', 256)
        )
        logging.info("Phase 1: Entraînement avec MSE terminé.")
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement avec MSE : {e}")
        return

    # PHASE 2 : Entraînement basé sur la récompense
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('model', {}).get('learning_rate_reward', 0.00004))  # Learning rate réduit pour phase de récompense
        reward_function = define_reward_function_multi_step(n_out=n_out)

        reward_epochs = config.get('model', {}).get('epochs_reward', 40)
        batch_size = config.get('model', {}).get('batch_size_reward', 128)
        train_size = X_train.shape[0]

        logging.info("Phase 2: Entraînement basé sur la récompense.")
        reward_history = []  # Pour la visualisation
        for epoch in range(reward_epochs):
            idx = np.arange(train_size)
            np.random.shuffle(idx)
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]

            num_batches = int(np.ceil(train_size / batch_size))
            epoch_rewards = []
            for b in range(num_batches):
                start = b * batch_size
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                combined_loss, mean_reward = update_model_with_rewards_multi_step(model, X_batch, y_batch, reward_function, optimizer, n_out=n_out)
                epoch_rewards.append(mean_reward)

            epoch_mean_reward = np.mean(epoch_rewards)
            reward_history.append(epoch_mean_reward)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch+1}/{reward_epochs} (récompense): Récompense moyenne sur l'epoch = {epoch_mean_reward:.4f}")
        logging.info("Phase 2: Entraînement basé sur la récompense terminé.")
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement basé sur la récompense : {e}")
        return

    # Évaluation sur le jeu de test
    try:
        predictions_test = model.predict(X_test, batch_size=1024)
        predictions_test_inv = scaler_target.inverse_transform(predictions_test)
        y_test_inv = scaler_target.inverse_transform(y_test)

        mae_test = np.mean(np.abs(predictions_test_inv - y_test_inv))
        logging.info(f"MAE sur le test set: {mae_test}")
        print(f"MAE sur le test set: {mae_test}")
    except Exception as e:
        logging.error(f"Erreur lors de l'évaluation sur le test set : {e}")
        return

    # Log des récompenses sur le jeu de test
    try:
        test_length = len(y_test)
        test_timestamps = df_cleaned['timestamp'].iloc[-test_length:].values
        log_rewards_multi_step(
            predictions=predictions_test_inv, 
            actual_prices=y_test_inv, 
            timestamps=test_timestamps, 
            symbol=symbol, 
            db_url='rewards.db',
            n_out=n_out
        )
        logging.info("Récompenses multi-step loggées avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors du log des récompenses multi-step : {e}")
        return

    # Analyse des récompenses
    try:
        stats = analyze_rewards(db_url='rewards.db')
        print("Statistiques des récompenses :", stats)
        logging.info(f"Statistiques des récompenses : {stats}")
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse des récompenses : {e}")
        return

    # Visualisation des indicateurs
    try:
        plot_all_indicators(df_cleaned, symbol)
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation des indicateurs : {e}")

    # Visualisation des prédictions sur les jeux d'entraînement et de test
    try:
        predictions_train = model.predict(X_train, batch_size=512)
        predictions_train_inv = scaler_target.inverse_transform(predictions_train)
        y_train_inv = scaler_target.inverse_transform(y_train)

        predictions_test = model.predict(X_test, batch_size=1024)
        predictions_test_inv = scaler_target.inverse_transform(predictions_test)
        y_test_inv = scaler_target.inverse_transform(y_test)

        # Stockage des prédictions d'entraînement et de test dans la base de données
        train_timestamps = df_cleaned['timestamp'].iloc[:len(y_train)].values
        store_predictions_to_db(predictions_train_inv, train_timestamps, symbol, db_path, n_out)

        test_timestamps = df_cleaned['timestamp'].iloc[-len(y_test):].values
        store_predictions_to_db(predictions_test_inv, test_timestamps, symbol, db_path, n_out)

        plot_predictions(
            train_true=y_train_inv, 
            train_pred=predictions_train_inv, 
            test_true=y_test_inv, 
            test_pred=predictions_test_inv, 
            n_steps=n_steps, 
            n_out=n_out,
            title='Prédictions vs Valeurs Réelles (Train et Test)'
        )
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation des prédictions : {e}")

    '''### Visualisation de l'évolution des récompenses
    try:
        plot_reward_history(reward_history)
    except Exception as e:
        logging.error(f"Erreur lors de la visualisation de l'évolution des récompenses : {e}")
    '''

   # Prédiction des prochaines n_out bougies
    try:
        new_data = df_cleaned.tail(n_steps)
        new_data_input = new_data[
            ['open','high','low','close','volume','rsi','stochastic_k','stochastic_d','atr','ma_court','ma_long']
        ].values.reshape(1, n_steps, n_features)
        
        # Normaliser les nouvelles données en utilisant les scalers précédemment sauvegardés
        scaler_features = joblib.load('scaler_features.save')
        scaler_target = joblib.load('scaler_target.save')
        new_data_input_scaled = scaler_features.transform(new_data_input.reshape(-1, n_features)).reshape(1, n_steps, n_features)
        
        # Prédiction multi-step
        next_preds = model.predict(new_data_input_scaled)
        next_preds_inv = scaler_target.inverse_transform(next_preds)
        print(f"Prochaines {n_out} valeurs de clôture prédites :", next_preds_inv.flatten())
        logging.info(f"Prochaines {n_out} valeurs de clôture prédites : {next_preds_inv.flatten()}")
        
        # Stockage des prédictions multi-step
        latest_timestamp = df_cleaned['timestamp'].iloc[-1]
        prediction_timestamps = np.array([latest_timestamp])  # Passer un seul timestamp
        store_predictions_to_db(
            predictions=next_preds_inv,
            timestamps=prediction_timestamps,
            symbol=symbol,
            db_path=db_path,
            n_out=n_out
        )
        logging.info("Prédictions multi-step stockées avec succès.")
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction des prochains points futurs : {e}")

if __name__ == '__main__':
    main()