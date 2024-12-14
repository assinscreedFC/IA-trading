import numpy as np
import pandas as pd
import logging
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from data_collector import get_market_data
from utils import setup_logging, load_config, setup_binance_api
from database import (
    store_market_data_to_db, 
    store_indicators_to_db,
    create_tables, 
    get_engine
)
from indicators import calculate_rsi, calculate_stochastic_rsi, calculate_atr
from visualization import plot_all_indicators
from preprocessing import prepare_data_for_lstm
from model import build_lstm_model
from rewards import define_reward_function, update_model_with_rewards, log_rewards, analyze_rewards

def train_with_mse(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=512):
    """
    Entraînement initial du modèle avec MSE.
    Utilisation d'early stopping et reduce_lr pour stabiliser.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='mse')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-5)

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
    setup_logging()
    config = load_config()

    logging.info("Démarrage du bot de trading.")
    db_path = config.get('db_path', 'trading_bot.db')

    engine = get_engine(db_path)
    create_tables(engine)

    symbol = config.get('symbol', 'BTCUSDT')
    interval = config.get('interval', '1m')
    limit = 100000  # Taille de dataset raisonnable

    try:
        binance_api = setup_binance_api(config)
        if binance_api:
            logging.info("API Binance initialisée avec succès.")
        else:
            logging.info("API Binance non initialisée.")
    except Exception as e:
        logging.error(f"Échec de l'initialisation de l'API Binance : {e}")
        return

    logging.info(f"Récupération des données de marché pour {symbol} avec intervalle {interval} et limite {limit}.")
    market_data = get_market_data(binance_api, symbol, interval, limit)

    if market_data is None or len(market_data) == 0:
        logging.error("Aucune donnée récupérée.")
        return

    print(f"Nombre de données récupérées : {len(market_data)}")
    df = pd.DataFrame(market_data)
    print(df.head(10))

    store_market_data_to_db(market_data, symbol, db_path)

    try:
        n_steps = 64  # Moins que 128, pour un équilibre entre réactivité et tendance
        X_train, y_train, X_val, y_val, X_test, y_test, scaler_features, scaler_target, df_cleaned = prepare_data_for_lstm(df, n_steps)
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données : {e}")
        return

    # Sauvegarde des scalers
    joblib.dump(scaler_features, 'scaler_features.save')
    joblib.dump(scaler_target, 'scaler_target.save')

    # Construction du modèle amélioré
    n_features = X_train.shape[2]
    model = build_lstm_model(n_steps, n_features)

    # PHASE 1 : MSE
    logging.info("Phase 1: Entraînement initial avec MSE.")
    train_with_mse(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=512)

    # PHASE 2 : Récompense
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001/2) # On met 0.00005 si on veut encore plus fin 
    reward_function = define_reward_function()

    reward_epochs = 300
    batch_size = 1024
    train_size = X_train.shape[0]

    logging.info("Phase 2: Entraînement basé sur la récompense.")
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
            y_batch = y_train_shuffled[start:end].reshape(-1,1)

            loss_val, mean_reward = update_model_with_rewards(model, X_batch, y_batch, reward_function, optimizer)
            epoch_rewards.append(mean_reward)

        logging.info(f"Epoch {epoch+1}/{reward_epochs} (récompense): Récompense moyenne sur l'epoch = {np.mean(epoch_rewards)}")

    # Évaluation
    predictions_test = model(X_test, training=False).numpy()
    predictions_test_inv = scaler_target.inverse_transform(predictions_test)
    y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1,1))

    mae_test = np.mean(np.abs(predictions_test_inv - y_test_inv))
    logging.info(f"MAE sur le test set: {mae_test}")
    print(f"MAE sur le test set: {mae_test}")

    # Log récompenses
    test_length = len(y_test)
    test_timestamps = df_cleaned['timestamp'].iloc[-test_length:].values
    log_rewards(predictions=predictions_test_inv.flatten(), actual_prices=y_test_inv.flatten(),
                timestamps=test_timestamps, symbol=symbol, db_url='rewards.db')

    stats = analyze_rewards(db_url='rewards.db')
    print("Statistiques des récompenses :", stats)

    plot_all_indicators(df_cleaned, symbol)

    # Visualisation prédictions Train/Test
    predictions_train = model(X_train, training=False).numpy()
    predictions_train_inv = scaler_target.inverse_transform(predictions_train)
    y_train_inv = scaler_target.inverse_transform(y_train.reshape(-1,1))

    plt.figure(figsize=(14,7))
    plt.plot(y_train_inv, label='Train Réel', color='green')
    plt.plot(predictions_train_inv, label='Train Prédit', color='orange')

    offset = len(y_train_inv)
    plt.plot(np.arange(offset, offset+len(y_test_inv)), y_test_inv, label='Test Réel', color='blue')
    plt.plot(np.arange(offset, offset+len(y_test_inv)), predictions_test_inv, label='Test Prédit', color='red')

    plt.title('Prédictions vs Valeurs Réelles (Train et Test)')
    plt.xlabel('Index échantillon (approx)')
    plt.ylabel('Prix de Clôture (Dénormalisé)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prédiction du prochain point futur
    new_data = df_cleaned.tail(n_steps)
    new_data_input = new_data[['open','high','low','close','volume','rsi','stochastic_k','stochastic_d','atr']].values.reshape(1,n_steps,n_features)
    next_pred = model(tf.constant(new_data_input, dtype=tf.float32))
    next_pred_inv = scaler_target.inverse_transform(next_pred.numpy())
    print("Prochaine valeur de clôture prédite :", next_pred_inv[0,0])

if __name__ == '__main__':
    main()