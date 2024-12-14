import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from indicators import calculate_rsi, calculate_stochastic_rsi, calculate_atr

def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Colonnes manquantes dans les données de marché : {missing_columns}")
        raise ValueError(f"Colonnes manquantes : {missing_columns}")
    
    data[required_columns] = data[required_columns].ffill()
    logging.info("Propagation avant (ffill) appliquée pour gérer les valeurs manquantes dans les données de marché.")
    
    initial_row_count = data.shape[0]
    data_cleaned = data.dropna(subset=required_columns)
    final_row_count = data_cleaned.shape[0]
    rows_dropped = initial_row_count - final_row_count
    logging.info(f"{rows_dropped} lignes supprimées après la propagation des valeurs manquantes dans les données de marché.")
    
    data_cleaned = data_cleaned.sort_values(by='timestamp').reset_index(drop=True)
    logging.info("DataFrame de marché trié par 'timestamp' en ordre croissant.")
    
    return data_cleaned

def clean_indicators_data(data: pd.DataFrame) -> pd.DataFrame:
    # CHANGEMENT: Inclure ma_court et ma_long dans les colonnes à nettoyer
    required_columns = ['rsi', 'stochastic_k', 'stochastic_d', 'atr', 'ma_court', 'ma_long']  # CHANGEMENT
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Colonnes manquantes dans les données des indicateurs : {missing_columns}")
        raise ValueError(f"Colonnes manquantes : {missing_columns}")
    
    data[required_columns] = data[required_columns].ffill()
    logging.info("Propagation avant (ffill) appliquée pour gérer les valeurs manquantes dans les indicateurs.")
    
    initial_row_count = data.shape[0]
    data_cleaned = data.dropna(subset=required_columns)
    final_row_count = data_cleaned.shape[0]
    rows_dropped = initial_row_count - final_row_count
    logging.info(f"{rows_dropped} lignes supprimées après la propagation des valeurs manquantes dans les indicateurs.")
    
    return data_cleaned

def select_features_and_target(data: pd.DataFrame) -> (np.ndarray, np.ndarray):
    # CHANGEMENT: Inclure ma_court et ma_long dans les features
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stochastic_k', 'stochastic_d', 'atr', 'ma_court', 'ma_long']  # CHANGEMENT
    target_column = 'close'
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        logging.error(f"Colonnes de caractéristiques manquantes : {missing_features}")
        raise ValueError(f"Colonnes de caractéristiques manquantes : {missing_features}")
    
    if target_column not in data.columns:
        logging.error(f"Colonne cible manquante : {target_column}")
        raise ValueError(f"Colonne cible manquante : {target_column}")
    
    features = data[feature_columns].astype(float).values
    target = data[target_column].astype(float).values
    logging.info("Caractéristiques et cible sélectionnées avec succès.")
    return features, target

def normalize_features_and_target(features: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler):
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    features_scaled = scaler_features.fit_transform(features)
    logging.info("Caractéristiques normalisées avec MinMaxScaler entre 0 et 1.")
    
    target = target.reshape(-1, 1)
    target_scaled = scaler_target.fit_transform(target)
    logging.info("Cible normalisée avec MinMaxScaler entre 0 et 1.")
    
    return features_scaled, target_scaled, scaler_features, scaler_target

def create_sequences(features_scaled: np.ndarray, target_scaled: np.ndarray, n_steps: int) -> (np.ndarray, np.ndarray):
    X, y = [], []
    for i in range(n_steps, len(features_scaled)):
        X.append(features_scaled[i - n_steps:i])
        y.append(target_scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    logging.info(f"{len(X)} séquences créées avec n_steps={n_steps}.")
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
    total_samples = X.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    logging.info("Données divisées en :")
    logging.info(f" - Entraînement : {X_train.shape[0]} échantillons")
    logging.info(f" - Validation : {X_val.shape[0]} échantillons")
    logging.info(f" - Test : {X_test.shape[0]} échantillons")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_for_lstm(data: pd.DataFrame, n_steps: int) -> tuple:
    try:
        df_cleaned = clean_market_data(data)
        logging.info("Données de marché nettoyées.")
        
        df_cleaned['rsi'] = calculate_rsi(df_cleaned, period=14)
        stochastic_rsi = calculate_stochastic_rsi(df_cleaned, window=14, smooth1=3, smooth2=3)
        df_cleaned['stochastic_k'] = stochastic_rsi['stochastic_k']
        df_cleaned['stochastic_d'] = stochastic_rsi['stochastic_d']
        df_cleaned['atr'] = calculate_atr(df_cleaned, window=14)
        logging.info("Indicateurs calculés.")

        # CHANGEMENT: Calcul des moyennes mobiles pour déterminer la tendance
        df_cleaned['ma_court'] = df_cleaned['close'].rolling(window=10).mean()  # CHANGEMENT
        df_cleaned['ma_long'] = df_cleaned['close'].rolling(window=50).mean()   # CHANGEMENT
        logging.info("Moyennes mobiles calculées (ma_court=10, ma_long=50).")

        df_cleaned = clean_indicators_data(df_cleaned)
        logging.info("Indicateurs nettoyés.")
        
        features, target = select_features_and_target(df_cleaned)
        features_scaled, target_scaled, scaler_features, scaler_target = normalize_features_and_target(features, target)
        
        X, y = create_sequences(features_scaled, target_scaled, n_steps)
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.15)
        logging.info("Données préparées pour le modèle LSTM.")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_features, scaler_target, df_cleaned
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données : {e}")
        raise
