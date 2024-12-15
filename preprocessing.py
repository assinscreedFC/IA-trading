# preprocessing.py

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from indicators import calculate_rsi, calculate_stochastic_rsi, calculate_atr

def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données de marché en gérant les valeurs manquantes et en triant par timestamp.

    Args:
        data (pd.DataFrame): DataFrame contenant les données de marché brutes.

    Returns:
        pd.DataFrame: DataFrame nettoyée avec les colonnes requises.
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Colonnes manquantes dans les données de marché : {missing_columns}")
        raise ValueError(f"Colonnes manquantes : {missing_columns}")
    
    # Propagation avant pour gérer les valeurs manquantes
    data[required_columns] = data[required_columns].ffill()
    logging.info("Propagation avant (ffill) appliquée pour gérer les valeurs manquantes dans les données de marché.")
    
    initial_row_count = data.shape[0]
    data_cleaned = data.dropna(subset=required_columns)
    final_row_count = data_cleaned.shape[0]
    rows_dropped = initial_row_count - final_row_count
    logging.info(f"{rows_dropped} lignes supprimées après la propagation des valeurs manquantes dans les données de marché.")
    
    # Tri par timestamp en ordre croissant
    data_cleaned = data_cleaned.sort_values(by='timestamp').reset_index(drop=True)
    logging.info("DataFrame de marché trié par 'timestamp' en ordre croissant.")
    
    return data_cleaned

def clean_indicators_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données des indicateurs en gérant les valeurs manquantes.

    Args:
        data (pd.DataFrame): DataFrame contenant les indicateurs bruts.

    Returns:
        pd.DataFrame: DataFrame nettoyée avec les colonnes des indicateurs.
    """
    required_columns = ['rsi', 'stochastic_k', 'stochastic_d', 'atr', 'ma_court', 'ma_long']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Colonnes manquantes dans les données des indicateurs : {missing_columns}")
        raise ValueError(f"Colonnes manquantes : {missing_columns}")
    
    # Propagation avant pour gérer les valeurs manquantes
    data[required_columns] = data[required_columns].ffill()
    logging.info("Propagation avant (ffill) appliquée pour gérer les valeurs manquantes dans les indicateurs.")
    
    initial_row_count = data.shape[0]
    data_cleaned = data.dropna(subset=required_columns)
    final_row_count = data_cleaned.shape[0]
    rows_dropped = initial_row_count - final_row_count
    logging.info(f"{rows_dropped} lignes supprimées après la propagation des valeurs manquantes dans les indicateurs.")
    
    return data_cleaned

def select_features_and_target(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sélectionne les caractéristiques et la cible pour l'entraînement du modèle.

    Args:
        data (pd.DataFrame): DataFrame contenant les données nettoyées avec indicateurs.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Matrices des caractéristiques et de la cible.
    """
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                       'rsi', 'stochastic_k', 'stochastic_d', 
                       'atr', 'ma_court', 'ma_long']
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

def normalize_features_and_target(features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Normalise les caractéristiques et la cible à l'aide de MinMaxScaler.

    Args:
        features (np.ndarray): Matrice des caractéristiques.
        target (np.ndarray): Vecteur de la cible.

    Returns:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]: 
            Caractéristiques normalisées, cible normalisée, et les scalers utilisés.
    """
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    features_scaled = scaler_features.fit_transform(features)
    logging.info("Caractéristiques normalisées avec MinMaxScaler entre 0 et 1.")
    
    target = target.reshape(-1, 1)
    target_scaled = scaler_target.fit_transform(target)
    logging.info("Cible normalisée avec MinMaxScaler entre 0 et 1.")
    
    return features_scaled, target_scaled, scaler_features, scaler_target

def create_sequences(features_scaled: np.ndarray, target_scaled: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée des séquences pour l'entraînement du modèle LSTM.

    Args:
        features_scaled (np.ndarray): Caractéristiques normalisées.
        target_scaled (np.ndarray): Cible normalisée.
        n_steps (int): Nombre de pas de temps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Séquences de caractéristiques et vecteur de cibles.
    """
    X, y = [], []
    for i in range(n_steps, len(features_scaled)):
        X.append(features_scaled[i - n_steps:i])
        y.append(target_scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    logging.info(f"{len(X)} séquences créées avec n_steps={n_steps}.")
    return X, y

def create_multi_step_sequences(features_scaled: np.ndarray, target_scaled: np.ndarray, n_steps: int, n_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée des séquences pour les prédictions multi-step du modèle LSTM.

    Args:
        features_scaled (np.ndarray): Caractéristiques normalisées.
        target_scaled (np.ndarray): Cible normalisée.
        n_steps (int): Nombre de pas de temps pour les séquences d'entrée.
        n_out (int): Nombre de pas de temps à prédire.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Séquences de caractéristiques et séquences de cibles.
    """
    X, y = [], []
    for i in range(n_steps, len(features_scaled) - n_out + 1):
        X.append(features_scaled[i - n_steps:i])
        y.append(target_scaled[i:i + n_out].flatten())
    X = np.array(X)
    y = np.array(y)
    logging.info(f"{len(X)} séquences créées avec n_steps={n_steps} et n_out={n_out}.")
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divise les données en ensembles d'entraînement, de validation et de test.

    Args:
        X (np.ndarray): Séquences de caractéristiques.
        y (np.ndarray): Vecteur de cibles.
        train_ratio (float, optional): Proportion des données pour l'entraînement. Par défaut à 0.7.
        val_ratio (float, optional): Proportion des données pour la validation. Par défaut à 0.15.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Données d'entraînement, de validation et de test.
    """
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

def split_data_multi_step(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divise les données en ensembles d'entraînement, de validation et de test pour les prédictions multi-step.

    Args:
        X (np.ndarray): Séquences de caractéristiques.
        y (np.ndarray): Séquences de cibles.
        train_ratio (float, optional): Proportion des données pour l'entraînement. Par défaut à 0.7.
        val_ratio (float, optional): Proportion des données pour la validation. Par défaut à 0.15.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Données d'entraînement, de validation et de test.
    """
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

def prepare_data_for_lstm(data: pd.DataFrame, n_steps: int, n_out: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, pd.DataFrame]:
    """
    Prépare les données pour l'entraînement du modèle LSTM en nettoyant, calculant les indicateurs,
    normalisant et créant des séquences multi-step.

    Args:
        data (pd.DataFrame): DataFrame contenant les données brutes de marché.
        n_steps (int): Nombre de pas de temps pour les séquences d'entrée.
        n_out (int): Nombre de pas de temps à prédire.

    Returns:
        Tuple contenant les ensembles d'entraînement, de validation, de test, les scalers et la DataFrame nettoyée.
    """
    try:
        df_cleaned = clean_market_data(data)
        logging.info("Données de marché nettoyées.")
        
        # Calcul des indicateurs techniques
        df_cleaned['rsi'] = calculate_rsi(df_cleaned, period=14)
        stochastic_rsi = calculate_stochastic_rsi(df_cleaned, window=14, smooth1=3, smooth2=3)
        df_cleaned['stochastic_k'] = stochastic_rsi['stochastic_k']
        df_cleaned['stochastic_d'] = stochastic_rsi['stochastic_d']
        df_cleaned['atr'] = calculate_atr(df_cleaned, window=14)
        logging.info("Indicateurs calculés.")
        
        # Calcul des moyennes mobiles
        df_cleaned['ma_court'] = df_cleaned['close'].rolling(window=10).mean()
        df_cleaned['ma_long'] = df_cleaned['close'].rolling(window=50).mean()
        logging.info("Moyennes mobiles calculées (ma_court=10, ma_long=50).")

        # Nettoyage des indicateurs
        df_cleaned = clean_indicators_data(df_cleaned)
        logging.info("Indicateurs nettoyés.")
        
        # Sélection des caractéristiques et de la cible
        features, target = select_features_and_target(df_cleaned)
        
        # Normalisation des caractéristiques et de la cible
        features_scaled, target_scaled, scaler_features, scaler_target = normalize_features_and_target(features, target)
        
        # Vérification des valeurs NaN ou Inf dans les données scalées
        if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
            logging.error("NaN ou Inf détecté dans les features_scaled")
            raise ValueError("NaN ou Inf détecté dans les features_scaled")
        if np.isnan(target_scaled).any() or np.isinf(target_scaled).any():
            logging.error("NaN ou Inf détecté dans les target_scaled")
            raise ValueError("NaN ou Inf détecté dans les target_scaled")
        logging.info("Aucune valeur NaN ou Inf détectée dans les données scalées.")
        
        # Création des séquences multi-step pour le modèle LSTM
        X, y = create_multi_step_sequences(features_scaled, target_scaled, n_steps, n_out)
        
        # Division des données en ensembles d'entraînement, de validation et de test
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_multi_step(X, y, train_ratio=0.7, val_ratio=0.15)
        logging.info("Données préparées pour le modèle LSTM multi-step.")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_features, scaler_target, df_cleaned
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données : {e}")
        raise