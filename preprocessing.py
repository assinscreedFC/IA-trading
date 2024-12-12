import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from indicators import calculate_rsi, calculate_stochastic_rsi, calculate_atr

def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et pré-traite les données de marché de base.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les données de marché brutes.
    
    Returns:
        pd.DataFrame: DataFrame nettoyé et trié.
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    
    # Vérifier les colonnes manquantes
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Colonnes manquantes dans les données de marché : {missing_columns}")
        raise ValueError(f"Colonnes manquantes : {missing_columns}")
    
    # Remplir les valeurs manquantes avec la propagation avant
    data[required_columns] = data[required_columns].ffill()
    logging.info("Propagation avant (ffill) appliquée pour gérer les valeurs manquantes dans les données de marché.")
    
    # Supprimer les lignes restantes avec des valeurs manquantes
    initial_row_count = data.shape[0]
    data_cleaned = data.dropna(subset=required_columns)
    final_row_count = data_cleaned.shape[0]
    rows_dropped = initial_row_count - final_row_count
    logging.info(f"{rows_dropped} lignes supprimées après la propagation des valeurs manquantes dans les données de marché.")
    
    # Trier les données par timestamp en ordre croissant
    data_cleaned = data_cleaned.sort_values(by='timestamp').reset_index(drop=True)
    logging.info("DataFrame de marché trié par 'timestamp' en ordre croissant.")
    
    return data_cleaned

def clean_indicators_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et pré-traite les colonnes des indicateurs.
    
    Args:
        data (pd.DataFrame): DataFrame contenant les indicateurs calculés.
    
    Returns:
        pd.DataFrame: DataFrame nettoyé avec les indicateurs.
    """
    required_columns = ['rsi', 'stochastic_k', 'stochastic_d', 'atr']
    
    # Vérifier les colonnes manquantes
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Colonnes manquantes dans les données des indicateurs : {missing_columns}")
        raise ValueError(f"Colonnes manquantes : {missing_columns}")
    
    # Remplir les valeurs manquantes avec la propagation avant
    data[required_columns] = data[required_columns].ffill()
    logging.info("Propagation avant (ffill) appliquée pour gérer les valeurs manquantes dans les indicateurs.")
    
    # Supprimer les lignes restantes avec des valeurs manquantes
    initial_row_count = data.shape[0]
    data_cleaned = data.dropna(subset=required_columns)
    final_row_count = data_cleaned.shape[0]
    rows_dropped = initial_row_count - final_row_count
    logging.info(f"{rows_dropped} lignes supprimées après la propagation des valeurs manquantes dans les indicateurs.")
    
    return data_cleaned

def select_features_and_target(data: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Sélectionne les caractéristiques (features) et la cible (target) à partir des données.
    
    Args:
        data (pd.DataFrame): DataFrame nettoyé contenant les caractéristiques et la cible.
    
    Returns:
        tuple: (features, target) où
            - features (np.ndarray): Tableau des caractéristiques sélectionnées.
            - target (np.ndarray): Tableau de la cible sélectionnée.
    """
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'stochastic_k', 'stochastic_d', 'atr']
    target_column = 'close'
    
    # Vérifier les colonnes manquantes
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        logging.error(f"Colonnes de caractéristiques manquantes : {missing_features}")
        raise ValueError(f"Colonnes de caractéristiques manquantes : {missing_features}")
    
    if target_column not in data.columns:
        logging.error(f"Colonne cible manquante : {target_column}")
        raise ValueError(f"Colonne cible manquante : {target_column}")
    
    # Extraire les caractéristiques et la cible
    features = data[feature_columns].astype(float).values
    target = data[target_column].astype(float).values
    
    logging.info("Caractéristiques et cible sélectionnées avec succès.")
    
    return features, target

def normalize_features_and_target(features: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler):
    """
    Normalise les caractéristiques et la cible en utilisant MinMaxScaler.
    
    Args:
        features (np.ndarray): Tableau des caractéristiques à normaliser.
        target (np.ndarray): Tableau de la cible à normaliser.
    
    Returns:
        tuple: (features_scaled, target_scaled, scaler_features, scaler_target)
            - features_scaled (np.ndarray): Caractéristiques normalisées.
            - target_scaled (np.ndarray): Cible normalisée.
            - scaler_features (MinMaxScaler): Objet scaler utilisé pour les caractéristiques.
            - scaler_target (MinMaxScaler): Objet scaler utilisé pour la cible.
    """
    # Initialiser les scalers
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    # Normaliser les caractéristiques
    features_scaled = scaler_features.fit_transform(features)
    logging.info("Caractéristiques normalisées avec MinMaxScaler entre 0 et 1.")
    
    # Normaliser la cible (reshape nécessaire pour scaler_target)
    target = target.reshape(-1, 1)
    target_scaled = scaler_target.fit_transform(target)
    logging.info("Cible normalisée avec MinMaxScaler entre 0 et 1.")
    
    return features_scaled, target_scaled, scaler_features, scaler_target

def create_sequences(features_scaled: np.ndarray, target_scaled: np.ndarray, n_steps: int) -> (np.ndarray, np.ndarray):
    """
    Crée des séquences temporelles pour le modèle LSTM.
    
    Args:
        features_scaled (np.ndarray): Tableau des caractéristiques normalisées.
        target_scaled (np.ndarray): Tableau de la cible normalisée.
        n_steps (int): Longueur des séquences.
    
    Returns:
        tuple: (X, y) où
            - X (np.ndarray): Tableau de séquences de forme (num_sequences, n_steps, num_features).
            - y (np.ndarray): Tableau de cibles correspondantes de forme (num_sequences,).
    """
    X, y = [], []
    for i in range(n_steps, len(features_scaled)):
        X.append(features_scaled[i - n_steps:i])
        y.append(target_scaled[i, 0])  # La cible est déjà normalisée et en 1D
    X = np.array(X)
    y = np.array(y)
    logging.info(f"{len(X)} séquences créées avec n_steps={n_steps}.")
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
    """
    Divise les données en ensembles d'entraînement, de validation et de test selon les ratios spécifiés.

    Args:
        X (np.ndarray): Tableau des séquences d'entrée.
        y (np.ndarray): Tableau des cibles correspondantes.
        train_ratio (float, optional): Proportion des données pour l'entraînement. Par défaut à 0.7.
        val_ratio (float, optional): Proportion des données pour la validation. Par défaut à 0.15.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
            - X_train (np.ndarray): Séquences pour l'entraînement.
            - y_train (np.ndarray): Cibles pour l'entraînement.
            - X_val (np.ndarray): Séquences pour la validation.
            - y_val (np.ndarray): Cibles pour la validation.
            - X_test (np.ndarray): Séquences pour le test.
            - y_test (np.ndarray): Cibles pour le test.
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

    logging.info(f"Données divisées en :")
    logging.info(f" - Entraînement : {X_train.shape[0]} échantillons")
    logging.info(f" - Validation : {X_val.shape[0]} échantillons")
    logging.info(f" - Test : {X_test.shape[0]} échantillons")

    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_for_lstm(data: pd.DataFrame, n_steps: int) -> tuple:
    """
    Prépare les données pour le modèle LSTM en exécutant toutes les étapes de prétraitement.

    Args:
        data (pd.DataFrame): DataFrame contenant les données de marché brutes.
        n_steps (int): Longueur des séquences pour le modèle LSTM.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, scaler_features, scaler_target, df_cleaned)
    """
    try:
        # Nettoyer les données de marché
        df_cleaned = clean_market_data(data)
        logging.info("Données de marché nettoyées.")
        
        # Calcul des indicateurs
        df_cleaned['rsi'] = calculate_rsi(df_cleaned, period=14)
        stochastic_rsi = calculate_stochastic_rsi(df_cleaned, window=14, smooth1=3, smooth2=3)
        df_cleaned['stochastic_k'] = stochastic_rsi['stochastic_k']
        df_cleaned['stochastic_d'] = stochastic_rsi['stochastic_d']
        df_cleaned['atr'] = calculate_atr(df_cleaned, window=14)
        logging.info("Indicateurs calculés.")
        
        # Nettoyer les indicateurs
        df_cleaned = clean_indicators_data(df_cleaned)
        logging.info("Indicateurs nettoyés.")
        
        # Sélectionner les caractéristiques et la cible
        features, target = select_features_and_target(df_cleaned)
        logging.info("Caractéristiques et cible sélectionnées.")
        
        # Normalisation des données
        features_scaled, target_scaled, scaler_features, scaler_target = normalize_features_and_target(features, target)
        logging.info("Données normalisées.")
        
        # Création des séquences
        X, y = create_sequences(features_scaled, target_scaled, n_steps)
        logging.info("Séquences créées.")
        
        # Séparation des données
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.15)
        logging.info("Données séparées en ensembles d'entraînement, de validation et de test.")
        
        # Vérifier les colonnes disponibles dans df_cleaned
        logging.info(f"Colonnes disponibles dans df_cleaned : {df_cleaned.columns.tolist()}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_features, scaler_target, df_cleaned
    except Exception as e:
        logging.error(f"Erreur lors de la préparation des données : {e}")
        raise