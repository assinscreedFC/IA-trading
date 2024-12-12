# model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_lstm_model(n_steps: int, n_features: int) -> tf.keras.Model:
    """
    Construit et compile un modèle LSTM séquentiel selon l'architecture spécifiée.

    Args:
        n_steps (int): Nombre de pas de temps (longueur des séquences).
        n_features (int): Nombre de caractéristiques (features) par pas de temps.

    Returns:
        tf.keras.Model: Modèle LSTM compilé prêt à être entraîné.
    """
    model = Sequential()
    
    # Première couche LSTM
    model.add(LSTM(units=150, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(rate=0.3))
    model.add(BatchNormalization())
    
    # Deuxième couche LSTM
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(rate=0.3))
    model.add(BatchNormalization())
    
    # Troisième couche LSTM
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(rate=0.3))
    model.add(BatchNormalization())
    
    # Couche Dense intermédiaire
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(rate=0.3))
    
    # Couche de sortie
    model.add(Dense(units=1))
    
    # Compilation du modèle
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.summary()  # Affiche le résumé du modèle pour vérification
    
    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=128) -> tf.keras.Model:
    """
    Entraîne le modèle LSTM avec des callbacks pour EarlyStopping et ModelCheckpoint.

    Args:
        model (tf.keras.Model): Modèle LSTM compilé.
        X_train (np.ndarray): Ensemble d'entraînement des séquences.
        y_train (np.ndarray): Cibles d'entraînement.
        X_val (np.ndarray): Ensemble de validation des séquences.
        y_val (np.ndarray): Cibles de validation.
        epochs (int, optional): Nombre d'époques d'entraînement. Par défaut à 100.
        batch_size (int, optional): Taille du batch. Par défaut à 128.

    Returns:
        tf.keras.Model: Modèle LSTM entraîné avec les meilleurs poids chargés.
    """
    # Configurer les callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_lstm_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    # Charger le meilleur modèle sauvegardé
    best_model = load_model('best_lstm_model.keras')
    
    return best_model