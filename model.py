import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional

def build_improved_model(n_steps: int, n_features: int, n_out: int) -> tf.keras.Model:
    """
    Construit et retourne un modèle LSTM amélioré avec des couches Bidirectional et GRU pour les prédictions multi-step.

    Args:
        n_steps (int): Nombre de pas de temps dans chaque séquence.
        n_features (int): Nombre de caractéristiques dans chaque pas de temps.
        n_out (int): Nombre de pas de temps à prédire.

    Returns:
        tf.keras.Model: Modèle LSTM amélioré construit.
    """
    model = Sequential()
    # Première couche Bidirectional LSTM
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), input_shape=(n_steps, n_features)))
    model.add(Dropout(0.3))
    
    # Deuxième couche GRU
    model.add(GRU(64, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.3))
    
    # Troisième couche GRU
    model.add(GRU(32, activation='tanh'))
    model.add(Dropout(0.3))
    
    # Couches Dense
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_out))  # Prédiction de n_out valeurs de clôture
    
    # Compilation du modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    model.summary()
    return model