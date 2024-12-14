import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, BatchNormalization, Dense, LayerNormalization

def build_lstm_model(n_steps: int, n_features: int) -> tf.keras.Model:
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(n_steps, n_features))))
    model.add(Dropout(rate=0.2))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(rate=0.2))
    model.add(LayerNormalization())
    
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(rate=0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())
    
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(BatchNormalization())
    
    model.add(Dense(units=1))
    
    return model