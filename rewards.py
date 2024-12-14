import numpy as np
import logging
import sqlite3
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Reward(Base):
    __tablename__ = 'rewards'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    predicted_close = Column(Float, nullable=False)
    actual_close = Column(Float, nullable=False)
    reward = Column(Float, nullable=False)

def define_reward_function():
    """
    Fonction de récompense améliorée pour inclure les moyennes mobiles.
    Retourne des récompenses positives pour de bonnes prédictions et négatives pour les mauvaises.
    """
    def reward_function(predictions, actuals, last_input_closes, last_rsi, last_stoch_k, last_ma_court, last_ma_long):
        # Calcul de l'erreur absolue et récompense de base
        errors = tf.abs(predictions - actuals)
        reward_base = 1.0 - errors  # Plus l'erreur est faible, plus la récompense est élevée

        # Calcul des différences et des signes
        actual_diff = actuals - last_input_closes
        pred_diff = predictions - last_input_closes

        actual_sign = tf.sign(actual_diff)
        pred_sign = tf.sign(pred_diff)

        # Facteur de direction
        direction_correct = tf.equal(actual_sign, pred_sign)
        direction_factor = tf.where(direction_correct, 0.1, -0.1)

        final_reward = reward_base + direction_factor

        # Conditions RSI/Stoch
        overbought_rsi = last_rsi > 70
        oversold_rsi = last_rsi < 30
        overbought_stoch = last_stoch_k > 80
        oversold_stoch = last_stoch_k < 20

        # Ajustements RSI/Stoch
        rsi_adj = tf.zeros_like(final_reward)
        rsi_adj = tf.where(overbought_rsi & (pred_sign > 0), rsi_adj - 0.05, rsi_adj)
        rsi_adj = tf.where(overbought_rsi & (pred_sign <= 0), rsi_adj + 0.05, rsi_adj)
        rsi_adj = tf.where(oversold_rsi & (pred_sign < 0), rsi_adj - 0.05, rsi_adj)
        rsi_adj = tf.where(oversold_rsi & (pred_sign >= 0), rsi_adj + 0.05, rsi_adj)

        stoch_adj = tf.zeros_like(final_reward)
        stoch_adj = tf.where(overbought_stoch & (pred_sign > 0), stoch_adj - 0.05, stoch_adj)
        stoch_adj = tf.where(overbought_stoch & (pred_sign <= 0), stoch_adj + 0.05, stoch_adj)
        stoch_adj = tf.where(oversold_stoch & (pred_sign < 0), stoch_adj - 0.05, stoch_adj)
        stoch_adj = tf.where(oversold_stoch & (pred_sign >= 0), stoch_adj + 0.05, stoch_adj)

        final_reward += (rsi_adj + stoch_adj)

        # Détermination de la tendance via MA
        trend_condition = tf.sign(last_ma_court - last_ma_long)  # +1 haussier, -1 baissier, 0 neutre
        bullish_rsi_stoch = (last_rsi > 50) & (last_stoch_k > 50)
        bearish_rsi_stoch = (last_rsi < 50) & (last_stoch_k < 50)

        trend_adj = tf.zeros_like(final_reward)
        # Tendance haussière confirmée
        confirmed_bullish = (trend_condition > 0) & bullish_rsi_stoch
        # Tendance baissière confirmée
        confirmed_bearish = (trend_condition < 0) & bearish_rsi_stoch

        trend_adj = tf.where(
            confirmed_bullish & (actual_sign > 0) & (pred_sign > 0),
            trend_adj + 0.07,
            trend_adj
        )
        trend_adj = tf.where(
            confirmed_bearish & (actual_sign < 0) & (pred_sign < 0),
            trend_adj + 0.07,
            trend_adj
        )

        final_reward += trend_adj

        # Conditions extrêmes
        both_overheated = (last_rsi > 80) & (last_stoch_k > 80)
        both_oversold = (last_rsi < 20) & (last_stoch_k < 20)

        strong_up_threshold = 0.005
        strong_down_threshold = -0.005

        strong_up_condition = pred_diff > (strong_up_threshold * last_input_closes)
        strong_down_condition = pred_diff < (strong_down_threshold * last_input_closes)

        # Conditions de reversal
        reversal_condition_overbought = both_overheated & (actual_sign > 0) & (pred_sign <= 0)
        reversal_condition_oversold = both_oversold & (actual_sign < 0) & (pred_sign >= 0)

        final_reward = tf.where(reversal_condition_overbought, final_reward - 0.05, final_reward)
        final_reward = tf.where(reversal_condition_oversold, final_reward - 0.05, final_reward)

        # Punition pour prédictions extrêmes
        final_reward = tf.where(both_overheated & strong_up_condition, final_reward - 0.1, final_reward)
        final_reward = tf.where(both_oversold & strong_down_condition, final_reward - 0.1, final_reward)

        # Assurer que les récompenses sont dans une plage raisonnable
        final_reward = tf.clip_by_value(final_reward, -1.0, 1.0)

        return final_reward
    return reward_function

def update_model_with_rewards(model, X, y, reward_function, optimizer):
    close_index = 3
    rsi_index = 5
    stoch_k_index = 6
    ma_court_index = 9
    ma_long_index = 10

    last_input_closes = X[:, -1, close_index:close_index+1]
    last_rsi = X[:, -1, rsi_index:rsi_index+1] * 100.0 
    last_stoch_k = X[:, -1, stoch_k_index:stoch_k_index+1] * 100.0
    last_ma_court = X[:, -1, ma_court_index:ma_court_index+1]
    last_ma_long = X[:, -1, ma_long_index:ma_long_index+1]

    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        rewards = reward_function(
            y_pred, 
            tf.constant(y, dtype=tf.float32), 
            tf.constant(last_input_closes, dtype=tf.float32),
            tf.constant(last_rsi, dtype=tf.float32),
            tf.constant(last_stoch_k, dtype=tf.float32),
            tf.constant(last_ma_court, dtype=tf.float32),
            tf.constant(last_ma_long, dtype=tf.float32)
        )
        mean_reward = tf.reduce_mean(rewards)
        loss = -mean_reward  # Maximiser les récompenses
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss.numpy(), mean_reward.numpy()

def log_rewards(predictions, actual_prices, timestamps, symbol='BTCUSDT', db_url='rewards.db'):
    # Utiliser la même fonction de récompense pour le logging
    # Normalement, cela devrait être similaire à la fonction utilisée pendant l'entraînement
    # Pour simplifier, utiliser 1 - erreur comme récompense
    errors = np.abs(predictions - actual_prices)
    rewards = 1.0 - errors  # Assurer que les récompenses sont positives

    engine = create_engine(f'sqlite:///{db_url}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        reward_objects = [
            Reward(
                symbol=symbol,
                timestamp=int(ts),
                predicted_close=float(predictions[i]),
                actual_close=float(actual_prices[i]),
                reward=float(rewards[i])
            )
            for i, ts in enumerate(timestamps)
        ]
        
        session.bulk_save_objects(reward_objects)
        session.commit()
        logging.info(f"{len(reward_objects)} récompenses insérées dans la base de données {db_url}.")
    except Exception as e:
        session.rollback()
        logging.error(f"Erreur lors de l'insertion des récompenses : {e}")
    finally:
        session.close()

def analyze_rewards(db_url='rewards.db'):
    engine = create_engine(f'sqlite:///{db_url}', echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        rewards_data = session.query(Reward).all()
        if not rewards_data:
            logging.info("Aucune récompense enregistrée.")
            return {}
        
        df = pd.DataFrame([{
            'symbol': r.symbol,
            'timestamp': r.timestamp,
            'predicted_close': r.predicted_close,
            'actual_close': r.actual_close,
            'reward': r.reward
        } for r in rewards_data])

        mean_reward = df['reward'].mean()
        median_reward = df['reward'].median()
        std_reward = df['reward'].std()

        stats = {
            'mean_reward': mean_reward,
            'median_reward': median_reward,
            'std_reward': std_reward
        }

        logging.info(f"Analyse des récompenses : {stats}")
        return stats
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse des récompenses : {e}")
        return {}
    finally:
        session.close()