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
    Reward function améliorée pour essayer de capturer:
    - Changement de tendance (reversal)
    - Continuation de tendance
    - Conditions extrêmes et patterns de reversal (surachat/survente)
    - Patterns "modérés" indiquant une compréhension des points de retournement

    Logique:
    - Base: reward_base = 1 - abs(pred-actual)
    - Direction: +0.1 si direction correcte, -0.1 sinon
    - Ajustements RSI/Stoch comme avant
    - Tendance claire haussière: RSI>50 & Stoch>50
    - Tendance claire baissière: RSI<50 & Stoch<50
    - Ajustements pour continuation modérée dans le sens de la tendance: +0.05
    - Ajustements pour reversals:
      * Si surachat extrême (RSI>80,Stoch>80) et on vient d'un marché haussier (actual_sign>0),
        alors prédire stable ou légère baisse => +0.05 (anticipation de reversal)
      * Si survente extrême (RSI<20,Stoch<20) et on vient d'un marché baissier (actual_sign<0),
        alors prédire stable ou légère hausse => +0.05 (anticipation de reversal)
    - Forte contradiction avec conditions extrêmes:
      * Surachat extrême mais on prédit une forte hausse => -0.1
      * Survente extrême mais on prédit une forte baisse => -0.1
    """
    def reward_function(predictions, actuals, last_input_closes, last_rsi, last_stoch_k):
        errors = tf.abs(predictions - actuals)
        reward_base = 1.0 - errors

        actual_diff = actuals - last_input_closes
        pred_diff = predictions - last_input_closes

        actual_sign = tf.sign(actual_diff)
        pred_sign = tf.sign(pred_diff)

        # Direction factor
        direction_correct = tf.equal(actual_sign, pred_sign)
        direction_factor = tf.where(direction_correct, 0.1, -0.1)

        final_reward = reward_base + direction_factor

        # Thresholds pour stable/moderate/strong moves
        stable_threshold = 0.0005
        moderate_threshold = 0.001
        strong_up_threshold = 0.005

        stable_condition = tf.abs(pred_diff) < (stable_threshold * (last_input_closes+1e-6))
        moderate_condition = tf.abs(pred_diff) < (moderate_threshold * (last_input_closes+1e-6))

        pred_direction_for_rsi = tf.where(stable_condition, tf.zeros_like(pred_sign), pred_sign)

        # Conditions RSI/Stoch
        overbought_rsi = last_rsi > 70
        oversold_rsi = last_rsi < 30
        overbought_stoch = last_stoch_k > 80
        oversold_stoch = last_stoch_k < 20

        # Ajustements RSI/Stoch initiaux
        rsi_adj = tf.zeros_like(final_reward)
        rsi_adj = tf.where(overbought_rsi & (pred_direction_for_rsi > 0), rsi_adj - 0.05, rsi_adj)
        rsi_adj = tf.where(overbought_rsi & (pred_direction_for_rsi <= 0), rsi_adj + 0.05, rsi_adj)
        rsi_adj = tf.where(oversold_rsi & (pred_direction_for_rsi < 0), rsi_adj - 0.05, rsi_adj)
        rsi_adj = tf.where(oversold_rsi & (pred_direction_for_rsi >= 0), rsi_adj + 0.05, rsi_adj)

        stoch_adj = tf.zeros_like(final_reward)
        stoch_adj = tf.where(overbought_stoch & (pred_direction_for_rsi > 0), stoch_adj - 0.05, stoch_adj)
        stoch_adj = tf.where(overbought_stoch & (pred_direction_for_rsi <= 0), stoch_adj + 0.05, stoch_adj)
        stoch_adj = tf.where(oversold_stoch & (pred_direction_for_rsi < 0), stoch_adj - 0.05, stoch_adj)
        stoch_adj = tf.where(oversold_stoch & (pred_direction_for_rsi >= 0), stoch_adj + 0.05, stoch_adj)

        final_reward += (rsi_adj + stoch_adj)

        # Tendance claire
        bullish_trend = (last_rsi > 50) & (last_stoch_k > 50)
        bearish_trend = (last_rsi < 50) & (last_stoch_k < 50)

        # Continuation de tendance modérée
        # Si bullish trend et actual_sign>0 => marché haussier réel
        # prédire légère hausse (moderate_condition & pred_sign>0) => +0.05
        trend_adj = tf.zeros_like(final_reward)
        trend_adj = tf.where(bullish_trend & (actual_sign > 0) & (pred_sign > 0) & moderate_condition, trend_adj+0.05, trend_adj)
        trend_adj = tf.where(bearish_trend & (actual_sign < 0) & (pred_sign < 0) & moderate_condition, trend_adj+0.05, trend_adj)

        # Reversal conditions
        both_overheated = (last_rsi>80) & (last_stoch_k>80)
        both_oversold = (last_rsi<20) & (last_stoch_k<20)

        # Forte hausse/forte baisse
        strong_up_condition = pred_diff > (strong_up_threshold*(last_input_closes+1e-6))
        strong_down_condition = pred_diff < -(strong_up_threshold*(last_input_closes+1e-6))

        # Si surachat extrême et marché haussier avant (actual_sign>0),
        # prédire stable ou légère baisse (pred_sign<=0 & moderate_condition ou stable_condition) => reversal +0.05
        reversal_condition_overbought = both_overheated & (actual_sign>0) & ((pred_sign<=0) | stable_condition)
        trend_adj = tf.where(reversal_condition_overbought, trend_adj+0.05, trend_adj)

        # Survente extrême et marché baissier (actual_sign<0),
        # prédire stable ou légère hausse => reversal +0.05
        reversal_condition_oversold = both_oversold & (actual_sign<0) & ((pred_sign>=0) | stable_condition)
        trend_adj = tf.where(reversal_condition_oversold, trend_adj+0.05, trend_adj)

        # Punition si surchauffe mais encore forte hausse
        trend_adj = tf.where(both_overheated & strong_up_condition, trend_adj - 0.1, trend_adj)

        # Punition si survendu extrême mais forte baisse
        trend_adj = tf.where(both_oversold & strong_down_condition, trend_adj - 0.1, trend_adj)

        final_reward += trend_adj

        return final_reward
    return reward_function

def update_model_with_rewards(model, X, y, reward_function, optimizer):
    close_index = 3
    rsi_index = 5
    stoch_k_index = 6

    last_input_closes = X[:, -1, close_index:close_index+1]
    last_rsi = X[:, -1, rsi_index:rsi_index+1]*100.0 
    last_stoch_k = X[:, -1, stoch_k_index:stoch_k_index+1]*100.0

    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        rewards = reward_function(y_pred, tf.constant(y, dtype=tf.float32), 
                                  tf.constant(last_input_closes, dtype=tf.float32),
                                  tf.constant(last_rsi, dtype=tf.float32),
                                  tf.constant(last_stoch_k, dtype=tf.float32))
        mean_reward = tf.reduce_mean(rewards)
        loss = -mean_reward
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss.numpy(), mean_reward.numpy()

def log_rewards(predictions, actual_prices, timestamps, symbol='BTCUSDT', db_url='rewards.db'):
    errors = np.abs(predictions - actual_prices)
    rewards = 1.0 - errors 

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