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
    step = Column(Integer, nullable=False)  # Indique l'étape de la prédiction (1, 2, 3)
    predicted_close = Column(Float, nullable=False)
    actual_close = Column(Float, nullable=False)
    reward = Column(Float, nullable=False)

def define_reward_function_multi_step(n_out: int):
    """
    Fonction de récompense pour les prédictions multi-step.
    Retourne des récompenses pour chaque étape de prédiction.

    Args:
        n_out (int): Nombre de pas de temps prédits.

    Returns:
        function: Fonction de récompense.
    """
    def reward_function(predictions, actuals, last_input_closes, last_rsi, last_stoch_k, last_ma_court, last_ma_long):
        """
        Calcul de la récompense pour les prédictions multi-step.

        Args:
            predictions (tf.Tensor): Prédictions du modèle (batch, n_out).
            actuals (tf.Tensor): Valeurs réelles (batch, n_out).
            last_input_closes (tf.Tensor): Dernières valeurs de clôture d'entrée.
            last_rsi (tf.Tensor): Dernières valeurs RSI.
            last_stoch_k (tf.Tensor): Dernières valeurs Stochastic K.
            last_ma_court (tf.Tensor): Dernières moyennes mobiles courtes.
            last_ma_long (tf.Tensor): Dernières moyennes mobiles longues.

        Returns:
            tf.Tensor: Récompenses pour chaque prédiction.
        """
        rewards = []
        for step in range(n_out):
            pred = predictions[:, step]
            actual = actuals[:, step]
            error = tf.abs(pred - actual)
            
            # Paramètres pour le calcul de la récompense
            sigma = 0.05  # Contrôle la sensibilité de la récompense par rapport à l'erreur
            max_reward = 1.0
            min_reward = -0.5

            # Calcul de la récompense basée sur une fonction exponentielle inversée
            reward = tf.exp(- (error ** 2) / (2 * sigma ** 2))  # Récompense entre 0 et 1

            # Ajustement de la récompense basée sur la direction
            actual_diff = actual - last_input_closes[:, 0]
            pred_diff = pred - last_input_closes[:, 0]
            actual_sign = tf.sign(actual_diff)
            pred_sign = tf.sign(pred_diff)

            direction_correct = tf.cast(tf.equal(actual_sign, pred_sign), tf.float32)
            reward += 0.2 * direction_correct  # Récompense additionnelle pour la bonne direction

            # Normalisation de la récompense dans une plage spécifique
            reward = tf.clip_by_value(reward, min_reward, max_reward)
            rewards.append(reward)
        
        # Moyenne des récompenses sur les étapes pour une récompense globale par échantillon
        final_reward = tf.reduce_mean(tf.stack(rewards), axis=0)
        return final_reward
    return reward_function

def update_model_with_rewards_multi_step(model, X, y, reward_function, optimizer, n_out: int):
    """
    Met à jour le modèle en fonction des récompenses multi-step.

    Args:
        model: Modèle TensorFlow.
        X: Entrées.
        y: Cibles.
        reward_function: Fonction de récompense.
        optimizer: Optimiseur TensorFlow.
        n_out (int): Nombre de pas de temps prédits.

    Returns:
        Tuple contenant la perte et la récompense moyenne.
    """
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

def log_rewards_multi_step(predictions, actual_prices, timestamps, symbol='BTCUSDT', db_url='rewards.db', n_out: int = 3):
    """
    Log les récompenses pour les prédictions multi-step dans la base de données.

    Args:
        predictions (np.ndarray): Prédictions multi-step (samples, n_out).
        actual_prices (np.ndarray): Valeurs réelles multi-step (samples, n_out).
        timestamps (np.ndarray): Timestamps correspondants aux prédictions.
        symbol (str, optional): Symbole de trading. Par défaut à 'BTCUSDT'.
        db_url (str, optional): URL de la base de données. Par défaut à 'rewards.db'.
        n_out (int, optional): Nombre de pas de temps prédits. Par défaut à 3.
    """
    try:
        # Calcul des récompenses basées sur l'erreur
        rewards = 1.0 - np.abs(predictions - actual_prices)  # Simplification pour l'exemple

        engine = create_engine(f'sqlite:///{db_url}', echo=False)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        reward_objects = []
        for i, ts in enumerate(timestamps):
            for step in range(1, n_out + 1):
                reward_objects.append(
                    Reward(
                        symbol=symbol,
                        timestamp=int(ts) + step * 60000,  # Supposant que chaque bougie est de 1 minute
                        step=step,
                        predicted_close=float(predictions[i, step - 1]),
                        actual_close=float(actual_prices[i, step - 1]),
                        reward=float(rewards[i, step - 1])
                    )
                )
        
        if reward_objects:
            session.bulk_save_objects(reward_objects)
            session.commit()
            logging.info(f"{len(reward_objects)} récompenses multi-step insérées dans la base de données {db_url}.")
        else:
            logging.info("Aucune récompense multi-step à insérer.")
    except Exception as e:
        session.rollback()
        logging.error(f"Erreur lors de l'insertion des récompenses multi-step : {e}")
    finally:
        session.close()

def analyze_rewards(db_url='rewards.db'):
    """
    Analyse les récompenses stockées dans la base de données.

    Args:
        db_url (str, optional): URL de la base de données. Par défaut à 'rewards.db'.

    Returns:
        dict: Statistiques des récompenses.
    """
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
            'step': r.step,
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