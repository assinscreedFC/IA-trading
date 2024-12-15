# database.py

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, BigInteger, Index,
    ForeignKeyConstraint, desc, asc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
import numpy as np
import pandas as pd

Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(BigInteger, index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class Indicator(Base):
    __tablename__ = 'indicators'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(BigInteger, index=True, nullable=False)
    rsi = Column(Float)
    stochastic_k = Column(Float)
    stochastic_d = Column(Float)
    atr = Column(Float)
    
    __table_args__ = (
        Index('idx_indicators_symbol_timestamp', 'symbol', 'timestamp'),
    )

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True, nullable=False)
    timestamp = Column(BigInteger, index=True, nullable=False)
    step = Column(Integer, nullable=False)  # Indique l'étape de la prédiction (1, 2, 3)
    predicted_close = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_predictions_symbol_timestamp_step', 'symbol', 'timestamp', 'step'),
        ForeignKeyConstraint(
            ['symbol', 'timestamp'],
            ['market_data.symbol', 'market_data.timestamp']
        )
    )

def get_engine(db_path):
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    return engine

def create_tables(engine):
    # Crée les tables si elles n'existent pas, sans les recréer
    Base.metadata.create_all(engine)
    logging.info("Tables créées ou déjà existantes.")

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()

def store_market_data_to_db(data, symbol, db_path):
    """
    Insère les données de marché dans la base de données sans duplications.
    """
    try:
        engine = get_engine(db_path)
        session = get_session(engine)
        
        # Récupérer les timestamps déjà présents pour le symbole
        existing_timestamps = set(
            ts for (ts,) in session.query(MarketData.timestamp).filter(MarketData.symbol == symbol).all()
        )
        
        # Filtrer les nouvelles données
        new_data = [
            MarketData(
                symbol=symbol,
                timestamp=entry['timestamp'],
                open=entry['open'],
                high=entry['high'],
                low=entry['low'],
                close=entry['close'],
                volume=entry['volume']
            )
            for entry in data if entry['timestamp'] not in existing_timestamps
        ]
        
        if new_data:
            session.bulk_save_objects(new_data)
            session.commit()
            logging.info(f"{len(new_data)} nouvelles bougies insérées dans la base de données pour {symbol}.")
        else:
            logging.info("Aucune nouvelle bougie à insérer.")
    except Exception as e:
        session.rollback()
        logging.error(f"Erreur lors de l'insertion des données de marché : {e}")
    finally:
        session.close()

def store_indicators_to_db(indicators: pd.DataFrame, symbol: str, db_path: str):
    """
    Insère les indicateurs dans la base de données sans duplications.
    """
    try:
        engine = get_engine(db_path)
        session = get_session(engine)
        
        # Récupérer les timestamps déjà présents pour le symbole
        existing_timestamps = set(
            ts for (ts,) in session.query(Indicator.timestamp).filter(Indicator.symbol == symbol).all()
        )
        
        # Filtrer les nouveaux indicateurs
        indicator_objects = [
            Indicator(
                symbol=symbol,
                timestamp=row['timestamp'],
                rsi=row['rsi'],
                stochastic_k=row['stochastic_k'],
                stochastic_d=row['stochastic_d'],
                atr=row['atr']
            )
            for _, row in indicators.iterrows() if row['timestamp'] not in existing_timestamps
        ]

        if indicator_objects:
            session.bulk_save_objects(indicator_objects)
            session.commit()
            logging.info(f"{len(indicator_objects)} nouveaux indicateurs insérés avec succès dans la base de données pour {symbol}.")
        else:
            logging.info("Aucun nouvel indicateur à insérer.")
    except Exception as e:
        session.rollback()
        logging.error(f"Erreur lors de l'insertion des indicateurs : {e}")
    finally:
        session.close()

def store_predictions_to_db(predictions: np.ndarray, timestamps: np.ndarray, symbol: str, db_path: str, n_out: int):
    """
    Insère les prédictions multi-step dans la base de données sans duplications.

    Args:
        predictions (np.ndarray): Prédictions multi-step (samples, n_out).
        timestamps (np.ndarray): Timestamps correspondants aux prédictions.
        symbol (str): Symbole de trading.
        db_path (str): Chemin vers la base de données.
        n_out (int): Nombre de pas de temps prédits.
    """
    try:
        engine = get_engine(db_path)
        session = get_session(engine)
        
        # Récupérer les timestamps et steps déjà présents pour le symbole
        existing_entries = set(
            (ts, step) for (ts, step) in session.query(Prediction.timestamp, Prediction.step).filter(Prediction.symbol == symbol).all()
        )
        
        prediction_objects = []
        for i, ts in enumerate(timestamps):
            for step in range(1, n_out + 1):
                predicted_close = predictions[i, step - 1]
                predicted_timestamp = ts + step * 60000  # Supposant que chaque bougie est de 1 minute
                if (predicted_timestamp, step) not in existing_entries:
                    prediction_objects.append(
                        Prediction(
                            symbol=symbol,
                            timestamp=int(predicted_timestamp),
                            step=step,
                            predicted_close=float(predicted_close),
                            confidence_level=1.0  # À ajuster selon votre logique
                        )
                    )
        
        if prediction_objects:
            session.bulk_save_objects(prediction_objects)
            session.commit()
            logging.info(f"{len(prediction_objects)} nouvelles prédictions insérées dans la base de données pour {symbol}.")
        else:
            logging.info("Aucune nouvelle prédiction à insérer.")
    except Exception as e:
        session.rollback()
        logging.error(f"Erreur lors de l'insertion des prédictions : {e}")
    finally:
        session.close()

def get_latest_timestamp(symbol: str, db_path: str) -> int:
    """
    Récupère le dernier timestamp enregistré pour un symbole donné dans la base de données.
    """
    try:
        engine = get_engine(db_path)
        session = get_session(engine)
        latest_ts = session.query(MarketData.timestamp).filter(MarketData.symbol == symbol).order_by(MarketData.timestamp.desc()).first()
        if latest_ts:
            logging.info(f"Dernier timestamp pour {symbol} : {latest_ts[0]}")
            return latest_ts[0]
        else:
            logging.info(f"Aucun timestamp trouvé pour {symbol}, début à partir du début des données.")
            return 0
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du dernier timestamp : {e}")
        return 0
    finally:
        session.close()

def get_earliest_timestamp(symbol: str, db_path: str) -> int:
    """
    Récupère le premier timestamp enregistré pour un symbole donné dans la base de données.
    """
    try:
        engine = get_engine(db_path)
        session = get_session(engine)
        earliest_ts = session.query(MarketData.timestamp).filter(MarketData.symbol == symbol).order_by(MarketData.timestamp.asc()).first()
        if earliest_ts:
            logging.info(f"Premier timestamp pour {symbol} : {earliest_ts[0]}")
            return earliest_ts[0]
        else:
            logging.info(f"Aucun timestamp trouvé pour {symbol}, début à partir du début des données.")
            return 0
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du premier timestamp : {e}")
        return 0
    finally:
        session.close()

def delete_old_candles(session, symbol: str, candles_to_delete: int):
    """
    Supprime les bougies les plus anciennes pour un symbole donné.
    
    Args:
        session: Session SQLAlchemy.
        symbol (str): Symbole de trading.
        candles_to_delete (int): Nombre de bougies à supprimer.
    """
    try:
        old_candles = session.query(MarketData).filter(MarketData.symbol == symbol).order_by(MarketData.timestamp.asc()).limit(candles_to_delete).all()
        for candle in old_candles:
            session.delete(candle)
    except Exception as e:
        logging.error(f"Erreur lors de la suppression des bougies anciennes : {e}")