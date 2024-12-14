# database.py
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, BigInteger, Index,
    ForeignKeyConstraint, desc, asc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
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
    predicted_close = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_predictions_symbol_timestamp', 'symbol', 'timestamp'),
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

def store_predictions_to_db(predictions: pd.DataFrame, symbol: str, db_path: str):
    """
    Insère les prédictions dans la base de données sans duplications.
    """
    try:
        engine = get_engine(db_path)
        session = get_session(engine)
        
        required_columns = ['timestamp', 'predicted_close', 'confidence_level']
        if not all(col in predictions.columns for col in required_columns):
            missing = [col for col in required_columns if col not in predictions.columns]
            raise ValueError(f"Les colonnes manquantes pour les prédictions : {missing}")

        predictions['timestamp'] = predictions['timestamp'].astype(int)
        
        # Récupérer les timestamps déjà présents pour le symbole
        existing_timestamps = set(
            ts for (ts,) in session.query(Prediction.timestamp).filter(Prediction.symbol == symbol).all()
        )
        
        # Filtrer les nouvelles prédictions
        prediction_objects = [
            Prediction(
                symbol=symbol,
                timestamp=row['timestamp'],
                predicted_close=row['predicted_close'],
                confidence_level=row['confidence_level']
            )
            for _, row in predictions.iterrows() if row['timestamp'] not in existing_timestamps
        ]
        
        if prediction_objects:
            session.bulk_save_objects(prediction_objects)
            session.commit()
            logging.info(f"{len(prediction_objects)} nouvelles prédictions insérées avec succès dans la base de données pour {symbol}.")
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