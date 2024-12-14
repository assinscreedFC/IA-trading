from sqlalchemy import (
    create_engine, Column, Integer, String, Float, BigInteger, Index,
    ForeignKeyConstraint
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
    # Supprimer les tables existantes pour enlever les anciennes contraintes
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logging.info("Tables recréées ou déjà existantes.")

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()

def store_market_data_to_db(data, symbol, db_path):
    try:
        engine = get_engine(db_path)
        create_tables(engine)
        session = get_session(engine)
        
        market_data_objects = [
            MarketData(
                symbol=symbol,
                timestamp=entry['timestamp'],
                open=entry['open'],
                high=entry['high'],
                low=entry['low'],
                close=entry['close'],
                volume=entry['volume']
            )
            for entry in data
        ]
        
        session.bulk_save_objects(market_data_objects)
        session.commit()
        logging.info(f"{len(market_data_objects)} entrées insérées dans la base de données pour {symbol}.")
    except Exception as e:
        session.rollback()
        logging.error(f"Erreur lors de l'insertion des données dans la base de données : {e}")
    finally:
        session.close()

def store_indicators_to_db(indicators: pd.DataFrame, symbol: str, db_path: str):
    try:
        engine = get_engine(db_path)
        create_tables(engine)
        session = get_session(engine)
        
        indicator_objects = [
            Indicator(
                symbol=symbol,
                timestamp=row['timestamp'],
                rsi=row['rsi'],
                stochastic_k=row['stochastic_k'],
                stochastic_d=row['stochastic_d'],
                atr=row['atr']
            )
            for _, row in indicators.iterrows()
        ]

        session.bulk_save_objects(indicator_objects)
        session.commit()
        logging.info(f"✅ {len(indicator_objects)} indicateurs insérés avec succès dans la base de données pour {symbol}.")
    except Exception as e:
        session.rollback()
        logging.error(f"❌ Erreur lors de l'insertion des indicateurs : {e}")
    finally:
        session.close()

def store_predictions_to_db(predictions: pd.DataFrame, symbol: str, db_path: str):
    try:
        engine = get_engine(db_path)
        create_tables(engine)
        session = get_session(engine)
        
        required_columns = ['timestamp', 'predicted_close', 'confidence_level']
        if not all(col in predictions.columns for col in required_columns):
            missing = [col for col in required_columns if col not in predictions.columns]
            raise ValueError(f"Les colonnes manquantes pour les prédictions : {missing}")

        predictions['timestamp'] = predictions['timestamp'].astype(int)

        prediction_objects = [
            Prediction(
                symbol=symbol,
                timestamp=row['timestamp'],
                predicted_close=row['predicted_close'],
                confidence_level=row['confidence_level']
            )
            for _, row in predictions.iterrows()
        ]
        
        session.bulk_save_objects(prediction_objects)
        session.commit()
        logging.info(f"✅ {len(prediction_objects)} prédictions insérées avec succès dans la base de données pour {symbol}.")
    except Exception as e:
        session.rollback()
        logging.error(f"❌ Erreur lors de l'insertion des prédictions : {e}")
    finally:
        session.close()
