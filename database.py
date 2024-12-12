from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
import pandas as pd

# Définir la base pour les modèles
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
    
    # Définir un index composite sur symbol et timestamp
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
    
    # Définir un index composite sur symbol et timestamp
    __table_args__ = (
        Index('idx_indicators_symbol_timestamp', 'symbol', 'timestamp'),
    )

def get_engine(db_path):
    """
    Crée et retourne un moteur SQLAlchemy connecté à la base de données spécifiée.
    
    Args:
        db_path (str): Chemin vers le fichier de base de données SQLite.
    
    Returns:
        sqlalchemy.engine.Engine: Moteur SQLAlchemy.
    """
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    return engine

def create_tables(engine):
    """
    Crée les tables définies par les modèles si elles n'existent pas déjà.
    
    Args:
        engine (sqlalchemy.engine.Engine): Moteur SQLAlchemy.
    """
    Base.metadata.create_all(engine)
    logging.info("Tables créées ou déjà existantes.")

# Créer une session factory
def get_session(engine):
    """
    Crée et retourne une session SQLAlchemy.
    
    Args:
        engine (sqlalchemy.engine.Engine): Moteur SQLAlchemy.
    
    Returns:
        sqlalchemy.orm.session.Session: Session SQLAlchemy.
    """
    Session = sessionmaker(bind=engine)
    return Session()

def store_market_data_to_db(data, symbol, db_path):
    """
    Insère les données OHLCV dans la table `market_data`.
    
    Args:
        data (list of dict): Liste des données OHLCV.
        symbol (str): La paire de trading (ex. 'BTCUSDT').
        db_path (str): Chemin vers le fichier de base de données SQLite.
    """
    try:
        engine = get_engine(db_path)
        create_tables(engine)
        session = get_session(engine)
        
        # Préparer les objets MarketData
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
        
        # Ajouter les données à la session
        session.bulk_save_objects(market_data_objects)
        session.commit()
        logging.info(f"{len(market_data_objects)} entrées insérées dans la base de données pour {symbol}.")
    except Exception as e:
        logging.error(f"Erreur lors de l'insertion des données dans la base de données : {e}")
    finally:
        session.close()

def store_indicators_to_db(indicators: pd.DataFrame, symbol: str, db_path: str):
    """
    Insère les valeurs RSI, Stochastic RSI (%K et %D), et ATR dans la table SQL 'indicators'.
    
    Args:
        indicators (pd.DataFrame): DataFrame contenant les colonnes 'timestamp', 'rsi', 'stochastic_k', 'stochastic_d', et 'atr'.
        symbol (str): Symbole de la paire de trading (ex. 'BTCUSDT').
        db_path (str): Chemin vers le fichier de base de données SQLite.
    """
    try:
        engine = get_engine(db_path)
        create_tables(engine)
        session = get_session(engine)
        
        # Préparer les objets Indicator
        indicator_objects = [
            Indicator(
                symbol=symbol,
                timestamp=row['timestamp'],
                rsi=row['rsi'],
                stochastic_k=row['stochastic_k'],
                stochastic_d=row['stochastic_d'],
                atr=row['atr']
            )
            for index, row in indicators.iterrows()
        ]
        
        # Ajouter les objets à la session
        session.bulk_save_objects(indicator_objects)
        
        # Commit la transaction
        session.commit()
        logging.info(f"✅ {len(indicator_objects)} indicateurs insérés avec succès dans la base de données pour {symbol}.")
    except Exception as e:
        session.rollback()
        logging.error(f"❌ Erreur lors de l'insertion des indicateurs : {e}")
    finally:
        session.close()