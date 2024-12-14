from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import MarketData, Indicator, Prediction
import logging

def check_market_data(session, limit=1000000):
    """
    Récupère et affiche les premières entrées de la table MarketData.

    Args:
        session (sqlalchemy.orm.session.Session): Session SQLAlchemy.
        limit (int): Nombre d'entrées à récupérer.
    """
    try:
        records = session.query(MarketData).limit(limit).all()
        print("\n--- MarketData ---")
        for record in records:
            print(f"ID: {record.id}, Symbol: {record.symbol}, Timestamp: {record.timestamp}, "
                  f"Open: {record.open}, High: {record.high}, Low: {record.low}, "
                  f"Close: {record.close}, Volume: {record.volume}")
    except Exception as e:
        print(f"Erreur lors de la récupération des données de MarketData : {e}")

def check_indicators(session, limit=5):
    """
    Récupère et affiche les premières entrées de la table Indicator.

    Args:
        session (sqlalchemy.orm.session.Session): Session SQLAlchemy.
        limit (int): Nombre d'entrées à récupérer.
    """
    try:
        records = session.query(Indicator).limit(limit).all()
        print("\n--- Indicators ---")
        for record in records:
            print(f"ID: {record.id}, Symbol: {record.symbol}, Timestamp: {record.timestamp}, "
                  f"RSI: {record.rsi}, Stochastic K: {record.stochastic_k}, "
                  f"Stochastic D: {record.stochastic_d}, ATR: {record.atr}")
    except Exception as e:
        print(f"Erreur lors de la récupération des indicateurs : {e}")

def check_predictions(session, limit=5):
    """
    Récupère et affiche les premières entrées de la table Prediction.

    Args:
        session (sqlalchemy.orm.session.Session): Session SQLAlchemy.
        limit (int): Nombre d'entrées à récupérer.
    """
    try:
        records = session.query(Prediction).limit(limit).all()
        print("\n--- Predictions ---")
        for record in records:
            print(f"ID: {record.id}, Symbol: {record.symbol}, Timestamp: {record.timestamp}, "
                  f"Predicted Close: {record.predicted_close}, Confidence Level: {record.confidence_level}")
    except Exception as e:
        print(f"Erreur lors de la récupération des prédictions : {e}")

def check_database(db_path):
    """
    Vérifie et affiche les premières entrées des tables MarketData, Indicator et Prediction.

    Args:
        db_path (str): Chemin vers le fichier de base de données SQLite.
    """
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Vérifier MarketData
        check_market_data(session, limit=1000000)
        
        # Vérifier Indicators
        check_indicators(session, limit=5)
        
        # Vérifier Predictions
        check_predictions(session, limit=5)
        
    except Exception as e:
        print(f"Erreur générale lors de la vérification de la base de données : {e}")
    finally:
        session.close()

if __name__ == '__main__':
    db_path = 'trading_bot.db'  # Assurez-vous que ce chemin est correct
    check_database(db_path)