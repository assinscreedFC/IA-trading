from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import MarketData

def check_database(db_path):
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Récupérer les premières 5 entrées
        records = session.query(MarketData).all()
        for record in records:
            print(f"ID: {record.id}, Symbol: {record.symbol}, Timestamp: {record.timestamp}, "
                  f"Open: {record.open}, High: {record.high}, Low: {record.low}, "
                  f"Close: {record.close}, Volume: {record.volume}")
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
    finally:
        session.close()

if __name__ == '__main__':
    db_path = 'trading_bot.db'  # Assurez-vous que ce chemin est correct
    check_database(db_path)