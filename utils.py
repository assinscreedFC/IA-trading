# utils.py
import logging
import yaml
from dotenv import load_dotenv
import os
from binance.client import Client

def setup_logging(log_file='trading_bot.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path='config.yaml'):
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path (str): Chemin vers le fichier de configuration YAML.
    
    Returns:
        dict: Configuration chargée.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration chargée depuis {config_path}.")
            return config
    except FileNotFoundError:
        logging.error(f"Fichier de configuration {config_path} non trouvé.")
        return {}
    except yaml.YAMLError as exc:
        logging.error(f"Erreur de parsing YAML: {exc}")
        return {}

def setup_binance_api(config):
    """
    Configure et retourne une instance de l'API Binance en utilisant python-binance.
    
    Args:
        config (dict): Configuration chargée depuis le fichier config.yaml.
    
    Returns:
        binance.client.Client: Instance configurée de l'API Binance.
    """
    api_key = config.get('api_key', '')
    api_secret = config.get('api_secret', '')

    if api_key and api_secret:
        logging.info("Clés API fournies. Initialisation de l'API Binance avec authentification.")
        client = Client(api_key, api_secret)
        try:
            client.get_account()
            logging.info("Connexion à Binance réussie.")
        except Exception as e:
            logging.error(f"Erreur lors de la connexion à Binance : {e}")
            raise
    else:
        logging.info("Clés API non fournies. Initialisation de l'API Binance en mode non authentifié.")
        client = Client()

    return client
