"""
Pakiet ai_models - zawiera modele AI używane w systemie tradingowym.
"""

import os
import importlib
import logging
from typing import Dict, List, Any, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_models")

def get_available_models():
    """
    Zwraca słownik dostępnych modeli AI.

    Returns:
        dict: Słownik z nazwami modeli i ich klasami
    """
    from .sentiment_ai import SentimentAnalyzer
    from .anomaly_detection import AnomalyDetector
    from .model_recognition import ModelRecognizer
    from .model_training import ModelTrainer
    from .real_exchange_env import RealExchangeEnv
    
    try:
        # Importuj fabrykę modeli
        from .model_factory import get_model_instance
        
        # Dodaj instancje wymagające parametrów
        model_trainer = get_model_instance("ModelTrainer")
        real_exchange_env = get_model_instance("RealExchangeEnv")
        
        return {
            "sentiment_analyzer": SentimentAnalyzer,
            "anomaly_detector": AnomalyDetector,
            "model_recognizer": ModelRecognizer,
            "model_trainer": ModelTrainer,
            "real_exchange_env": RealExchangeEnv
        }
    except Exception as e:
        logger.warning(f"Błąd podczas ładowania niektórych modeli: {e}")
        return {
            "sentiment_analyzer": SentimentAnalyzer,
            "anomaly_detector": AnomalyDetector,
            "model_recognizer": ModelRecognizer
        }

# Eksportuj nazwy klas do przestrzeni nazw pakietu
try:
    from .sentiment_ai import SentimentAnalyzer
    from .anomaly_detection import AnomalyDetector
    from .model_recognition import ModelRecognizer
except ImportError as e:
    # Wydrukuj ostrzeżenie, ale nie przerywaj importu pakietu
    import logging
    logging.warning(f"Ostrzeżenie podczas importowania modeli AI: {e}")


def list_model_files() -> List[str]:
    """
    Zwraca listę plików modeli w folderze ai_models.

    Returns:
        List[str]: Lista nazw plików .py w folderze
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = []

    for file in os.listdir(current_dir):
        if file.endswith('.py') and file != '__init__.py' and not file.startswith('_'):
            model_files.append(file)

    return model_files

# Inicjalizacja - wypisanie znalezionych modeli
available_models = get_available_models()
logger.info(f"ai_models: Znaleziono {len(available_models)} modeli AI")
for name, model_class in available_models.items():
    logger.info(f"  - {name}: {model_class.__name__}")