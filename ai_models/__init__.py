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