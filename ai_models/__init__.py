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
    available_models = {}
    
    # Próba importu i dodania SentimentAnalyzer z obsługą błędów
    try:
        from .sentiment_ai import SentimentAnalyzer
        available_models["sentiment_analyzer"] = SentimentAnalyzer
        logger.info("Załadowano model: SentimentAnalyzer")
    except ImportError as e:
        logger.warning(f"Nie można załadować modelu SentimentAnalyzer: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas ładowania SentimentAnalyzer: {e}")
    
    # Próba importu i dodania AnomalyDetector z obsługą błędów
    try:
        from .anomaly_detection import AnomalyDetector
        available_models["anomaly_detector"] = AnomalyDetector
        logger.info("Załadowano model: AnomalyDetector")
    except ImportError as e:
        logger.warning(f"Nie można załadować modelu AnomalyDetector: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas ładowania AnomalyDetector: {e}")
    
    # Próba importu i dodania ModelRecognizer z obsługą błędów
    try:
        from .model_recognition import ModelRecognizer
        available_models["model_recognizer"] = ModelRecognizer
        logger.info("Załadowano model: ModelRecognizer")
    except ImportError as e:
        logger.warning(f"Nie można załadować modelu ModelRecognizer: {e}")
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas ładowania ModelRecognizer: {e}")
    
    logger.info(f"Łącznie załadowano {len(available_models)} modeli")
    return available_models

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