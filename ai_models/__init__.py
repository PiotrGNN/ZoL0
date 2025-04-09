"""
ai_models/__init__.py
---------------------
Inicjalizacja pakietu modeli AI.

Ten moduł importuje i eksportuje wszystkie dostępne modele AI
z folderu ai_models, ułatwiając ich wykrywanie i ładowanie.
"""

import os
import importlib
import logging
import sys
from typing import Dict, List, Any, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_models")

# Słownik dostępnych klas modeli
available_models = {}

# Próba importu konkretnych modeli
try:
    from .anomaly_detection import AnomalyDetector
    available_models['anomaly_detector'] = AnomalyDetector
    logger.info("Zaimportowano AnomalyDetector")
except ImportError as e:
    logger.warning(f"Nie można zaimportować AnomalyDetector: {e}")

try:
    from .sentiment_ai import SentimentAnalyzer 
    available_models['sentiment_analyzer'] = SentimentAnalyzer
    logger.info("Zaimportowano SentimentAnalyzer")
except ImportError as e:
    logger.warning(f"Nie można zaimportować SentimentAnalyzer: {e}")

try:
    from .model_recognition import ModelRecognizer
    available_models['model_recognizer'] = ModelRecognizer
    logger.info("Zaimportowano ModelRecognizer")
except ImportError as e:
    logger.warning(f"Nie można zaimportować ModelRecognizer: {e}")

def get_available_models() -> Dict[str, Any]:
    """
    Zwraca słownik dostępnych modeli AI.

    Returns:
        Dict[str, Any]: Słownik nazw modeli i ich klas
    """
    return available_models

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
logger.info(f"ai_models: Znaleziono {len(available_models)} modeli AI")
for name, model_class in available_models.items():
    logger.info(f"  - {name}: {model_class.__name__}")