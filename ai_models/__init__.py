
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
from typing import Dict, List, Any

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ai_models")

# Eksportujemy konkretne modele
from .anomaly_detection import AnomalyDetector
from .sentiment_ai import SentimentAnalyzer
from .model_recognition import ModelRecognizer
from .feature_engineering import add_rsi, add_macd, feature_pipeline

# Lista dostępnych modeli
__all__ = [
    'AnomalyDetector',
    'SentimentAnalyzer',
    'ModelRecognizer',
    'add_rsi',
    'add_macd', 
    'feature_pipeline'
]

# Słownik dostępnych klas modeli
available_models = {
    'anomaly_detector': AnomalyDetector,
    'sentiment_analyzer': SentimentAnalyzer,
    'model_recognizer': ModelRecognizer
}

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
