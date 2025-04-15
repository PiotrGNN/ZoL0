
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
        Dict[str, Any]: Słownik z nazwami modeli i ich klasami
    """
    available_models = {}
    
    try:
        # Ścieżka do bieżącego katalogu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Sprawdź wszystkie pliki .py w katalogu
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') and filename != '__init__.py' and not filename.startswith('_'):
                module_name = filename[:-3]  # Usuń '.py'
                
                try:
                    # Importuj moduł
                    module = importlib.import_module(f'ai_models.{module_name}')
                    
                    # Znajdź klasy w module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if isinstance(attr, type) and attr.__module__ == f'ai_models.{module_name}':
                            # Dodaj klasę do słownika
                            available_models[attr_name.lower()] = attr
                except ImportError as e:
                    logger.warning(f"Nie można zaimportować modułu {module_name}: {e}")
    except Exception as e:
        logger.error(f"Błąd podczas wyszukiwania modeli: {e}")
    
    logger.info(f"Znaleziono {len(available_models)} modeli")
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

# Eksportuj nazwy klas do przestrzeni nazw pakietu
try:
    from .sentiment_ai import SentimentAnalyzer
    from .anomaly_detection import AnomalyDetector
    from .model_recognition import ModelRecognizer
except ImportError as e:
    # Wydrukuj ostrzeżenie, ale nie przerywaj importu pakietu
    logger.warning(f"Ostrzeżenie podczas importowania modeli AI: {e}")

# Inicjalizacja - wypisanie znalezionych modeli
available_models = get_available_models()
logger.info(f"ai_models: Znaleziono {len(available_models)} modeli AI")
for name, model_class in available_models.items():
    logger.debug(f"Dostępny model: {name} ({model_class.__name__})")
