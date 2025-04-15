
"""
model_utils.py
-------------
Moduł z pomocniczymi funkcjami do zarządzania modelami AI.
"""

import os
import json
import pickle
import logging
import joblib
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_model_dirs():
    """
    Tworzy katalogi niezbędne do przechowywania modeli.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_models/checkpoints", exist_ok=True)

def list_available_models() -> List[Dict[str, Any]]:
    """
    Zwraca listę dostępnych modeli z ich metadanymi.
    
    Returns:
        List[Dict]: Lista modeli z ich metadanymi
    """
    models = []
    
    # Sprawdź czy katalog models istnieje
    if not os.path.exists("models"):
        return models
    
    # Przeszukaj katalog models
    for filename in os.listdir("models"):
        if filename.endswith("_model.pkl") or filename.endswith(".pkl"):
            model_name = filename.replace("_model.pkl", "").replace(".pkl", "")
            model_path = os.path.join("models", filename)
            metadata_path = os.path.join("models", f"{model_name}_metadata.json")
            
            # Podstawowe informacje o modelu
            model_info = {
                "name": model_name,
                "path": model_path,
                "size": os.path.getsize(model_path),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Dodaj metadane, jeśli istnieją
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    model_info["metadata"] = metadata
                except json.JSONDecodeError:
                    logger.warning(f"Błąd parsowania metadanych dla {model_name}")
            
            models.append(model_info)
    
    return models

def save_model(model, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Zapisuje model do pliku wraz z metadanymi.
    
    Args:
        model: Model do zapisania
        model_name (str): Nazwa modelu
        metadata (Dict): Metadane modelu
        
    Returns:
        bool: True jeśli zapis się powiódł, False w przeciwnym razie
    """
    try:
        # Upewnij się, że katalog istnieje
        ensure_model_dirs()
        
        # Ścieżki do plików
        model_path = os.path.join("models", f"{model_name.lower()}_model.pkl")
        metadata_path = os.path.join("models", f"{model_name.lower()}_metadata.json")
        
        # Zapisz model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Przygotuj metadane
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "name": model_name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": type(model).__name__,
            "module": type(model).__module__
        })
        
        # Zapisz metadane
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model {model_name} zapisany do {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania modelu {model_name}: {e}")
        return False

def load_model(model_name: str) -> Tuple[Any, Dict[str, Any], bool]:
    """
    Wczytuje model z pliku.
    
    Args:
        model_name (str): Nazwa modelu
        
    Returns:
        Tuple: (model, metadata, success)
    """
    try:
        # Ścieżki do plików
        model_path = os.path.join("models", f"{model_name.lower()}_model.pkl")
        metadata_path = os.path.join("models", f"{model_name.lower()}_metadata.json")
        
        # Sprawdź czy plik istnieje
        if not os.path.exists(model_path):
            logger.warning(f"Plik modelu {model_path} nie istnieje")
            return None, {}, False
        
        # Wczytaj model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Wczytaj metadane, jeśli istnieją
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Błąd parsowania metadanych dla {model_name}")
        
        logger.info(f"Model {model_name} wczytany z {model_path}")
        return model, metadata, True
    
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania modelu {model_name}: {e}")
        return None, {}, False

def create_model_checkpoint(model, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Tworzy checkpoint modelu.
    
    Args:
        model: Model do zapisania
        model_name (str): Nazwa modelu
        metadata (Dict): Metadane modelu
        
    Returns:
        str: Ścieżka do zapisanego checkpointu lub pusty string w przypadku błędu
    """
    try:
        # Upewnij się, że katalog istnieje
        ensure_model_dirs()
        
        # Utwórz nazwę pliku z timestampem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join("saved_models/checkpoints", f"{model_name}_{timestamp}.pkl")
        metadata_path = os.path.join("saved_models/checkpoints", f"{model_name}_{timestamp}_metadata.json")
        
        # Zapisz model
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Przygotuj metadane
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "name": model_name,
            "checkpoint_time": timestamp,
            "model_type": type(model).__name__,
            "module": type(model).__module__
        })
        
        # Zapisz metadane
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint modelu {model_name} zapisany do {checkpoint_path}")
        return checkpoint_path
    
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia checkpointu modelu {model_name}: {e}")
        return ""

def delete_model(model_name: str) -> bool:
    """
    Usuwa model i jego metadane.
    
    Args:
        model_name (str): Nazwa modelu
        
    Returns:
        bool: True jeśli usunięcie się powiodło, False w przeciwnym razie
    """
    try:
        # Ścieżki do plików
        model_path = os.path.join("models", f"{model_name.lower()}_model.pkl")
        metadata_path = os.path.join("models", f"{model_name.lower()}_metadata.json")
        
        # Usuń pliki jeśli istnieją
        if os.path.exists(model_path):
            os.remove(model_path)
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        logger.info(f"Model {model_name} usunięty")
        return True
    
    except Exception as e:
        logger.error(f"Błąd podczas usuwania modelu {model_name}: {e}")
        return False

def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Pobiera metadane modelu.
    
    Args:
        model_name (str): Nazwa modelu
        
    Returns:
        Dict: Metadane modelu lub pusty słownik w przypadku błędu
    """
    try:
        # Ścieżka do pliku metadanych
        metadata_path = os.path.join("models", f"{model_name.lower()}_metadata.json")
        
        # Sprawdź czy plik istnieje
        if not os.path.exists(metadata_path):
            logger.warning(f"Plik metadanych {metadata_path} nie istnieje")
            return {}
        
        # Wczytaj metadane
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    except Exception as e:
        logger.error(f"Błąd podczas pobierania metadanych modelu {model_name}: {e}")
        return {}
