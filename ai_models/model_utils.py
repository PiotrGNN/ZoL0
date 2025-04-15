
"""
model_utils.py - Narzędzia do zarządzania modelami AI
"""

import os
import sys
import json
import shutil
import logging
import joblib
import pickle
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Eksport dostępnych funkcji i klas
__all__ = [
    "ensure_model_dirs", 
    "save_model", 
    "load_model", 
    "create_model_checkpoint",
    "list_available_models", 
    "is_model_newer_than_data", 
    "delete_model",
    "copy_model_to_saved", 
    "load_model_metadata"
]

def ensure_model_dirs():
    """Tworzy wymagane katalogi dla modeli, jeśli nie istnieją."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_models/checkpoints", exist_ok=True)

def save_model(model, name: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
    """
    Zapisuje model do pliku wraz z metadanymi.
    
    Args:
        model: Model do zapisania
        name: Nazwa modelu (bez rozszerzenia)
        metadata: Opcjonalne metadane modelu
    
    Returns:
        Tuple[bool, str]: Status sukcesu i ścieżka do zapisanego pliku
    """
    ensure_model_dirs()
    
    # Jeśli nie podano metadanych, utwórz pusty słownik
    if metadata is None:
        metadata = {}
    
    # Dodaj informacje o czasie zapisu
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['model_type'] = type(model).__name__
    metadata['model_module'] = type(model).__module__
    
    # Ścieżki plików
    model_path = os.path.join("models", f"{name}_model.pkl")
    metadata_path = os.path.join("models", f"{name}_metadata.json")
    
    try:
        # Zapisz model
        joblib.dump(model, model_path)
        logger.info(f"Model {name} został zapisany do {model_path}")
        
        # Zapisz metadane
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadane modelu {name} zostały zapisane do {metadata_path}")
        
        return True, model_path
    
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania modelu {name}: {e}")
        return False, str(e)

def load_model(name: str) -> Tuple[Any, Dict[str, Any], bool]:
    """
    Ładuje model z pliku wraz z metadanymi.
    
    Args:
        name: Nazwa modelu (bez rozszerzenia)
    
    Returns:
        Tuple[Any, Dict[str, Any], bool]: Model, metadane i status sukcesu
    """
    # Ścieżki plików
    model_path = os.path.join("models", f"{name}_model.pkl")
    metadata_path = os.path.join("models", f"{name}_metadata.json")
    
    # Domyślne metadane
    metadata = {}
    
    try:
        # Sprawdź, czy plik modelu istnieje
        if not os.path.exists(model_path):
            logger.warning(f"Model {name} nie istnieje: {model_path}")
            return None, metadata, False
        
        # Ładuj model
        model = joblib.load(model_path)
        logger.info(f"Model {name} został załadowany z {model_path}")
        
        # Ładuj metadane, jeśli istnieją
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadane modelu {name} zostały załadowane z {metadata_path}")
        else:
            logger.warning(f"Brak pliku metadanych dla modelu {name}")
        
        return model, metadata, True
    
    except Exception as e:
        logger.error(f"Błąd podczas ładowania modelu {name}: {e}")
        return None, metadata, False

def create_model_checkpoint(model, name: str, metadata: Dict[str, Any] = None) -> Tuple[bool, str]:
    """
    Tworzy checkpoint modelu z timestampem.
    
    Args:
        model: Model do zapisania
        name: Nazwa modelu (bez rozszerzenia)
        metadata: Opcjonalne metadane modelu
    
    Returns:
        Tuple[bool, str]: Status sukcesu i ścieżka do zapisanego pliku
    """
    ensure_model_dirs()
    
    # Jeśli nie podano metadanych, utwórz pusty słownik
    if metadata is None:
        metadata = {}
    
    # Dodaj informacje o czasie zapisu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['checkpoint_timestamp'] = timestamp
    metadata['model_type'] = type(model).__name__
    metadata['model_module'] = type(model).__module__
    
    # Ścieżki plików
    checkpoint_path = os.path.join("saved_models", "checkpoints", f"{name}_{timestamp}.pkl")
    metadata_path = os.path.join("saved_models", "checkpoints", f"{name}_{timestamp}_metadata.json")
    
    try:
        # Zapisz model
        joblib.dump(model, checkpoint_path)
        logger.info(f"Checkpoint modelu {name} został zapisany do {checkpoint_path}")
        
        # Zapisz metadane
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadane checkpointu modelu {name} zostały zapisane do {metadata_path}")
        
        return True, checkpoint_path
    
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia checkpointu modelu {name}: {e}")
        return False, str(e)

def list_available_models() -> List[Dict[str, Any]]:
    """
    Zwraca listę dostępnych modeli wraz z informacjami.
    
    Returns:
        List[Dict[str, Any]]: Lista informacji o modelach
    """
    ensure_model_dirs()
    
    models = []
    models_dir = "models"
    
    try:
        # Znajdź wszystkie pliki .pkl
        model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model_name = model_file.replace('_model.pkl', '')
            
            # Informacje o pliku
            file_info = {
                'name': model_name,
                'path': model_path,
                'size': os.path.getsize(model_path),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Sprawdź, czy istnieje plik metadanych
            metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    file_info['metadata'] = metadata
                    file_info['model_type'] = metadata.get('model_type', 'Nieznany')
                    file_info['accuracy'] = metadata.get('accuracy', 'N/A')
                except Exception as e:
                    logger.warning(f"Błąd podczas ładowania metadanych dla {model_name}: {e}")
            
            # Próba identyfikacji typu modelu, jeśli brak metadanych
            if 'model_type' not in file_info:
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    file_info['model_type'] = type(model).__name__
                except Exception as e:
                    logger.warning(f"Nie można zidentyfikować typu modelu {model_name}: {e}")
                    file_info['model_type'] = 'Nieznany'
            
            models.append(file_info)
        
        logger.info(f"Znaleziono {len(models)} modeli w katalogu {models_dir}")
    
    except Exception as e:
        logger.error(f"Błąd podczas listowania modeli: {e}")
    
    return models

def is_model_newer_than_data(model_name: str, data_path: str) -> bool:
    """
    Sprawdza, czy model jest nowszy niż dane.
    
    Args:
        model_name: Nazwa modelu (bez rozszerzenia)
        data_path: Ścieżka do pliku z danymi
    
    Returns:
        bool: True, jeśli model jest nowszy niż dane
    """
    model_path = os.path.join("models", f"{model_name}_model.pkl")
    
    # Jeśli model nie istnieje, zwróć False
    if not os.path.exists(model_path):
        return False
    
    # Jeśli plik danych nie istnieje, zwróć True (zakładamy, że model jest aktualny)
    if not os.path.exists(data_path):
        return True
    
    # Porównaj czasy modyfikacji
    model_time = os.path.getmtime(model_path)
    data_time = os.path.getmtime(data_path)
    
    return model_time > data_time

def delete_model(name: str) -> bool:
    """
    Usuwa model i jego metadane.
    
    Args:
        name: Nazwa modelu (bez rozszerzenia)
    
    Returns:
        bool: Status sukcesu
    """
    model_path = os.path.join("models", f"{name}_model.pkl")
    metadata_path = os.path.join("models", f"{name}_metadata.json")
    
    try:
        # Usuń plik modelu
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Usunięto model {name}")
        
        # Usuń plik metadanych
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Usunięto metadane modelu {name}")
        
        return True
    
    except Exception as e:
        logger.error(f"Błąd podczas usuwania modelu {name}: {e}")
        return False

def copy_model_to_saved(name: str) -> bool:
    """
    Kopiuje model i jego metadane do katalogu saved_models.
    
    Args:
        name: Nazwa modelu (bez rozszerzenia)
    
    Returns:
        bool: Status sukcesu
    """
    ensure_model_dirs()
    
    model_path = os.path.join("models", f"{name}_model.pkl")
    metadata_path = os.path.join("models", f"{name}_metadata.json")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_model_path = os.path.join("saved_models", f"{name}_{timestamp}.pkl")
    saved_metadata_path = os.path.join("saved_models", f"{name}_{timestamp}_metadata.json")
    
    try:
        # Kopiuj plik modelu
        if os.path.exists(model_path):
            shutil.copy2(model_path, saved_model_path)
            logger.info(f"Skopiowano model {name} do {saved_model_path}")
        else:
            logger.warning(f"Model {name} nie istnieje: {model_path}")
            return False
        
        # Kopiuj plik metadanych
        if os.path.exists(metadata_path):
            shutil.copy2(metadata_path, saved_metadata_path)
            logger.info(f"Skopiowano metadane modelu {name} do {saved_metadata_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Błąd podczas kopiowania modelu {name}: {e}")
        return False

def load_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Ładuje metadane modelu.
    
    Args:
        model_name: Nazwa modelu (bez rozszerzenia)
    
    Returns:
        Dict[str, Any]: Metadane modelu
    """
    metadata_path = os.path.join("models", f"{model_name}_metadata.json")
    
    # Jeśli plik metadanych nie istnieje, zwróć pusty słownik
    if not os.path.exists(metadata_path):
        logger.warning(f"Brak pliku metadanych dla modelu {model_name}")
        return {}
    
    try:
        # Ładuj metadane
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Metadane modelu {model_name} zostały załadowane z {metadata_path}")
        
        return metadata
    
    except Exception as e:
        logger.error(f"Błąd podczas ładowania metadanych modelu {model_name}: {e}")
        return {}
