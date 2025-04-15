
"""
model_utils.py
-------------
Narzędzia pomocnicze do zarządzania modelami AI, w tym zapisywanie, wczytywanie
i walidacja modeli.
"""

import os
import json
import logging
import time
import datetime
import hashlib
from typing import Dict, Any, Optional, Tuple, List, Union

import joblib
import numpy as np

logger = logging.getLogger("model_utils")

# Stałe
MODELS_DIR = "models"
SAVED_MODELS_DIR = "saved_models"
MODELS_METADATA_FILE = os.path.join(MODELS_DIR, "models_metadata.json")

def ensure_model_dirs():
    """Tworzy niezbędne katalogi dla modeli, jeśli nie istnieją."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVED_MODELS_DIR, "checkpoints"), exist_ok=True)

def calculate_data_hash(data: Any) -> str:
    """
    Oblicza hash dla danych treningowych.
    
    Args:
        data: Dane treningowe (może być np. numpy array, DataFrame, lista, słownik)
        
    Returns:
        str: Hash danych
    """
    # Konwersja różnych typów danych do formatu tekstowego
    if isinstance(data, dict):
        try:
            import json
            data_str = json.dumps(data, sort_keys=True)
        except:
            data_str = str(data)
    elif hasattr(data, 'tolist') and callable(getattr(data, 'tolist')):
        try:
            # Dla np.array lub pandas
            data_str = str(data.tolist())
        except:
            data_str = str(data)
    elif hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
        try:
            # Dla pandas DataFrame
            data_str = str(data.to_dict())
        except:
            data_str = str(data)
    else:
        data_str = str(data)
    
    # Obliczenie hasha
    hash_obj = hashlib.sha256(data_str.encode())
    return hash_obj.hexdigest()

def save_model(model: Any, model_name: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Zapisuje model wraz z metadanymi.
    
    Args:
        model: Model do zapisania
        model_name: Nazwa modelu
        metadata: Opcjonalne metadane
        
    Returns:
        Tuple[bool, str]: (sukces, ścieżka_do_pliku)
    """
    ensure_model_dirs()
    
    # Przygotuj metadane
    if metadata is None:
        metadata = {}
    
    # Dodaj podstawowe metadane
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(' ', '_').lower()
    
    # Pełna ścieżka pliku
    file_path = os.path.join(MODELS_DIR, f"{safe_model_name}_model.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"{safe_model_name}_metadata.json")
    
    # Dodatkowa archiwalna kopia
    archive_path = os.path.join(SAVED_MODELS_DIR, f"{safe_model_name}_{timestamp}.pkl")
    
    # Uzupełnij metadane
    metadata.update({
        "model_name": model_name,
        "original_file": file_path,
        "archived_file": archive_path,
        "saved_at": timestamp,
        "model_type": type(model).__name__,
        "model_module": type(model).__module__
    })
    
    # Zapisz model
    try:
        joblib.dump(model, file_path)
        # Dodatkowo zapisz archiwalną kopię
        joblib.dump(model, archive_path)
        
        # Zapisz metadane
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Aktualizuj główny plik metadanych
        update_models_metadata(model_name, metadata)
        
        logger.info(f"Model {model_name} zapisany do {file_path} i {archive_path}")
        return True, file_path
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania modelu {model_name}: {e}")
        return False, str(e)

def load_model(model_name: str) -> Tuple[Any, Dict[str, Any], bool]:
    """
    Wczytuje model na podstawie nazwy.
    
    Args:
        model_name: Nazwa modelu
        
    Returns:
        Tuple[Any, Dict, bool]: (model, metadane, sukces)
    """
    safe_model_name = model_name.replace(' ', '_').lower()
    file_path = os.path.join(MODELS_DIR, f"{safe_model_name}_model.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"{safe_model_name}_metadata.json")
    
    try:
        # Sprawdź, czy pliki istnieją
        if not os.path.exists(file_path):
            logger.warning(f"Plik modelu {file_path} nie istnieje")
            return None, {}, False
        
        # Wczytaj model
        model = joblib.load(file_path)
        
        # Wczytaj metadane, jeśli istnieją
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Błąd podczas odczytu metadanych {metadata_path}: {e}")
        
        logger.info(f"Model {model_name} wczytany z {file_path}")
        return model, metadata, True
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania modelu {model_name}: {e}")
        return None, {}, False

def update_models_metadata(model_name: str, metadata: Dict[str, Any]) -> bool:
    """
    Aktualizuje główny plik metadanych modeli.
    
    Args:
        model_name: Nazwa modelu
        metadata: Metadane modelu
        
    Returns:
        bool: Sukces
    """
    try:
        # Wczytaj istniejące metadane
        all_metadata = {}
        if os.path.exists(MODELS_METADATA_FILE):
            try:
                with open(MODELS_METADATA_FILE, 'r') as f:
                    all_metadata = json.load(f)
            except:
                all_metadata = {}
        
        # Aktualizuj informacje o tym modelu
        all_metadata[model_name] = metadata
        
        # Zapisz z powrotem
        with open(MODELS_METADATA_FILE, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji metadanych modeli: {e}")
        return False

def get_all_models_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Pobiera metadane wszystkich zapisanych modeli.
    
    Returns:
        Dict[str, Dict[str, Any]]: Metadane modeli
    """
    if os.path.exists(MODELS_METADATA_FILE):
        try:
            with open(MODELS_METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Błąd podczas odczytu metadanych modeli: {e}")
    
    return {}

def model_exists(model_name: str) -> bool:
    """
    Sprawdza, czy model o podanej nazwie istnieje.
    
    Args:
        model_name: Nazwa modelu
        
    Returns:
        bool: Czy model istnieje
    """
    safe_model_name = model_name.replace(' ', '_').lower()
    file_path = os.path.join(MODELS_DIR, f"{safe_model_name}_model.pkl")
    return os.path.exists(file_path)

def create_model_checkpoint(model: Any, model_name: str, checkpoint_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Tworzy punkt kontrolny modelu.
    
    Args:
        model: Model
        model_name: Nazwa modelu
        checkpoint_name: Opcjonalna nazwa punktu kontrolnego
        
    Returns:
        Tuple[bool, str]: (sukces, ścieżka do punktu kontrolnego)
    """
    ensure_model_dirs()
    
    # Przygotuj nazwę punktu kontrolnego
    safe_model_name = model_name.replace(' ', '_').lower()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if checkpoint_name:
        checkpoint_filename = f"{safe_model_name}_{checkpoint_name}_{timestamp}.pkl"
    else:
        checkpoint_filename = f"{safe_model_name}_checkpoint_{timestamp}.pkl"
    
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, "checkpoints", checkpoint_filename)
    
    try:
        # Zapisz punkt kontrolny
        joblib.dump(model, checkpoint_path)
        logger.info(f"Utworzono punkt kontrolny modelu {model_name}: {checkpoint_path}")
        return True, checkpoint_path
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia punktu kontrolnego modelu {model_name}: {e}")
        return False, str(e)

def check_model_data_compatibility(model_name: str, data: Any) -> bool:
    """
    Sprawdza, czy dane są kompatybilne z zapisanym modelem.
    
    Args:
        model_name: Nazwa modelu
        data: Dane do sprawdzenia
        
    Returns:
        bool: Czy dane są kompatybilne
    """
    # Wczytaj metadane modelu
    safe_model_name = model_name.replace(' ', '_').lower()
    metadata_path = os.path.join(MODELS_DIR, f"{safe_model_name}_metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Brak metadanych dla modelu {model_name}")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Sprawdź, czy metadane zawierają informację o hashu danych
        if "data_hash" not in metadata:
            logger.warning(f"Metadane modelu {model_name} nie zawierają informacji o hashu danych")
            return False
        
        # Oblicz hash aktualnych danych
        current_hash = calculate_data_hash(data)
        
        # Porównaj hashe
        return metadata["data_hash"] == current_hash
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania kompatybilności danych z modelem {model_name}: {e}")
        return False

def list_available_models() -> List[Dict[str, Any]]:
    """
    Zwraca listę dostępnych modeli z podstawowymi informacjami.
    
    Returns:
        List[Dict[str, Any]]: Lista modeli
    """
    ensure_model_dirs()
    
    models_info = []
    
    # Zbierz informacje o modelach w głównym katalogu
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith("_model.pkl"):
            model_name = filename.replace("_model.pkl", "")
            
            # Sprawdź metadane
            metadata_path = os.path.join(MODELS_DIR, f"{model_name}_metadata.json")
            metadata = {}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            models_info.append({
                "name": model_name,
                "path": os.path.join(MODELS_DIR, filename),
                "size": os.path.getsize(os.path.join(MODELS_DIR, filename)),
                "last_modified": datetime.datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(MODELS_DIR, filename))
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": metadata
            })
    
    return models_info

# Inicjalizacja przy importowaniu
ensure_model_dirs()
