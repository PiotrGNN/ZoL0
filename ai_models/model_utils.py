"""
model_utils.py
-------------
Moduł zawierający narzędzia pomocnicze do pracy z modelami.
"""

import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import glob

# Klasa opakowująca funkcje pomocnicze w interfejs zgodny z fit/predict
class ModelUtilsWrapper:
    """
    Klasa opakowująca funkcje pomocnicze z model_utils.py, aby zapewnić 
    interfejs zgodny z wymaganiem posiadania metod fit() i predict().
    """
    
    def __init__(self):
        """Inicjalizacja wrappera narzędzi modelowych."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ModelUtilsWrapper zainicjalizowany")
        self.models = {}
        self.metadata = {}
        
    def fit(self, data=None, model_path=None):
        """
        Symuluje trening modelu (w rzeczywistości tylko sprawdza dostępne modele).
        
        Args:
            data: Dane treningowe (opcjonalne)
            model_path: Ścieżka do modelu (opcjonalne)
            
        Returns:
            bool: True jeśli operacja się powiodła
        """
        try:
            self.models = {}
            available_models = list_available_models(model_path if model_path else "models")
            for model_info in available_models:
                self.models[model_info.get("name", "unknown")] = model_info
            self.logger.info(f"Znaleziono {len(self.models)} dostępnych modeli")
            return True
        except Exception as e:
            self.logger.error(f"Błąd podczas fit(): {e}")
            return False
    
    def predict(self, data=None):
        """
        Symuluje przewidywanie modelu (w rzeczywistości tylko zwraca metadane modeli).
        
        Args:
            data: Dane wejściowe (opcjonalne)
            
        Returns:
            Dict: Informacje o dostępnych modelach
        """
        try:
            if not self.models:
                self.fit()
            
            return {
                "available_models": list(self.models.keys()),
                "models_count": len(self.models),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas predict(): {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Konfiguracja logowania
logger = logging.getLogger("model_utils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_model(path: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Load model with version compatibility checking.
    
    Args:
        path: Path to the model file
        
    Returns:
        Tuple of (model, metadata) or (None, error_info)
    """
    try:
        import joblib
        import sklearn
        
        logger.info(f"Loading model from {path}")
        model_data = joblib.load(path)
        
        if isinstance(model_data, dict) and "sklearn_version" in model_data:
            saved_version = model_data["sklearn_version"]
            current_version = sklearn.__version__
            
            # Check major version compatibility
            if saved_version.split('.')[0] != current_version.split('.')[0]:
                logger.warning(f"Model was saved with scikit-learn {saved_version}, "
                             f"but current version is {current_version}")
            
            model = model_data["model"]
            metadata = {
                **model_data.get("metadata", {}),
                "sklearn_version": saved_version,
                "current_sklearn_version": current_version,
                "load_time": datetime.now().isoformat()
            }
            
            return model, metadata
        else:
            # Handle old format models
            return model_data, {
                "note": "Legacy model format",
                "current_sklearn_version": sklearn.__version__,
                "load_time": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, {"error": str(e)}

def save_model(model: Any, path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save model with version compatibility handling.
    
    Args:
        model: The model to save
        path: Path to save the model
        metadata: Optional metadata to save with the model
        
    Returns:
        bool: Success status
    """
    try:
        import joblib
        import sklearn
        
        # Create model data with version info
        model_data = {
            "model": model,
            "sklearn_version": sklearn.__version__,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model with protocol=4 for better compatibility
        joblib.dump(model_data, path, protocol=4)
        logger.info(f"Successfully saved model to {path} (sklearn {sklearn.__version__})")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False

def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Ładuje tylko metadane modelu bez ładowania całego modelu.
    
    Args:
        model_path: Ścieżka do pliku modelu
        
    Returns:
        Dict[str, Any]: Metadane modelu
    """
    try:
        # Sprawdź czy plik istnieje
        if not os.path.exists(model_path):
            logger.error(f"Plik modelu {model_path} nie istnieje")
            return {"error": f"Plik modelu {model_path} nie istnieje"}
        
        # Sprawdź rozszerzenie pliku
        _, ext = os.path.splitext(model_path)
        
        # Sprawdź, czy istnieje plik metadanych
        metadata_path = model_path.replace(ext, '_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Załadowano metadane z {metadata_path}")
                return metadata
            except Exception as e:
                logger.warning(f"Błąd podczas ładowania metadanych z {metadata_path}: {e}")
        
        # Jeśli nie ma osobnego pliku metadanych, spróbuj załadować z pliku modelu
        if ext.lower() == '.pkl':
            with open(model_path, 'rb') as f:
                try:
                    # Otwórz plik i sprawdź czy zawiera metadane
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'metadata' in data:
                        logger.info(f"Załadowano metadane z pliku modelu {model_path}")
                        return data['metadata']
                    else:
                        # Brak metadanych, zwróć podstawowe informacje o pliku
                        return {
                            "name": os.path.basename(model_path).split('.')[0],
                            "file_path": model_path,
                            "file_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                            "file_size_bytes": os.path.getsize(model_path)
                        }
                except Exception as e:
                    logger.error(f"Błąd podczas ładowania metadanych z pliku modelu {model_path}: {e}")
                    return {"error": str(e)}
        else:
            # Dla innych typów plików zwróć podstawowe informacje o pliku
            return {
                "name": os.path.basename(model_path).split('.')[0],
                "file_path": model_path,
                "file_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                "file_size_bytes": os.path.getsize(model_path)
            }
            
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas ładowania metadanych: {e}")
        return {"error": str(e)}

def list_available_models(model_dir: str = "models") -> List[Dict[str, Any]]:
    """
    Zwraca listę dostępnych modeli z ich metadanymi.
    
    Args:
        model_dir: Katalog z modelami
        
    Returns:
        List[Dict[str, Any]]: Lista modeli z metadanymi
    """
    available_models = []
    
    try:
        # Sprawdź, czy katalog istnieje
        if not os.path.exists(model_dir):
            logger.warning(f"Katalog {model_dir} nie istnieje")
            return []
        
        # Znajdź wszystkie pliki modeli
        model_files = []
        for ext in ['.pkl', '.h5']:
            model_files.extend(glob.glob(os.path.join(model_dir, f"*{ext}")))
            
        # Jeśli nie ma podkatalogu "saved_models", sprawdź też tam
        if model_dir != "saved_models" and os.path.exists("saved_models"):
            for ext in ['.pkl', '.h5']:
                model_files.extend(glob.glob(os.path.join("saved_models", f"*{ext}")))
        
        # Załaduj metadane dla każdego modelu
        for model_path in model_files:
            metadata = load_model_metadata(model_path)
            metadata["file_path"] = model_path
            available_models.append(metadata)
        
        logger.info(f"Znaleziono {len(available_models)} modeli")
        return available_models
    except Exception as e:
        logger.error(f"Błąd podczas listowania dostępnych modeli: {e}")
        return []

def check_model_compatibility(model: Any, input_shape: Tuple) -> bool:
    """
    Sprawdza, czy model jest kompatybilny z podanym kształtem danych wejściowych.
    
    Args:
        model: Model do sprawdzenia
        input_shape: Oczekiwany kształt danych wejściowych
        
    Returns:
        bool: True jeśli model jest kompatybilny, False w przeciwnym razie
    """
    try:
        # Stwórz przykładowe dane wejściowe
        sample_input = np.random.random(input_shape)
        
        # Sprawdź, czy model ma metodę predict
        if not hasattr(model, 'predict'):
            logger.error("Model nie ma metody predict")
            return False
        
        # Spróbuj wykonać predykcję
        model.predict(sample_input)
        
        logger.info(f"Model jest kompatybilny z kształtem {input_shape}")
        return True
    except Exception as e:
        logger.error(f"Model nie jest kompatybilny z kształtem {input_shape}: {e}")
        return False

def convert_model_format(model_path: str, target_format: str) -> Optional[str]:
    """
    Konwertuje model do innego formatu.
    
    Args:
        model_path: Ścieżka do pliku modelu
        target_format: Format docelowy ('pkl', 'h5', 'onnx')
        
    Returns:
        Optional[str]: Ścieżka do przekonwertowanego modelu lub None w przypadku błędu
    """
    try:
        # Załaduj model
        model, metadata = load_model(model_path)
        if model is None:
            logger.error(f"Nie udało się załadować modelu z {model_path}")
            return None
        
        # Przygotuj ścieżkę docelową
        base_path = os.path.splitext(model_path)[0]
        target_path = f"{base_path}.{target_format}"
        
        # Konwertuj model
        if target_format.lower() == 'pkl':
            # Zapisz jako plik PKL
            success = save_model(model, target_path, metadata)
            if success:
                logger.info(f"Model przekonwertowany do {target_path}")
                return target_path
            else:
                logger.error(f"Nie udało się zapisać modelu do {target_path}")
                return None
                
        elif target_format.lower() == 'h5':
            try:
                # Sprawdź, czy model jest instancją TensorFlow
                import tensorflow as tf
                if isinstance(model, tf.keras.Model):
                    model.save(target_path)
                    
                    # Zapisz metadane jako osobny plik JSON
                    metadata_path = target_path.replace('.h5', '_metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                        
                    logger.info(f"Model przekonwertowany do {target_path}")
                    return target_path
                else:
                    logger.error(f"Model nie jest instancją tf.keras.Model, nie można przekonwertować do .h5")
                    return None
            except ImportError:
                logger.error("Nie można zaimportować TensorFlow. Zainstaluj pakiet tensorflow.")
                return None
            except Exception as e:
                logger.error(f"Błąd podczas konwersji do formatu H5: {e}")
                return None
                
        elif target_format.lower() == 'onnx':
            try:
                import onnx
                import skl2onnx
                import tensorflow as tf
                
                # Sprawdź typ modelu i skonwertuj odpowiednio
                if 'sklearn' in str(type(model)):
                    # Konwersja modelu scikit-learn do ONNX
                    try:
                        # Pozyskaj wymiary wejściowe z metadanych
                        if 'features_shape' in metadata:
                            n_features = metadata['features_shape'][1] if len(metadata['features_shape']) > 1 else 1
                        else:
                            n_features = 10  # Domyślna wartość
                            
                        # Definiuj wejście
                        initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, n_features]))]
                        
                        # Konwertuj model
                        onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
                        
                        # Zapisz model
                        onnx.save(onnx_model, target_path)
                        
                        logger.info(f"Model scikit-learn przekonwertowany do {target_path}")
                        return target_path
                    except Exception as e:
                        logger.error(f"Błąd podczas konwersji modelu scikit-learn do ONNX: {e}")
                        return None
                        
                elif isinstance(model, tf.keras.Model):
                    # Konwersja modelu TensorFlow do ONNX
                    try:
                        import tf2onnx
                        
                        # Konwertuj model
                        onnx_model, _ = tf2onnx.convert.from_keras(model)
                        
                        # Zapisz model
                        onnx.save(onnx_model, target_path)
                        
                        logger.info(f"Model TensorFlow przekonwertowany do {target_path}")
                        return target_path
                    except Exception as e:
                        logger.error(f"Błąd podczas konwersji modelu TensorFlow do ONNX: {e}")
                        return None
                else:
                    logger.error(f"Nieobsługiwany typ modelu dla konwersji do ONNX: {type(model)}")
                    return None
            except ImportError:
                logger.error("Nie można zaimportować wymaganych pakietów do konwersji ONNX. Zainstaluj 'onnx', 'skl2onnx' i 'tf2onnx'.")
                return None
        else:
            logger.error(f"Nieobsługiwany format docelowy: {target_format}")
            return None
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas konwersji modelu: {e}")
        return None

def get_model_performance(model_path: str, test_data: Optional[Union[pd.DataFrame, np.ndarray]] = None, 
                       test_labels: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Zwraca informacje o wydajności modelu.
    
    Args:
        model_path: Ścieżka do pliku modelu
        test_data: Dane testowe (opcjonalne)
        test_labels: Etykiety testowe (opcjonalne)
        
    Returns:
        Dict[str, Any]: Informacje o wydajności modelu
    """
    try:
        # Załaduj model i metadane
        model, metadata = load_model(model_path)
        if model is None:
            logger.error(f"Nie udało się załadować modelu z {model_path}")
            return {"error": f"Nie udało się załadować modelu z {model_path}"}
        
        # Przygotuj informacje o modelu
        performance_info = {
            "model_name": metadata.get("name", os.path.basename(model_path).split('.')[0]),
            "model_type": metadata.get("model_type", str(type(model))),
            "train_date": metadata.get("train_date", "unknown"),
            "accuracy": metadata.get("accuracy", "N/A"),
            "metrics": metadata.get("metrics", {}),
            "file_path": model_path,
            "file_size_bytes": os.path.getsize(model_path),
            "creation_date": datetime.fromtimestamp(os.path.getctime(model_path)).isoformat(),
            "modification_date": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
        }
        
        # Jeśli podano dane testowe, wykonaj dodatkowe testy wydajności
        if test_data is not None:
            try:
                # Sprawdź, czy model ma metodę predict
                if hasattr(model, 'predict'):
                    # Mierz czas predykcji
                    import time
                    start_time = time.time()
                    predictions = model.predict(test_data)
                    execution_time = time.time() - start_time
                    
                    performance_info["prediction_time_ms"] = execution_time * 1000
                    performance_info["samples_per_second"] = len(test_data) / execution_time if hasattr(test_data, '__len__') else "unknown"
                    
                    # Jeśli podano etykiety testowe, oblicz metryki
                    if test_labels is not None:
                        try:
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            # Oblicz metryki
                            mse = mean_squared_error(test_labels, predictions)
                            mae = mean_absolute_error(test_labels, predictions)
                            r2 = r2_score(test_labels, predictions)
                            
                            performance_info["test_metrics"] = {
                                "mse": float(mse),
                                "mae": float(mae),
                                "r2": float(r2)
                            }
                        except ImportError:
                            logger.warning("Nie można zaimportować scikit-learn. Zainstaluj pakiet scikit-learn.")
                        except Exception as e:
                            logger.warning(f"Błąd podczas obliczania metryk: {e}")
            except Exception as e:
                logger.warning(f"Błąd podczas testowania wydajności: {e}")
        
        return performance_info
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas pobierania informacji o wydajności modelu: {e}")
        return {"error": str(e)}
