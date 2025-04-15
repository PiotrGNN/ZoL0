"""
Model Training Module
--------------------
This module provides functionality to train and evaluate models 
for market prediction and analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Union, List, Any, Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    tf = None

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor # Added for example usage
import pickle


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def prepare_data_for_model(data, features_count=None, expected_features=None):
    """
    Przygotowuje dane do formatu akceptowanego przez modele ML.

    Args:
        data: Dane wejściowe (np. DataFrame, słownik, lista)
        features_count: Opcjonalna liczba cech do zwrócenia (dopasowanie wymiarów)
        expected_features: Alias dla features_count, dla wstecznej kompatybilności

    Returns:
        Przygotowane dane w formacie numpy array lub DataFrame
    """
    # Jeśli podano expected_features, użyj go jako features_count
    if expected_features is not None and features_count is None:
        features_count = expected_features
    try:
        if isinstance(data, dict):
            # Jeśli dane są słownikiem, konwertuj na DataFrame lub numpy array
            if "close" in data and "open" in data:
                # Tworzenie prostych cech na podstawie dostępnych kolumn
                X = np.column_stack((
                    np.array(data['close']),
                    np.array(data['open']),
                ))

                # Jeśli określono features_count, dopasuj liczbę cech
                if features_count is not None and X.shape[1] != features_count:
                    logging.warning(f"Dopasowuję liczbę cech: {X.shape[1]} -> {features_count}")
                    if X.shape[1] > features_count:
                        # Jeśli mamy za dużo cech, wybierz pierwsze features_count
                        X = X[:, :features_count]
                    else:
                        # Jeśli mamy za mało cech, dodaj kolumny z zerami
                        padding = np.zeros((X.shape[0], features_count - X.shape[1]))
                        X = np.hstack((X, padding))
                return X
            else:
                # Gdy nie ma typowych kolumn OHLC, konwertujemy wszystkie wartości
                features = []
                for key, values in data.items():
                    if isinstance(values, (list, np.ndarray)):
                        features.append(np.array(values).reshape(-1, 1))

                result = np.hstack(features) if features else None

                # Dopasuj liczę cech jeśli potrzeba
                if features_count is not None and result is not None and result.shape[1] != features_count:
                    logging.warning(f"Dopasowuję liczbę cech: {result.shape[1]} -> {features_count}")
                    if result.shape[1] > features_count:
                        result = result[:, :features_count]
                    else:
                        padding = np.zeros((result.shape[0], features_count - result.shape[1]))
                        result = np.hstack((result, padding))
                return result

        elif isinstance(data, pd.DataFrame):
            # Dla DataFrame wybierz tylko kolumny numeryczne
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            result = data[numeric_cols].values

            # Dopasuj liczę cech jeśli potrzeba
            if features_count is not None and result.shape[1] != features_count:
                logging.warning(f"Dopasowuję liczbę cech: {result.shape[1]} -> {features_count}")
                if result.shape[1] > features_count:
                    result = result[:, :features_count]
                else:
                    padding = np.zeros((result.shape[0], features_count - result.shape[1]))
                    result = np.hstack((result, padding))
            return result

        elif isinstance(data, np.ndarray):
            # Już jest w formie numpy array, tylko dopasuj wymiary jeśli potrzeba
            result = data
            if features_count is not None and len(result.shape) > 1 and result.shape[1] != features_count:
                logging.warning(f"Dopasowuję liczbę cech: {result.shape[1]} -> {features_count}")
                if result.shape[1] > features_count:
                    result = result[:, :features_count]
                else:
                    padding = np.zeros((result.shape[0], features_count - result.shape[1]))
                    result = np.hstack((result, padding))
            return result

        elif isinstance(data, list):
            # Konwertuj listę na numpy array
            result = np.array(data)

            # Dostosuj wymiar jeśli potrzeba
            if features_count is not None and len(result.shape) > 1 and result.shape[1] != features_count:
                logging.warning(f"Dopasowuję liczbę cech: {result.shape[1]} -> {features_count}")
                if result.shape[1] > features_count:
                    result = result[:, :features_count]
                else:
                    padding = np.zeros((result.shape[0], features_count - result.shape[1]))
                    result = np.hstack((result, padding))
            return result

        else:
            logging.warning(f"Nieznany format danych: {type(data)}")
            return None
    except Exception as e:
        logging.error(f"Błąd podczas przygotowania danych: {e}")
        return None

def compute_data_hash(X, y=None):
    """
    Oblicza hash dla danych wejściowych, który można wykorzystać do sprawdzenia 
    czy dane treningowe się zmieniły.
    
    Args:
        X: Dane wejściowe (features)
        y: Dane wyjściowe (target), opcjonalne
        
    Returns:
        str: Hash reprezentujący dane
    """
    import hashlib
    import numpy as np
    
    try:
        # Oblicz statystyki dla X
        if hasattr(X, 'shape'):
            x_shape = str(X.shape)
            x_mean = str(np.mean(X) if X.size > 0 else 0)
            x_std = str(np.std(X) if X.size > 0 else 0)
        else:
            x_shape = str(len(X))
            x_mean = "unknown"
            x_std = "unknown"
            
        # Oblicz statystyki dla y, jeśli zostało podane
        if y is not None:
            if hasattr(y, 'shape'):
                y_shape = str(y.shape)
                y_mean = str(np.mean(y) if y.size > 0 else 0)
                y_std = str(np.std(y) if y.size > 0 else 0)
            else:
                y_shape = str(len(y))
                y_mean = "unknown"
                y_std = "unknown"
            
            # Połącz statystyki X i y
            data_repr = f"{x_shape}_{x_mean}_{x_std}_{y_shape}_{y_mean}_{y_std}"
        else:
            # Użyj tylko statystyk X
            data_repr = f"{x_shape}_{x_mean}_{x_std}"
        
        # Oblicz hash MD5
        return hashlib.md5(data_repr.encode()).hexdigest()
    except Exception as e:
        logging.error(f"Błąd podczas obliczania hasha danych: {e}")
        # W przypadku błędu zwróć losowy hash
        import time
        return hashlib.md5(str(time.time()).encode()).hexdigest()


class ModelTrainer:
    def __init__(
        self,
        model: Union["tf.keras.Model", Any],
        model_name: str,
        saved_model_dir: str = "saved_models",
        online_learning: bool = False,
        use_gpu: bool = False,
        early_stopping_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.saved_model_dir = saved_model_dir
        self.online_learning = online_learning
        self.use_gpu = use_gpu
        self.early_stopping_params = early_stopping_params or {}
        self.history = {}
        self.model_metadata = {}
        self.last_data_hash = None
        
        # Upewnij się, że katalog istnieje
        os.makedirs(self.saved_model_dir, exist_ok=True)
        
        # Upewnij się, że katalog models istnieje (do zapisywania modeli w nowym formacie)
        os.makedirs("models", exist_ok=True)

        if tf is not None and isinstance(model, tf.keras.Model) and use_gpu:
            logging.info("Trening z wykorzystaniem GPU/TPU (o ile jest dostępne).")
            
        # Spróbuj załadować istniejący model zamiast trenować od zera
        self._try_load_existing_model()

    def _try_load_existing_model(self) -> bool:
        """
        Próbuje załadować wcześniej zapisany model o tej samej nazwie.
        
        Returns:
            bool: True jeśli udało się załadować model, False w przeciwnym przypadku
        """
        try:
            # Sprawdź najpierw w katalogu models/
            model_path = os.path.join("models", f"{self.model_name.lower()}_model.pkl")
            
            if os.path.exists(model_path):
                logging.info(f"Znaleziono zapisany model {self.model_name} w {model_path}")
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                if 'model' in model_data and 'metadata' in model_data:
                    self.model = model_data['model']
                    self.model_metadata = model_data['metadata']
                    self.last_data_hash = self.model_metadata.get('data_hash')
                    
                    logging.info(f"Załadowano model {self.model_name} (trenowany: {self.model_metadata.get('train_date', 'nieznany')})")
                    return True
                else:
                    logging.warning(f"Plik modelu {model_path} nie zawiera wymaganych danych")
            
            # Sprawdź również w standardowym katalogu saved_model_dir
            # Znajdź najnowszy plik modelu
            model_files = []
            for file in os.listdir(self.saved_model_dir):
                if file.startswith(f"{self.model_name}_") and (file.endswith(".pkl") or file.endswith(".h5")):
                    file_path = os.path.join(self.saved_model_dir, file)
                    model_files.append((file_path, os.path.getmtime(file_path)))
            
            if model_files:
                # Sortuj po czasie modyfikacji (od najnowszego)
                model_files.sort(key=lambda x: x[1], reverse=True)
                latest_model_path = model_files[0][0]
                
                logging.info(f"Znaleziono zapisany model {self.model_name} w {latest_model_path}")
                
                # Załaduj model
                if latest_model_path.endswith(".pkl"):
                    with open(latest_model_path, 'rb') as f:
                        # Stary format - sam model bez metadanych
                        self.model = pickle.load(f)
                        self.model_metadata = {'train_date': datetime.fromtimestamp(model_files[0][1]).isoformat()}
                elif latest_model_path.endswith(".h5") and tf is not None:
                    self.model = tf.keras.models.load_model(latest_model_path)
                    self.model_metadata = {'train_date': datetime.fromtimestamp(model_files[0][1]).isoformat()}
                
                logging.info(f"Załadowano model {self.model_name} (trenowany: {self.model_metadata.get('train_date', 'nieznany')})")
                return True
            
            return False
            
        except Exception as e:
            logging.warning(f"Nie można załadować zapisanego modelu {self.model_name}: {e}")
            return False

    def walk_forward_split(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> List[tuple]:
        splits = []
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            splits.append((X_train, y_train, X_val, y_val))
        return splits

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        force_train: bool = False,
    ) -> Dict[str, Any]:
        """
        Trenuje model na danych, z inteligentnym cachingiem.
        
        Args:
            X: Dane wejściowe
            y: Dane wyjściowe
            n_splits: Liczba podziałów danych w walidacji krzyżowej
            epochs: Liczba epok dla modeli głębokiego uczenia
            batch_size: Rozmiar partii danych dla modeli głębokiego uczenia
            force_train: Czy wymusić trenowanie modelu nawet jeśli istnieje zapisany model
            
        Returns:
            Dict[str, Any]: Wyniki treningu
        """
        try:
            # Oblicz hash danych treningowych
            current_data_hash = compute_data_hash(X, y)
            
            # Sprawdź czy model już istnieje i czy dane się zmieniły
            if not force_train and self.last_data_hash == current_data_hash and self.model_metadata:
                logging.info(f"Model {self.model_name} jest aktualny (hash danych się nie zmienił)")
                return {
                    "success": True,
                    "message": "Model jest aktualny (dane nie zmieniły się)",
                    "is_updated": False,
                    "metrics": self.model_metadata.get("metrics", {}),
                    "accuracy": self.model_metadata.get("accuracy", 0.0)
                }
            
            # Jeśli dane się zmieniły lub wymuszono trening, trenuj model
            splits = self.walk_forward_split(X, y, n_splits=n_splits)
            metrics = {"mse": [], "mae": []}

            for fold, (X_train, y_train, X_val, y_val) in enumerate(splits, start=1):
                logging.info("Rozpoczynam trening (fold %d / %d)", fold, n_splits)

                # Prepare data for the model (handling different data types)
                X_train_prepared = prepare_data_for_model(X_train, expected_features=2) # Specifying expected features
                X_val_prepared = prepare_data_for_model(X_val, expected_features=2) # Specifying expected features
                y_train_prepared = prepare_data_for_model(y_train)
                y_val_prepared = prepare_data_for_model(y_val)


                if tf is not None and isinstance(self.model, tf.keras.Model):
                    callbacks = []
                    if self.early_stopping_params:
                        callbacks.append(EarlyStopping(**self.early_stopping_params))
                    checkpoint_path = os.path.join(
                        self.saved_model_dir, f"{self.model_name}_fold{fold}.h5"
                    )
                    callbacks.append(
                        ModelCheckpoint(
                            checkpoint_path,
                            save_best_only=True,
                            monitor=self.early_stopping_params.get(
                                "monitor", "val_loss"
                            ),
                            verbose=1,
                        )
                    )

                    history = self.model.fit(
                        X_train_prepared,
                        y_train_prepared,
                        validation_data=(X_val_prepared, y_val_prepared),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1,
                    )
                    self.history = history.history
                    self.model = tf.keras.models.load_model(checkpoint_path)
                    predictions = self.model.predict(X_val_prepared)

                else:
                    self.model.fit(X_train_prepared, y_train_prepared)
                    predictions = self.model.predict(X_val_prepared)

                mse = mean_squared_error(y_val_prepared, predictions)
                mae = mean_absolute_error(y_val_prepared, predictions)
                metrics["mse"].append(mse)
                metrics["mae"].append(mae)

                logging.info("Fold %d - MSE: %.4f, MAE: %.4f", fold, mse, mae)

                if self.online_learning and hasattr(self.model, "partial_fit"):
                    logging.info(
                        "Aktualizacja modelu metodą partial_fit (online learning)."
                    )
                    self.model.partial_fit(X_val_prepared, y_val_prepared)

            avg_mse = float(np.mean(metrics["mse"]))
            avg_mae = float(np.mean(metrics["mae"]))
            accuracy = 1.0 / (1.0 + avg_mse)
            logging.info("Średnie MSE: %.4f, Średnie MAE: %.4f, Accuracy: %.4f", avg_mse, avg_mae, accuracy)
            
            # Zapisz metadane modelu
            self.model_metadata = {
                "train_date": datetime.now().isoformat(),
                "data_hash": current_data_hash,
                "metrics": {
                    "mse": avg_mse,
                    "mae": avg_mae
                },
                "accuracy": accuracy,
                "features_shape": X.shape if hasattr(X, 'shape') else None,
                "target_shape": y.shape if hasattr(y, 'shape') else None,
                "name": self.model_name,
                "model_type": str(type(self.model)),
            }
            
            # Dodaj parametry modelu do metadanych, jeśli model je posiada
            if hasattr(self.model, 'get_params'):
                self.model_metadata["params"] = self.model.get_params()
            
            # Zapisz model w starym i nowym formacie
            self._save_model_old_format()
            self._save_model_new_format()
            
            self.last_data_hash = current_data_hash
            
            return {
                "success": True,
                "is_updated": True,
                "metrics": metrics,
                "accuracy": accuracy,
                "model_metadata": self.model_metadata
            }

        except Exception as e:
            logging.error("Błąd podczas treningu modelu: %s", e)
            return {"success": False, "error": str(e)}

    def _save_model_old_format(self) -> None:
        """Zapisuje model w starym formacie (dla kompatybilności)"""
        try:
            # Sprawdzenie, czy model został wytrenowany
            if hasattr(self.model, 'n_features_in_') or hasattr(self.model, 'feature_importances_') or \
               (tf is not None and isinstance(self.model, tf.keras.Model) and len(self.model.layers) > 0):

                # Jeśli to model Sklearn i nie był trenowany, dodajemy ostrzeżenie
                if hasattr(self.model, 'fit') and not hasattr(self.model, 'n_features_in_') and \
                   not (tf is not None and isinstance(self.model, tf.keras.Model)):
                    logging.warning("Model %s może nie być wytrenowany! Sprawdź czy wywołano fit() przed zapisem.", 
                                  self.model_name)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = os.path.join(
                    self.saved_model_dir, f"{self.model_name}_{timestamp}"
                )

                if tf is not None and isinstance(self.model, tf.keras.Model):
                    # Sprawdzenie czy model ma warstwy
                    if not self.model.layers:
                        logging.error("Model Sequential nie ma warstw! Anulowanie zapisu.")
                        return

                    model_filename += ".h5"
                    self.model.save(model_filename)
                else:
                    model_filename += ".pkl"
                    with open(model_filename, "wb") as f:
                        pickle.dump(self.model, f)

                logging.info("Model zapisany w starym formacie: %s", model_filename)
            else:
                logging.error("Model %s nie został wytrenowany lub nie ma warstw! Anulowanie zapisu.", 
                            self.model_name)

        except Exception as e:
            logging.error("Błąd podczas zapisywania modelu w starym formacie: %s", e)
    
    def _save_model_new_format(self) -> None:
        """Zapisuje model w nowym formacie z metadanymi"""
        try:
            # Sprawdzenie, czy model został wytrenowany
            if hasattr(self.model, 'n_features_in_') or hasattr(self.model, 'feature_importances_') or \
               (tf is not None and isinstance(self.model, tf.keras.Model) and len(self.model.layers) > 0):
                
                # Przygotuj ścieżkę do zapisu modelu
                model_filename = os.path.join("models", f"{self.model_name.lower()}_model.pkl")
                
                # Przygotuj dane modelu z metadanymi
                model_data = {
                    "model": self.model,
                    "metadata": self.model_metadata
                }
                
                # Zapisz model z metadanymi
                with open(model_filename, "wb") as f:
                    pickle.dump(model_data, f)
                
                logging.info("Model zapisany w nowym formacie z metadanymi: %s", model_filename)
            else:
                logging.error("Model %s nie został wytrenowany lub nie ma warstw! Anulowanie zapisu.", 
                            self.model_name)
                            
        except Exception as e:
            logging.error("Błąd podczas zapisywania modelu w nowym formacie: %s", e)
            
    def save_model(self) -> None:
        """Zapisuje model w obu formatach (stary i nowy)"""
        self._save_model_old_format()
        self._save_model_new_format()


def get_example_data():
    """
    Tworzy i zwraca przykładowe dane do treningu modeli.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X_train (cechy) i y_train (wartości docelowe)
    """
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
    X_train = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, size=500),
            "feature2": np.random.normal(0, 1, size=500),
        },
        index=dates,
    )
    y_train = (
        X_train["feature1"] * 1.5
        + X_train["feature2"] * (-2.0)
        + np.random.normal(0, 0.5, size=500)
    )
    return X_train, y_train

# Tworzenie przykładowych danych, które można zaimportować w innych modułach
X_train, y_train = get_example_data()

if __name__ == "__main__":
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        trainer = ModelTrainer(
            model=model,
            model_name="RandomForest_Model",
            online_learning=True,
            early_stopping_params={"monitor": "val_loss", "patience": 5},
        )
        metrics = trainer.train(X_train, y_train, n_splits=5)
        logging.info("Trening zakończony. Wyniki walidacji: %s", metrics)
    except Exception as e:
        logging.error("Błąd w przykładowym użyciu ModelTrainer: %s", e)
        raise