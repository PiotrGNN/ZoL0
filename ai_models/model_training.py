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


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def prepare_data_for_model(data: Union[Dict, List, np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Konwertuje różne formaty danych do formatu odpowiedniego dla modeli ML/AI.

    Args:
        data: Dane wejściowe w różnych formatach (słownik, lista, DataFrame, array)

    Returns:
        np.ndarray: Dane w formacie numpy.ndarray gotowe do użycia w modelach
    """
    if data is None:
        raise ValueError("Dane wejściowe nie mogą być None")

    # Jeśli to słownik OHLCV, konwertujemy na DataFrame
    if isinstance(data, dict):
        logging.info("Konwersja słownika danych na DataFrame")
        try:
            # Sprawdźmy, czy mamy do czynienia ze słownikiem, który zawiera listy
            is_dict_of_lists = all(isinstance(v, (list, np.ndarray)) for v in data.values() if v is not None)

            # Jeśli to słownik OHLCV z listami, np. {'open': [1,2,3], 'high': [2,3,4], ...}
            if is_dict_of_lists:
                # Usuwamy timestamp jeśli istnieje (nie jest zwykle potrzebny jako cecha)
                data_without_timestamp = {k: v for k, v in data.items() if k != 'timestamp'}

                # Konwertuj na DataFrame
                df = pd.DataFrame(data_without_timestamp)

                # Wybieramy kolumny OHLCV jeśli istnieją
                cols_to_use = []
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        cols_to_use.append(col)

                # Jeśli znaleziono odpowiednie kolumny, użyj ich
                if cols_to_use:
                    # Sprawdź czy wszystkie wartości są liczbami
                    for col in cols_to_use:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Sprawdź czy jest indeks - jeśli nie, dodaj go
                    if df.index.name is None:
                        df.reset_index(drop=True, inplace=True)

                    logging.info(f"Przekształcono słownik na tablicę numpy o kształcie {df[cols_to_use].values.shape}")
                    return df[cols_to_use].values

                # W przeciwnym razie użyj wszystkich kolumn numerycznych
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    logging.info(f"Używam tylko kolumn numerycznych: {numeric_cols}")
                    return df[numeric_cols].values

                logging.info(f"Używam wszystkich kolumn z DataFrame o kształcie {df.values.shape}")
                return df.values

            # Jeśli to pojedynczy element (np. dict z pojedynczymi wartościami)
            else:
                # Próbujemy utworzyć jednoelementowy DataFrame
                df = pd.DataFrame([data])
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    logging.info(f"Przekształcono słownik pojedynczych wartości na tablicę numpy")
                    return df[numeric_cols].values

                return df.values

        except Exception as e:
            logging.error(f"Błąd podczas konwersji słownika: {e}")
            # Jako fallback, tworzymy tablicę 2D z pierwszych wartości liczbowych, jakie znajdziemy
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    if all(isinstance(x, (int, float)) for x in value):
                        logging.info(f"Używam wartości z klucza {key} jako danych wejściowych")
                        return np.array(value).reshape(-1, 1)

            raise ValueError(f"Nie można przekonwertować słownika do formatu numerycznego: {data}")

    # Jeśli to DataFrame, konwertujemy na numpy array
    elif isinstance(data, pd.DataFrame):
        logging.info("Konwersja DataFrame na numpy array")
        return data.values

    # Jeśli to lista, konwertujemy na numpy array
    elif isinstance(data, list):
        logging.info("Konwersja listy na numpy array")
        return np.array(data)

    # Jeśli to już numpy array, zwracamy bez zmian
    elif isinstance(data, np.ndarray):
        return data

    else:
        raise TypeError(f"Nieobsługiwany format danych: {type(data)}")


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
        os.makedirs(self.saved_model_dir, exist_ok=True)
        self.history = {}

        if tf is not None and isinstance(model, tf.keras.Model) and use_gpu:
            logging.info("Trening z wykorzystaniem GPU/TPU (o ile jest dostępne).")

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
    ) -> Dict[str, List[float]]:
        try:
            splits = self.walk_forward_split(X, y, n_splits=n_splits)
            metrics = {"mse": [], "mae": []}

            for fold, (X_train, y_train, X_val, y_val) in enumerate(splits, start=1):
                logging.info("Rozpoczynam trening (fold %d / %d)", fold, n_splits)

                # Prepare data for the model (handling different data types)
                X_train_prepared = prepare_data_for_model(X_train)
                X_val_prepared = prepare_data_for_model(X_val)
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
            logging.info("Średnie MSE: %.4f, Średnie MAE: %.4f", avg_mse, avg_mae)
            self.save_model()
            return metrics

        except Exception as e:
            logging.error("Błąd podczas treningu modelu: %s", e)
            raise

    def save_model(self) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = os.path.join(
                self.saved_model_dir, f"{self.model_name}_{timestamp}"
            )

            if tf is not None and isinstance(self.model, tf.keras.Model):
                model_filename += ".h5"
                self.model.save(model_filename)
            else:
                model_filename += ".pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(self.model, f)

            logging.info("Model zapisany w: %s", model_filename)

        except Exception as e:
            logging.error("Błąd podczas zapisywania modelu: %s", e)
            raise


if __name__ == "__main__":
    try:
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
        X = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, size=500),
                "feature2": np.random.normal(0, 1, size=500),
            },
            index=dates,
        )
        y = (
            X["feature1"] * 1.5
            + X["feature2"] * (-2.0)
            + np.random.normal(0, 0.5, size=500)
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        trainer = ModelTrainer(
            model=model,
            model_name="RandomForest_Model",
            online_learning=True,
            early_stopping_params={"monitor": "val_loss", "patience": 5},
        )
        metrics = trainer.train(X, y, n_splits=5)
        logging.info("Trening zakończony. Wyniki walidacji: %s", metrics)
    except Exception as e:
        logging.error("Błąd w przykładowym użyciu ModelTrainer: %s", e)
        raise