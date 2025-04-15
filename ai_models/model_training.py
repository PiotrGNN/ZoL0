"""
model_training.py
-----------------
Moduł do trenowania modeli AI na danych finansowych.

Funkcjonalności:
- Pipeline dzielący dane na zbiór treningowy, walidacyjny i testowy z wykorzystaniem walk-forward validation.
- Automatyczna optymalizacja hiperparametrów (może być integrowana z modułem model_tuner.py).
- Obsługa trybu online learning (aktualizacja modelu przy napływie nowych danych).
- Mechanizmy early stopping oraz monitorowanie metryk w czasie treningu (dla modeli Keras/TensorFlow).
- Skalowalność i wsparcie dla trenowania na GPU/TPU (w przypadku modeli TensorFlow/Keras).
- Odporny na błędy (logowanie wyjątków, automatyczna próba wznowienia) oraz zapisywanie wyników do folderu saved_models.
"""

import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Dla modeli Keras (jeśli używasz TensorFlow)
try:
    # # # # # # import tensorflow as tf  # Zakomentowano - opcjonalny pakiet  # Zakomentowano - opcjonalny pakiet  # Zakomentowano - opcjonalny pakiet  # Zakomentowano - opcjonalny pakiet
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    tf = None

# Konfiguracja logowania (opcjonalnie może być zastąpione przez setup_logging w main.py)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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
        """
        Inicjalizacja trenera modelu.

        Parameters:
            model: Instancja modelu (może być scikit-learn lub Keras/TensorFlow).
            model_name (str): Nazwa modelu, używana do zapisywania wyników.
            saved_model_dir (str): Folder, w którym zapisywane są wytrenowane modele.
            online_learning (bool): Jeśli True, model będzie aktualizowany na bieżąco (partial_fit).
            use_gpu (bool): Jeśli True, trenuj model z wykorzystaniem GPU/TPU (dotyczy głównie Keras).
            early_stopping_params (dict): Parametry dla early stopping (np. {'monitor': 'val_loss', 'patience': 5}).
        """
        self.model = model
        self.model_name = model_name
        self.saved_model_dir = saved_model_dir
        self.online_learning = online_learning
        self.use_gpu = use_gpu
        self.early_stopping_params = early_stopping_params or {}
        os.makedirs(self.saved_model_dir, exist_ok=True)

        # Historia treningu (dla modeli Keras)
        self.history: Dict[str, List[float]] = {}

        # Jeśli używamy GPU/TPU w TensorFlow
        if tf is not None and isinstance(model, tf.keras.Model) and use_gpu:
            logging.info("Trening z wykorzystaniem GPU/TPU (o ile jest dostępne).")
            # Możesz tu włączyć strategię tf.distribute.MirroredStrategy() itp.

    def walk_forward_split(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> List[tuple]:
        """
        Dzieli dane przy użyciu walk-forward validation.

        Returns:
            List[tuple]: Lista krotek (X_train, y_train, X_val, y_val).
        """
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
        """
        Trenuje model przy użyciu walk-forward validation.

        Parameters:
            X (pd.DataFrame): Dane wejściowe (cechy).
            y (pd.Series): Wartości docelowe (target).
            n_splits (int): Liczba podziałów walk-forward.
            epochs (int): Liczba epok (dla modeli Keras).
            batch_size (int): Rozmiar batcha (dla modeli Keras).

        Returns:
            Dict[str, List[float]]: Metryki walidacyjne z poszczególnych foldów (np. 'mse', 'mae').
        """
        try:
            splits = self.walk_forward_split(X, y, n_splits=n_splits)
            metrics = {"mse": [], "mae": []}

            for fold, (X_train, y_train, X_val, y_val) in enumerate(splits, start=1):
                logging.info("Rozpoczynam trening (fold %d / %d)", fold, n_splits)

                # Jeśli model to Keras (tf.keras.Model)
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
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1,
                    )
                    # Zachowujemy historię treningu
                    self.history = history.history

                    # Załaduj najlepszy model z checkpointu
                    self.model = tf.keras.models.load_model(checkpoint_path)
                    predictions = self.model.predict(X_val)

                else:
                    # Dla modeli scikit-learn
                    self.model.fit(X_train, y_train)
                    predictions = self.model.predict(X_val)

                # Obliczamy metryki
                mse = mean_squared_error(y_val, predictions)
                mae = mean_absolute_error(y_val, predictions)
                metrics["mse"].append(mse)
                metrics["mae"].append(mae)

                logging.info("Fold %d - MSE: %.4f, MAE: %.4f", fold, mse, mae)

                # Jeśli online learning i model wspiera partial_fit
                if self.online_learning and hasattr(self.model, "partial_fit"):
                    logging.info(
                        "Aktualizacja modelu metodą partial_fit (online learning)."
                    )
                    # Przekazujemy dane walidacyjne do partial_fit
                    self.model.partial_fit(X_val, y_val)

            # Wyliczamy średnie metryki z wszystkich foldów
            avg_mse = float(np.mean(metrics["mse"]))
            avg_mae = float(np.mean(metrics["mae"]))
            logging.info("Średnie MSE: %.4f, Średnie MAE: %.4f", avg_mse, avg_mae)

            # Zapisujemy ostateczny model
            self.save_model()
            return metrics

        except Exception as e:
            logging.error("Błąd podczas treningu modelu: %s", e)
            raise

    def save_model(self) -> None:
        """
        Zapisuje wytrenowany model do folderu saved_models.
        Dla modeli scikit-learn używa pickle, dla Keras zapisuje w formacie .h5.
        """
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


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Generowanie przykładowych danych (np. dane finansowe)
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

        # Przykład użycia z modelem scikit-learn (np. RandomForestRegressor)
        from sklearn.ensemble import RandomForestRegressor

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
"""
model_training.py
----------------
Moduł do trenowania modeli AI w systemie.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

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
        logger.info("Konwersja słownika danych na DataFrame")
        df = pd.DataFrame(data)
        return df.values
        
    # Jeśli to DataFrame, konwertujemy na numpy array
    elif isinstance(data, pd.DataFrame):
        logger.info("Konwersja DataFrame na numpy array")
        return data.values
        
    # Jeśli to lista, konwertujemy na numpy array
    elif isinstance(data, list):
        logger.info("Konwersja listy na numpy array")
        return np.array(data)
        
    # Jeśli to już numpy array, zwracamy bez zmian
    elif isinstance(data, np.ndarray):
        return data
        
    else:
        raise TypeError(f"Nieobsługiwany format danych: {type(data)}")

def reshape_for_lstm(data: np.ndarray, time_steps: int = 1, features: int = None) -> np.ndarray:
    """
    Przekształca dane do formatu wymaganego przez modele LSTM: [samples, time_steps, features]
    
    Args:
        data: Dane wejściowe jako numpy array
        time_steps: Liczba kroków czasowych (sequence length)
        features: Liczba cech (jeśli None, zostanie wywnioskowana z danych)
        
    Returns:
        np.ndarray: Dane w formacie odpowiednim dla LSTM
    """
    if features is None:
        if len(data.shape) > 1:
            features = data.shape[1]  # liczba kolumn
        else:
            features = 1
            
    # Jeśli dane są jednowymiarowe, przekształć na 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        
    # Jeśli time_steps > 1, utwórz sekwencje
    if time_steps > 1:
        samples = len(data) - time_steps + 1
        new_data = np.zeros((samples, time_steps, features))
        
        for i in range(samples):
            new_data[i] = data[i:i+time_steps].reshape(time_steps, features)
        
        return new_data
    else:
        # Dodaj wymiar time_steps
        return data.reshape(data.shape[0], 1, features)

class ModelTrainer:
    """
    Klasa do trenowania modeli AI/ML
    """
    
    def __init__(self, model_type: str = "keras"):
        """
        Inicjalizuje trener modeli.
        
        Args:
            model_type: Typ modelu do trenowania (keras, sklearn, xgboost)
        """
        self.model_type = model_type
        self.logger = logging.getLogger(f"ModelTrainer.{model_type}")
        self.logger.info(f"Inicjalizacja trenera modeli typu {model_type}")
