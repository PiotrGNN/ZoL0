
"""
model_recognition.py
--------------------
Moduł porównuje wyniki kilku modeli ML (np. Random Forest, XGBoost, LSTM) 
i automatycznie wybiera najlepszy model do przewidywania ruchów cen.

Funkcjonalności:
- Ocena modeli przy użyciu różnych metryk
- Automatyczna walidacja krzyżowa na danych historycznych
- Łączenie metryk przy użyciu wag odzwierciedlających priorytety
"""

import logging
from math import sqrt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class ModelRecognizer:
    """
    Klasa do porównywania i wybierania najlepszych modeli predykcyjnych.
    """
    
    def __init__(self, metric_weights=None, cv_splits=5, random_state=42):
        """
        Inicjalizacja ModelRecognizer.

        Parameters:
            metric_weights (dict): Wagi dla metryk, np. {'mse': 0.4, 'mae': 0.3, 'rmse': 0.2, 'sharpe': 0.1}.
            cv_splits (int): Liczba podziałów w walidacji krzyżowej.
            random_state (int): Ziarno losowości.
        """
        self.metric_weights = metric_weights or {
            "mse": 0.4,
            "mae": 0.3,
            "rmse": 0.2,
            "r2": 0.1,
        }
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
        # Inicjalizacja domyślnych modeli
        self._init_default_models()
        
    def _init_default_models(self):
        """
        Inicjalizuje domyślne modele do testowania.
        """
        # Dodajemy model RandomForest
        self.add_model("RandomForest", RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state
        ))
        
        # Dodajemy model XGBoost
        self.add_model("XGBoost", xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=self.random_state
        ))
        
        # Dodajemy model Keras LSTM jeśli TensorFlow jest dostępny
        if TF_AVAILABLE:
            self.add_model("LSTM", self._create_lstm_model)
        
    def _create_lstm_model(self, input_shape=(10, 1)):
        """
        Tworzy i zwraca model LSTM.
        
        Parameters:
            input_shape (tuple): Kształt danych wejściowych (okresy, cechy)
            
        Returns:
            model: Model Keras LSTM
        """
        try:
            # Ustawienie globalnych ustawień dla TensorFlow
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            
            # Tworzenie modelu LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mean_squared_error'
            )
            
            return model
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia modelu LSTM: {e}")
            return None

    def add_model(self, name, model):
        """
        Dodaje model do zestawu.

        Parameters:
            name (str): Nazwa modelu.
            model: Instancja modelu lub funkcja tworząca model.
        """
        self.models[name] = model
        logger.info("Model '%s' został dodany.", name)

    def evaluate_models(self, X, y, n_periods=None):
        """
        Ocena wszystkich dodanych modeli i obliczanie metryk.

        Parameters:
            X (pd.DataFrame lub np.array): Dane wejściowe.
            y (pd.Series lub np.array): Wartości docelowe.
            n_periods (int): Liczba okresów dla modeli sekwencyjnych (np. LSTM)
        """
        # Dla modeli nieseqwencyjnych
        X_scaled = self.scaler.fit_transform(X)
        kf = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            try:
                logger.info("Ocena modelu '%s'...", name)
                
                # Specjalne przetwarzanie dla modeli Keras
                if name == "LSTM" and TF_AVAILABLE:
                    self._evaluate_lstm_model(name, X_scaled, y, n_periods or 10)
                else:
                    self._evaluate_standard_model(name, model, X_scaled, y, kf)
                    
            except Exception as e:
                logger.error("Błąd przy ocenie modelu '%s': %s", name, e)
                self.results[name] = None

        self._select_best_model()
        
    def _evaluate_standard_model(self, name, model, X, y, kf):
        """
        Ocenia standardowy model ML z wykorzystaniem walidacji krzyżowej.
        
        Parameters:
            name (str): Nazwa modelu
            model: Model ML
            X: Dane wejściowe
            y: Wartości docelowe
            kf: Obiekt KFold
        """
        mse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Trenowanie modelu
            model.fit(X_train, y_train)
            
            # Predykcje
            y_pred = model.predict(X_test)
            
            # Obliczanie metryk
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)
        
        # Średnie metryki
        avg_mse = np.mean(mse_scores)
        avg_mae = np.mean(mae_scores)
        avg_rmse = sqrt(avg_mse)
        avg_r2 = np.mean(r2_scores)
        
        self.results[name] = {
            "mse": avg_mse,
            "mae": avg_mae,
            "rmse": avg_rmse,
            "r2": avg_r2
        }
        
        logger.info(
            "Model '%s' - MSE: %.4f, MAE: %.4f, RMSE: %.4f, R2: %.4f",
            name, avg_mse, avg_mae, avg_rmse, avg_r2
        )
        
    def _evaluate_lstm_model(self, name, X, y, n_periods):
        """
        Ocenia model LSTM.
        
        Parameters:
            name (str): Nazwa modelu
            X: Dane wejściowe
            y: Wartości docelowe
            n_periods: Liczba okresów dla sekwencji
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow nie jest dostępny. Nie można ocenić modelu LSTM.")
            self.results[name] = None
            return
            
        try:
            # Przygotowanie danych w formacie sekwencji
            X_seq, y_seq = self._create_sequences(X, y, n_periods)
            
            # Podział danych na zbiory treningowy i testowy
            train_size = int(len(X_seq) * 0.8)
            X_train, X_test = X_seq[:train_size], X_seq[train_size:]
            y_train, y_test = y_seq[:train_size], y_seq[train_size:]
            
            # Tworzenie i trening modelu
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self._create_lstm_model(input_shape)
            
            # Uproszczony trening (bez callbacks)
            model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                verbose=0
            )
            
            # Predykcje
            y_pred = model.predict(X_test, verbose=0)
            
            # Obliczanie metryk
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            self.results[name] = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }
            
            logger.info(
                "Model '%s' - MSE: %.4f, MAE: %.4f, RMSE: %.4f, R2: %.4f",
                name, mse, mae, rmse, r2
            )
            
            # Zapisujemy model jako właściwość klasy
            self.models[name] = model
            
        except Exception as e:
            logger.error(f"Błąd przy ocenie modelu LSTM: {e}")
            self.results[name] = None
            
    def _create_sequences(self, X, y, n_periods):
        """
        Tworzy sekwencje dla modeli rekurencyjnych.
        
        Parameters:
            X: Dane wejściowe
            y: Wartości docelowe
            n_periods: Liczba okresów dla sekwencji
            
        Returns:
            X_seq, y_seq: Dane sekwencyjne
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - n_periods):
            X_seq.append(X[i:i+n_periods])
            y_seq.append(y[i+n_periods])
            
        return np.array(X_seq), np.array(y_seq)

    def _normalize_metrics(self):
        """
        Normalizuje metryki między modelami, aby umożliwić porównanie.
        """
        metric_keys = ["mse", "mae", "rmse", "r2"]
        valid_results = {
            name: metrics
            for name, metrics in self.results.items()
            if metrics is not None
        }
        
        if not valid_results:
            logger.warning("Brak ważnych wyników do normalizacji.")
            return {}
            
        norm_metrics = {key: {} for key in metric_keys}

        for key in metric_keys:
            values = [metrics[key] for metrics in valid_results.values()]
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            
            for name, metrics in valid_results.items():
                # Dla r2 wyższa wartość jest lepsza, dla innych niższa
                if key == "r2":
                    norm_metrics[key][name] = (metrics[key] - min_val) / range_val
                else:
                    norm_metrics[key][name] = (max_val - metrics[key]) / range_val
                    
        return norm_metrics

    def _select_best_model(self):
        """
        Łączy znormalizowane metryki przy użyciu wag i wybiera model z najwyższym łącznym wynikiem.
        """
        norm_metrics = self._normalize_metrics()
        
        if not norm_metrics:
            logger.warning("Brak metryk do wyboru najlepszego modelu.")
            return
            
        scores = self.calculate_scores(norm_metrics)

        if scores:
            self.best_model_name = max(scores, key=scores.get)
            self.best_model = self.models[self.best_model_name]
            logger.info(
                "Najlepszy model to '%s' z score: %.4f",
                self.best_model_name,
                scores[self.best_model_name],
            )
        else:
            logger.warning("Brak ocenionych modeli do wyboru najlepszego.")

    def calculate_scores(self, norm_metrics):
        """
        Oblicza łączny score dla każdego modelu.
        """
        scores = {}
        for name in list(norm_metrics.get("mse", {}).keys()):
            score = sum(
                self.metric_weights.get(metric, 0) * norm_metrics.get(metric, {}).get(name, 0)
                for metric in self.metric_weights
            )
            scores[name] = score
            logger.info("Łączny score dla modelu '%s': %.4f", name, score)
        return scores
        
    def fit(self, X, y, **kwargs):
        """
        Trenuje najlepszy model.
        
        Parameters:
            X: Dane wejściowe
            y: Wartości docelowe
            kwargs: Dodatkowe parametry dla treningu
            
        Returns:
            self: Instancja klasy
        """
        if self.best_model is None:
            logger.warning("Brak najlepszego modelu. Najpierw wywołaj evaluate_models().")
            return self
            
        try:
            X_scaled = self.scaler.transform(X)
            
            if self.best_model_name == "LSTM" and TF_AVAILABLE:
                # Dla LSTM potrzebujemy sekwencji
                n_periods = kwargs.get("n_periods", 10)
                X_seq, y_seq = self._create_sequences(X_scaled, y, n_periods)
                
                # Parametry treningu
                epochs = kwargs.get("epochs", 20)
                batch_size = kwargs.get("batch_size", 32)
                
                self.best_model.fit(
                    X_seq, y_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
            else:
                # Standardowe modele
                self.best_model.fit(X_scaled, y)
                
            logger.info(f"Model {self.best_model_name} został wytrenowany.")
            return self
            
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu {self.best_model_name}: {e}")
            return self

    def predict(self, X, **kwargs):
        """
        Dokonuje predykcji przy użyciu najlepszego modelu.
        
        Parameters:
            X: Dane wejściowe
            kwargs: Dodatkowe parametry
            
        Returns:
            predictions: Wyniki predykcji
        """
        if self.best_model is None:
            logger.warning("Brak najlepszego modelu. Najpierw wywołaj evaluate_models().")
            return np.zeros(len(X))
            
        try:
            X_scaled = self.scaler.transform(X)
            
            if self.best_model_name == "LSTM" and TF_AVAILABLE:
                # Dla LSTM potrzebujemy sekwencji
                n_periods = kwargs.get("n_periods", 10)
                
                # Jeśli mamy tylko jedną próbkę
                if len(X_scaled.shape) == 1 or X_scaled.shape[0] == 1:
                    # Tworzymy sztuczną sekwencję
                    X_seq = np.array([X_scaled] * n_periods).reshape(1, n_periods, X_scaled.shape[-1])
                else:
                    # Zakładamy, że X zawiera już wszystkie potrzebne dane do sekwencji
                    # i przekształcamy je do 3D (próbki, okresy, cechy)
                    X_seq = X_scaled.reshape(-1, n_periods, X_scaled.shape[-1])
                    
                return self.best_model.predict(X_seq, verbose=0).flatten()
            else:
                # Standardowe modele
                return self.best_model.predict(X_scaled)
                
        except Exception as e:
            logger.error(f"Błąd podczas predykcji z modelem {self.best_model_name}: {e}")
            return np.zeros(len(X))

    def get_best_model(self):
        """Zwraca najlepszy model."""
        return self.best_model

    def get_results(self):
        """Zwraca wyniki oceny modeli."""
        return self.results


# Przykładowe użycie
if __name__ == "__main__":
    try:
        # Generowanie przykładowych danych
        np.random.seed(42)
        X = np.random.rand(1000, 5)
        y = 2*X[:, 0] - 3*X[:, 1] + 0.5*X[:, 2] + np.random.normal(0, 0.1, 1000)
        
        # Inicjalizacja i ocena modeli
        recognizer = ModelRecognizer()
        recognizer.evaluate_models(X, y)
        
        # Trenowanie najlepszego modelu
        recognizer.fit(X, y)
        
        # Predykcja
        y_pred = recognizer.predict(X[:5])
        print("Predykcje:", y_pred)
        print("Rzeczywiste wartości:", y[:5])
    except Exception as e:
        logger.error(f"Błąd w przykładowym użyciu: {e}")
