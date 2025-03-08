"""
model_recognition.py
--------------------
Moduł porównuje wyniki kilku modeli ML (np. Random Forest, XGBoost, LSTM, Prophet, itp.)
i automatycznie wybiera najlepszy model do przewidywania ruchów cen.

Funkcjonalności:
- Ocena modeli przy użyciu różnych metryk (MSE, MAE, RMSE, Sharpe Ratio).
- Automatyczna walidacja krzyżowa (cross-validation) na danych historycznych.
- Łączenie metryk przy użyciu wag odzwierciedlających priorytety (zysk, stabilność, drawdown).
- System logowania wyników i możliwość śledzenia zmian w czasie.
- Obsługa dużych zbiorów danych i możliwość równoległego trenowania modeli.
- Skalowalność dla różnych rozmiarów portfeli.
"""

import logging
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class ModelRecognizer:
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
            "sharpe": 0.1,
        }
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def add_model(self, name, model):
        """
        Dodaje model do zestawu.

        Parameters:
            name (str): Nazwa modelu.
            model: Instancja modelu.
        """
        self.models[name] = model
        logging.info("Model '%s' został dodany.", name)

    def evaluate_models(self, X, y):
        """
        Ocena wszystkich dodanych modeli przy użyciu walidacji krzyżowej i obliczanie metryk.

        Parameters:
            X (pd.DataFrame lub np.array): Dane wejściowe.
            y (pd.Series lub np.array): Wartości docelowe.
        """
        cv = KFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )
        for name, model in self.models.items():
            try:
                logging.info("Ocena modelu '%s'...", name)
                predictions = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
                mse = mean_squared_error(y, predictions)
                mae = mean_absolute_error(y, predictions)
                rmse = sqrt(mse)
                returns = np.diff(predictions)
                sharpe = (
                    np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0.0
                )

                self.results[name] = {
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "sharpe": sharpe,
                }
                logging.info(
                    "Model '%s' - MSE: %.4f, MAE: %.4f, RMSE: %.4f, Sharpe: %.4f",
                    name,
                    mse,
                    mae,
                    rmse,
                    sharpe,
                )
            except Exception as e:
                logging.error("Błąd przy ocenie modelu '%s': %s", name, e)
                self.results[name] = None

        self._select_best_model()

    def _normalize_metrics(self):
        """
        Normalizuje metryki między modelami, aby umożliwić porównanie.
        """
        metric_keys = ["mse", "mae", "rmse", "sharpe"]
        valid_results = {
            name: metrics
            for name, metrics in self.results.items()
            if metrics is not None
        }
        norm_metrics = {key: {} for key in metric_keys}

        for key in metric_keys:
            values = [metrics[key] for metrics in valid_results.values()]
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            for name, metrics in valid_results.items():
                norm_metrics[key][name] = (
                    (max_val - metrics[key]) / range_val
                    if key in ["mse", "mae", "rmse"]
                    else (metrics[key] - min_val) / range_val
                )
        return norm_metrics

    def _select_best_model(self):
        """
        Łączy znormalizowane metryki przy użyciu wag i wybiera model z najwyższym łącznym wynikiem.
        """
        norm_metrics = self._normalize_metrics()
        scores = self.calculate_scores(norm_metrics)

        if scores:
            self.best_model_name = max(scores, key=scores.get)
            self.best_model = self.models[self.best_model_name]
            logging.info(
                "Najlepszy model to '%s' z score: %.4f",
                self.best_model_name,
                scores[self.best_model_name],
            )
        else:
            logging.warning("Brak ocenionych modeli do wyboru najlepszego.")

    def calculate_scores(self, norm_metrics):
        """
        Oblicza łączny score dla każdego modelu.
        """
        scores = {}
        for name in norm_metrics["mse"]:
            score = sum(
                self.metric_weights[metric] * norm_metrics[metric][name]
                for metric in self.metric_weights
            )
            scores[name] = score
            logging.info("Łączny score dla modelu '%s': %.4f", name, score)
        return scores

    def get_best_model(self):
        """Zwraca najlepszy model."""
        return self.best_model

    def get_results(self):
        """Zwraca wyniki oceny modeli."""
        return self.results


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 1, 500),
                "feature2": np.random.uniform(0, 1, 500),
            }
        )
        y = X["feature1"] * 2 + X["feature2"] * (-1) + np.random.normal(0, 0.1, 500)

        recognizer = ModelRecognizer(cv_splits=5)

        recognizer.add_model(
            "RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)
        )

        recognizer.evaluate_models(X, y)

        logging.info("Wybrany najlepszy model: %s", recognizer.best_model_name)
        logging.info("Wyniki oceny modeli: %s", recognizer.get_results())
    except Exception as e:
        logging.error("Błąd w przykładowym użyciu ModelRecognizer: %s", e)
        raise
