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
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Przykładowe importy modeli (należy upewnić się, że odpowiednie biblioteki są zainstalowane)
from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb  # odkomentuj, jeśli masz zainstalowany xgboost
# from tensorflow.keras.models import Sequential  # przykładowy model LSTM
# from prophet import Prophet  # lub from fbprophet import Prophet

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class ModelRecognizer:
    def __init__(self, metric_weights=None, cv_splits=5, random_state=42):
        """
        Inicjalizacja ModelRecognizer.
        
        Parameters:
            metric_weights (dict): Wagi dla metryk, np. {'mse': 0.4, 'mae': 0.3, 'rmse': 0.2, 'sharpe': 0.1}.
                                   Metryki błędu (mse, mae, rmse) – im niższe, tym lepiej;
                                   Sharpe Ratio – im wyższe, tym lepiej.
            cv_splits (int): Liczba podziałów w walidacji krzyżowej.
            random_state (int): Ziarno losowości.
        """
        # Domyślne wagi, jeśli nie podano
        self.metric_weights = metric_weights or {'mse': 0.4, 'mae': 0.3, 'rmse': 0.2, 'sharpe': 0.1}
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.models = {}      # słownik: {model_name: model_instance}
        self.results = {}     # słownik z wynikami oceny dla każdego modelu
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
        cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        for name, model in self.models.items():
            try:
                logging.info("Ocena modelu '%s'...", name)
                # Uzyskanie predykcji przez walidację krzyżową
                predictions = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
                mse = mean_squared_error(y, predictions)
                mae = mean_absolute_error(y, predictions)
                rmse = sqrt(mse)
                
                # Obliczenie przykładowego Sharpe Ratio:
                # Obliczamy zwroty jako różnice kolejnych prognoz (to uproszczenie)
                returns = np.diff(predictions)
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0.0

                self.results[name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'sharpe': sharpe
                }
                logging.info("Model '%s' - MSE: %.4f, MAE: %.4f, RMSE: %.4f, Sharpe: %.4f", 
                             name, mse, mae, rmse, sharpe)
            except Exception as e:
                logging.error("Błąd przy ocenie modelu '%s': %s", name, e)
                self.results[name] = None

        self._select_best_model()

    def _normalize_metrics(self):
        """
        Normalizuje metryki między modelami, aby umożliwić porównanie.
        Dla metryk błędu: im niższa wartość, tym lepsza – normalizujemy poprzez inwersję.
        Dla Sharpe Ratio: im wyższa wartość, tym lepsza.
        
        Returns:
            dict: Słownik z znormalizowanymi metrykami dla każdego modelu.
        """
        metric_keys = ['mse', 'mae', 'rmse', 'sharpe']
        # Inicjalizujemy struktury, pomijając modele, które nie zostały ocenione poprawnie.
        valid_results = {name: metrics for name, metrics in self.results.items() if metrics is not None}
        norm_metrics = {key: {} for key in metric_keys}
        
        for key in metric_keys:
            values = [metrics[key] for metrics in valid_results.values()]
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            for name, metrics in valid_results.items():
                if key in ['mse', 'mae', 'rmse']:
                    # Odwracamy skalę: im niższa wartość, tym lepsza
                    norm = (max_val - metrics[key]) / range_val
                else:  # dla Sharpe Ratio, im wyższa, tym lepsza
                    norm = (metrics[key] - min_val) / range_val
                norm_metrics[key][name] = norm
        return norm_metrics

    def _select_best_model(self):
        """
        Łączy znormalizowane metryki przy użyciu wag i wybiera model z najwyższym łącznym wynikiem.
        """
        norm_metrics = self._normalize_metrics()
        scores = {}
        for name in self.results.keys():
            if self.results[name] is None:
                continue
            score = (
                self.metric_weights['mse'] * norm_metrics['mse'][name] +
                self.metric_weights['mae'] * norm_metrics['mae'][name] +
                self.metric_weights['rmse'] * norm_metrics['rmse'][name] +
                self.metric_weights['sharpe'] * norm_metrics['sharpe'][name]
            )
            scores[name] = score
            logging.info("Łączny score dla modelu '%s': %.4f", name, score)
        if scores:
            self.best_model_name = max(scores, key=scores.get)
            self.best_model = self.models[self.best_model_name]
            logging.info("Najlepszy model to '%s' z score: %.4f", self.best_model_name, scores[self.best_model_name])
        else:
            logging.warning("Brak ocenionych modeli do wyboru najlepszego.")

    def get_best_model(self):
        """
        Zwraca najlepszy model wybrany na podstawie przeprowadzonej oceny.
        
        Returns:
            Model: Najlepszy model lub None, jeśli nie został wybrany.
        """
        return self.best_model

    def get_results(self):
        """
        Zwraca wyniki oceny modeli.
        
        Returns:
            dict: Słownik wyników oceny.
        """
        return self.results

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane (np. historyczne ceny)
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 500),
            'feature2': np.random.uniform(0, 1, 500)
        })
        # Przykładowa zmienna docelowa: np. zmiana ceny
        y = X['feature1'] * 2 + X['feature2'] * (-1) + np.random.normal(0, 0.1, 500)

        recognizer = ModelRecognizer(cv_splits=5)
        
        # Dodaj przykładowe modele
        recognizer.add_model('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
        # recognizer.add_model('XGBoost', xgb.XGBRegressor(n_estimators=100, random_state=42))
        # Inne modele, takie jak LSTM lub Prophet, można dodać według potrzeb

        # Ocena modeli
        recognizer.evaluate_models(X, y)
        
        best_model = recognizer.get_best_model()
        logging.info("Wybrany najlepszy model: %s", recognizer.best_model_name)
        logging.info("Wyniki oceny modeli: %s", recognizer.get_results())
    except Exception as e:
        logging.error("Błąd w przykładowym użyciu ModelRecognizer: %s", e)
        raise
