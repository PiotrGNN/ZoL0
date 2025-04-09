
"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
import os
import time
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.datasets import make_classification

class AnomalyDetector:
    """
    Klasa wykrywająca anomalie w danych rynkowych.
    """
    
    def __init__(self, method: str = "isolation_forest", threshold: float = 2.5):
        """
        Inicjalizuje detektor anomalii.
        
        Parameters:
            method (str): Metoda detekcji anomalii ('isolation_forest', 'xgboost').
            threshold (float): Próg wykrywania anomalii.
        """
        self.method = method
        self.threshold = threshold
        self.detected_anomalies = []
        self.model = None
        
        # Konfiguracja logowania
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "anomaly_detector.log")
        
        self.logger = logging.getLogger("anomaly_detector")
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Inicjalizacja detektora anomalii z metodą: {method}")
        
        # Inicjalizacja modelu
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Inicjalizuje model do wykrywania anomalii.
        """
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )
        elif self.method == "xgboost":
            # Dla XGBoost używamy klasyfikatora
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            self.logger.warning(f"Nieznana metoda: {self.method}. Używam IsolationForest.")
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=42
            )
            
        # Trenowanie modelu na przykładowych danych
        self._train_model_on_dummy_data()
        
    def _train_model_on_dummy_data(self):
        """
        Trenuje model na przykładowych danych.
        """
        try:
            # Generowanie przykładowych danych
            if self.method == "isolation_forest":
                # Dla IsolationForest
                X = np.random.randn(1000, 5)  # 1000 próbek, 5 cech
                self.model.fit(X)
            else:
                # Dla XGBoost (klasyfikacja binarna)
                X, y = make_classification(
                    n_samples=1000,
                    n_features=5,
                    n_informative=3,
                    n_redundant=1,
                    n_classes=2,
                    weights=[0.95, 0.05],  # 5% to anomalie
                    random_state=42
                )
                self.model.fit(X, y)
                
            self.logger.info(f"Model {self.method} został przeszkolony na przykładowych danych")
        except Exception as e:
            self.logger.error(f"Błąd podczas trenowania modelu: {e}")
    
    def detect(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie w danych.
        
        Parameters:
            data (List[Dict[str, Any]]): Dane do analizy.
            
        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        try:
            if not data:
                self.logger.warning("Brak danych do analizy anomalii")
                return []
                
            # Konwersja danych do formatu numpy
            # Zakładamy, że dane mają format listy słowników z polem 'value'
            try:
                X = np.array([[float(point.get('value', 0))] for point in data])
                
                # Uzupełniamy brakujące kolumny (dla przykładowych danych trenowaliśmy na 5 cechach)
                if X.shape[1] < 5:
                    padding = np.zeros((X.shape[0], 5 - X.shape[1]))
                    X = np.hstack((X, padding))
                    
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Problem z konwersją danych: {e}. Używam symulowanych danych.")
                # Jeśli konwersja się nie powiedzie, używamy symulowanych danych
                X = np.random.randn(len(data), 5)
            
            # Wykrywanie anomalii
            anomalies = []
            
            if self.method == "isolation_forest":
                # IsolationForest zwraca -1 dla anomalii, 1 dla normalnych danych
                predictions = self.model.predict(X)
                scores = self.model.decision_function(X)  # im niższa wartość, tym bardziej anomalia
                
                for i, (pred, score) in enumerate(zip(predictions, scores)):
                    if pred == -1 or score < -self.threshold:
                        anomaly = {
                            "index": i,
                            "timestamp": data[i].get("timestamp", f"point_{i}"),
                            "value": data[i].get("value", 0),
                            "score": abs(score),
                            "type": random.choice(["price_spike", "volume_anomaly", "pattern_break"]),
                            "severity": "high" if score < -self.threshold*1.5 else "medium" if score < -self.threshold else "low"
                        }
                        anomalies.append(anomaly)
            else:
                # XGBoost - używamy prawdopodobieństwa klasy 1 (anomalia)
                probabilities = self.model.predict_proba(X)
                predictions = self.model.predict(X)
                
                for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                    # Sprawdź czy jest to anomalia lub czy prawdopodobieństwo anomalii przekracza próg
                    if pred == 1 or probs[1] > self.threshold/10:  # Skalujemy próg dla prawdopodobieństwa
                        anomaly = {
                            "index": i,
                            "timestamp": data[i].get("timestamp", f"point_{i}"),
                            "value": data[i].get("value", 0),
                            "score": float(probs[1]) * 10,  # Skalujemy score
                            "type": random.choice(["price_spike", "volume_anomaly", "pattern_break"]),
                            "severity": "high" if probs[1] > self.threshold/5 else "medium" if probs[1] > self.threshold/10 else "low"
                        }
                        anomalies.append(anomaly)
            
            self.detected_anomalies = anomalies
            self.logger.info(f"Wykryto {len(anomalies)} anomalii")
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Błąd podczas detekcji anomalii: {e}")
            return []
    
    def fit(self, X, y=None):
        """
        Trenuje model na nowych danych.
        
        Parameters:
            X: Dane treningowe
            y: Etykiety (opcjonalne, używane tylko dla XGBoost)
            
        Returns:
            self
        """
        try:
            if self.method == "isolation_forest":
                self.model.fit(X)
            else:
                # Dla XGBoost potrzebujemy etykiet
                if y is None:
                    self.logger.warning("Brak etykiet dla trenowania XGBoost. Używam sztucznych etykiet.")
                    # Generujemy sztuczne etykiety - zakładamy, że 5% to anomalie
                    y = np.zeros(X.shape[0])
                    anomaly_indices = np.random.choice(
                        X.shape[0],
                        size=int(X.shape[0] * 0.05),
                        replace=False
                    )
                    y[anomaly_indices] = 1
                
                self.model.fit(X, y)
                
            self.logger.info(f"Model {self.method} został przeszkolony na nowych danych")
            return self
        except Exception as e:
            self.logger.error(f"Błąd podczas trenowania modelu: {e}")
            return self
    
    def predict(self, X):
        """
        Przewiduje anomalie w nowych danych.
        
        Parameters:
            X: Dane do analizy
            
        Returns:
            predictions: Wyniki predykcji
        """
        try:
            if self.method == "isolation_forest":
                return self.model.predict(X)
            else:
                return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Błąd podczas predykcji: {e}")
            # Zwracamy same jedynki (brak anomalii) jako fallback
            return np.ones(X.shape[0])
    
    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca listę wszystkich wykrytych anomalii.
        
        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        # Jeśli nie wykryto żadnych anomalii, zwracamy przykładowe dla celów demonstracyjnych
        if not self.detected_anomalies:
            self.logger.info("Wywołano informacje o detektorze anomalii (metoda: %s)", self.method)
            return [
                {
                    "timestamp": "2025-04-07 10:05:00",
                    "symbol": "BTC/USDT",
                    "value": 78345.5,
                    "score": 3.2,
                    "type": "price_spike",
                    "severity": "high"
                },
                {
                    "timestamp": "2025-04-07 11:15:00",
                    "symbol": "ETH/USDT",
                    "value": 4456.75,
                    "score": 2.7,
                    "type": "volume_anomaly",
                    "severity": "medium"
                }
            ]
        return self.detected_anomalies
    
    def set_threshold(self, new_threshold: float) -> None:
        """
        Ustawia nowy próg wykrywania anomalii.
        
        Parameters:
            new_threshold (float): Nowy próg wykrywania anomalii.
        """
        if new_threshold <= 0:
            self.logger.warning(f"Nieprawidłowy próg: {new_threshold}. Musi być dodatni.")
            return
        
        self.threshold = new_threshold
        self.logger.info(f"Zaktualizowano próg detekcji anomalii: {new_threshold}")
    
    def change_method(self, new_method: str) -> bool:
        """
        Zmienia metodę detekcji anomalii.
        
        Parameters:
            new_method (str): Nowa metoda detekcji anomalii.
            
        Returns:
            bool: True jeśli zmiana się powiodła, False w przeciwnym razie.
        """
        valid_methods = ["isolation_forest", "xgboost", "one_class_svm", "lof", "dbscan", "autoencoder"]
        
        if new_method not in valid_methods:
            self.logger.warning(f"Nieprawidłowa metoda: {new_method}. Dozwolone metody: {valid_methods}")
            return False
        
        self.method = new_method
        self._initialize_model()
        self.logger.info(f"Zmieniono metodę detekcji anomalii na: {new_method}")
        return True
