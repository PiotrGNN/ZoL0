"""
anomaly_detection.py
-------------------
Moduł odpowiedzialny za wykrywanie anomalii w danych finansowych
z wykorzystaniem różnych algorytmów uczenia maszynowego.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Konfiguracja loggera
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Klasa implementująca różne algorytmy wykrywania anomalii w szeregach czasowych.
    Pozwala na identyfikację nietypowych wzorców cenowych lub wolumenowych.
    """

    def __init__(self, method: str = "isolation_forest", **kwargs):
        """
        Inicjalizacja detektora anomalii.

        Args:
            method (str): Metoda wykrywania anomalii ('isolation_forest', 'lof', 'zscore')
            **kwargs: Parametry specyficzne dla wybranego algorytmu
        """
        self.method = method
        self.model = None
        self.threshold = kwargs.get("threshold", 3.0)
        self.contamination = kwargs.get("contamination", 0.05)

        if method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=kwargs.get("random_state", 42),
                n_estimators=kwargs.get("n_estimators", 100)
            )
        elif method == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=kwargs.get("n_neighbors", 20),
                contamination=self.contamination,
                novelty=True
            )

        logger.info(f"Inicjalizacja detektora anomalii z metodą: {method}")

    def train(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> None:
        """
        Trenowanie modelu na podstawie historycznych danych.

        Args:
            data (pd.DataFrame): DataFrame z danymi historycznymi
            feature_columns (List[str], optional): Kolumny używane do wykrywania anomalii
        """
        # Wybór kolumn lub wszystkich kolumn, jeśli nie podano
        X = data[feature_columns] if feature_columns else data

        try:
            if self.method in ["isolation_forest", "lof"]:
                self.model.fit(X)
                logger.info(f"Model {self.method} został wytrenowany na danych o kształcie {X.shape}")
            elif self.method == "zscore":
                # W metodzie Z-score wystarczy zapisać średnie i odchylenia standardowe
                self.mean = X.mean()
                self.std = X.std()
                logger.info("Parametry Z-score zostały obliczone")
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu {self.method}: {e}")
            raise

    def detect(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.Series:
        """
        Wykrywanie anomalii w podanych danych.

        Args:
            data (pd.DataFrame): DataFrame do analizy
            feature_columns (List[str], optional): Kolumny używane do wykrywania anomalii

        Returns:
            pd.Series: Seria boolean określająca, które punkty są anomaliami
        """
        # Wybór kolumn lub wszystkich kolumn, jeśli nie podano
        X = data[feature_columns] if feature_columns else data

        try:
            if self.method == "isolation_forest":
                # -1 dla anomalii, 1 dla normalnych obserwacji
                predictions = self.model.predict(X)
                anomalies = pd.Series(predictions == -1, index=data.index)

            elif self.method == "lof":
                # -1 dla anomalii, 1 dla normalnych obserwacji
                predictions = self.model.predict(X)
                anomalies = pd.Series(predictions == -1, index=data.index)

            elif self.method == "zscore":
                # Obliczenie Z-score dla każdej kolumny
                z_scores = (X - self.mean) / self.std
                # Dane są anomaliami, jeśli jakikolwiek Z-score przekracza próg
                anomalies = (z_scores.abs() > self.threshold).any(axis=1)

            logger.info(f"Wykryto {anomalies.sum()} anomalii w danych o kształcie {X.shape}")
            return anomalies

        except Exception as e:
            logger.error(f"Błąd podczas wykrywania anomalii: {e}")
            raise

    def get_anomaly_scores(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.Series:
        """
        Oblicza wyniki anomalii dla każdej obserwacji.

        Args:
            data (pd.DataFrame): DataFrame do analizy
            feature_columns (List[str], optional): Kolumny używane do wykrywania anomalii

        Returns:
            pd.Series: Seria z wynikami anomalii
        """
        X = data[feature_columns] if feature_columns else data

        try:
            if self.method == "isolation_forest":
                # Niższe wartości oznaczają większe prawdopodobieństwo anomalii
                scores = self.model.decision_function(X)
                return pd.Series(scores, index=data.index)

            elif self.method == "lof":
                # Niższe wartości oznaczają większe prawdopodobieństwo anomalii
                scores = self.model.decision_function(X)
                return pd.Series(scores, index=data.index)

            elif self.method == "zscore":
                # Obliczenie maksymalnego Z-score dla każdej obserwacji
                z_scores = (X - self.mean) / self.std
                max_z_scores = z_scores.abs().max(axis=1)
                return max_z_scores

        except Exception as e:
            logger.error(f"Błąd podczas obliczania wyników anomalii: {e}")
            raise

    def info(self) -> None:
        """Wyświetla informacje o aktualnej konfiguracji detektora anomalii."""
        print(f"Anomaly Detector - Metoda: {self.method}")
        print(f"Parametry: Próg={self.threshold}, Zanieczyszczenie={self.contamination}")
        print("Status: Gotowy do użycia")
        logger.info(f"Wywołano informacje o detektorze anomalii (metoda: {self.method})")


# Przykład użycia
if __name__ == "__main__":
    # Konfiguracja loggera dla bezpośredniego uruchomienia
    logging.basicConfig(level=logging.INFO)

    # Generowanie przykładowych danych
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'price': np.random.normal(100, 10, n_samples),
        'volume': np.random.exponential(1000, n_samples)
    })

    # Dodanie kilku anomalii
    anomaly_indices = np.random.choice(n_samples, 10, replace=False)
    data.loc[anomaly_indices, 'price'] = np.random.normal(150, 20, 10)
    data.loc[anomaly_indices, 'volume'] = np.random.exponential(5000, 10)

    # Inicjalizacja i testowanie detektora
    detector = AnomalyDetector(method="isolation_forest", contamination=0.01)
    detector.train(data)
    anomalies = detector.detect(data)

    print(f"Wykryto {anomalies.sum()} anomalii")
    print("Indeksy anomalii:", anomalies[anomalies].index.tolist())