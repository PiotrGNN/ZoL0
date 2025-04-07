"""
anomaly_detection.py - Moduł implementujący metody wykrywania anomalii rynkowych
"""

import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class AnomalyDetector:
    """
    Klasa implementująca różne metody wykrywania anomalii w danych rynkowych.

    Dostępne metody:
    - isolation_forest: Wykrywanie anomalii za pomocą algorytmu Isolation Forest
    - local_outlier_factor: Wykrywanie anomalii za pomocą Local Outlier Factor
    """

    def __init__(self, method="isolation_forest", threshold=3.0, contamination=0.05):
        """
        Inicjalizacja detektora anomalii.

        Args:
            method (str): Metoda wykrywania anomalii ('isolation_forest' lub 'local_outlier_factor')
            threshold (float): Próg uznania punktu za anomalię (dla metod bazujących na odległości)
            contamination (float): Oczekiwany odsetek anomalii w danych (0.0 - 0.5)
        """
        self.method = method.lower()
        self.threshold = threshold
        self.contamination = contamination
        self.model = None
        self.logger = logging.getLogger(__name__)

        # Inicjalizacja modelu na podstawie wybranej metody
        self._initialize_model()

        self.logger.info(f"Inicjalizacja detektora anomalii z metodą: {self.method}")

    def _initialize_model(self):
        """Inicjalizacja odpowiedniego modelu wykrywania anomalii."""
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True
            )
        else:
            raise ValueError(f"Nieznana metoda wykrywania anomalii: {self.method}")

    def fit(self, data):
        """
        Trenowanie modelu na danych.

        Args:
            data (np.ndarray): Dane treningowe
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.logger.info(f"Trenowanie modelu {self.method} na danych o wymiarze {data.shape}")
        self.model.fit(data)
        return self

    def predict(self, data):
        """
        Wykrywanie anomalii w danych.

        Args:
            data (np.ndarray): Dane do analizy

        Returns:
            np.ndarray: Tablica etykiet (-1 dla anomalii, 1 dla normalnych punktów)
        """
        if not hasattr(self.model, 'predict') or self.model is None:
            raise ValueError("Model nie został zainicjalizowany lub wytrenowany")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.logger.info(f"Wykrywanie anomalii w danych o wymiarze {data.shape}")
        return self.model.predict(data)

    def detect_anomalies(self, data, return_scores=False):
        """
        Wykrywanie anomalii w danych i zwracanie indeksów lub wyników.

        Args:
            data (np.ndarray): Dane do analizy
            return_scores (bool): Czy zwrócić wyniki anomalii zamiast indeksów

        Returns:
            np.ndarray: Indeksy wykrytych anomalii lub wyniki anomalii
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        predictions = self.predict(data)
        anomaly_indices = np.where(predictions == -1)[0]

        if return_scores:
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(data)
                return scores
            else:
                self.logger.warning("Model nie ma funkcji decision_function, zwracam indeksy")
                return anomaly_indices

        return anomaly_indices

    def info(self):
        """Wyświetla informacje o obecnym stanie detektora anomalii."""
        print(f"Anomaly Detector - Metoda: {self.method}")
        print(f"Parametry: Próg={self.threshold}, Zanieczyszczenie={self.contamination}")
        print("Status: Gotowy do użycia")

        self.logger.info(f"Wywołano informacje o detektorze anomalii (metoda: {self.method})")

# Alias dla kompatybilności wstecznej
AnomalyDetectionModel = AnomalyDetector

# Przykład użycia
if __name__ == "__main__":
    # Konfiguracja loggera dla bezpośredniego uruchomienia
    logging.basicConfig(level=logging.INFO)

    # Generowanie przykładowych danych
    np.random.seed(42)
    n_samples = 1000
    data = np.random.normal(100, 10, (n_samples, 2)) #Data as numpy array

    # Dodanie kilku anomalii
    anomaly_indices = np.random.choice(n_samples, 10, replace=False)
    data[anomaly_indices, 0] = np.random.normal(150, 20, 10)
    data[anomaly_indices, 1] = np.random.exponential(5000, 10)

    # Inicjalizacja i testowanie detektora
    detector = AnomalyDetectionModel(method="isolation_forest", contamination=0.01)
    detector.fit(data)
    anomalies = detector.detect_anomalies(data)

    print(f"Wykryto {len(anomalies)} anomalii")
    print("Indeksy anomalii:", anomalies.tolist())