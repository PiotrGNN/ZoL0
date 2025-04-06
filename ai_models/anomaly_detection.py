"""
anomaly_detection.py
---------------------
Moduł do wykrywania anomalii na rynkach finansowych.

Główne funkcje:
- Wykrywanie nietypowych zachowań cenowych
- Identyfikacja potencjalnych manipulacji rynkowych
- Ostrzeganie o anomaliach poprzez system powiadomień
"""

import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Konfiguracja logowania
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Klasa implementująca różne metody wykrywania anomalii w danych finansowych.
    """

    def __init__(self, contamination=0.05, method="isolation_forest"):
        """
        Inicjalizacja detektora anomalii.

        Args:
            contamination (float): Szacunkowa proporcja anomalii w danych (0.0-0.5)
            method (str): Metoda wykrywania anomalii:
                        'isolation_forest', 'local_outlier_factor', 'one_class_svm'
        """
        self.contamination = contamination
        self.method = method
        self.model = None
        self.threshold = None
        logger.info(f"Inicjalizacja AnomalyDetector z metodą {method}")

        # Inicjalizacja wybranego modelu
        self._initialize_model()

    def _initialize_model(self):
        """Inicjalizuje model do wykrywania anomalii na podstawie wybranej metody."""
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20
            )
        elif self.method == "one_class_svm":
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel="rbf",
                gamma="scale"
            )
        else:
            raise ValueError(f"Nieznana metoda: {self.method}")

    @staticmethod
    def check_dependencies():
        """Sprawdza dostępność wymaganych zależności."""
        try:
            import numpy
            import sklearn
            return True
        except ImportError:
            return False

    def fit(self, data):
        """
        Trenowanie modelu na danych.

        Args:
            data (np.ndarray): Dane wejściowe, kształt [n_samples, n_features]

        Returns:
            self: Wytrenowany model
        """
        if data is None or len(data) == 0:
            raise ValueError("Dane wejściowe nie mogą być puste")

        # Sprawdzenie wymiarów danych
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        logger.info(f"Trenowanie modelu {self.method} na danych o kształcie {data.shape}")
        self.model.fit(data)
        return self

    def predict(self, data):
        """
        Wykrywanie anomalii w danych.

        Args:
            data (np.ndarray): Dane do analizy

        Returns:
            np.ndarray: -1 dla anomalii, 1 dla zwykłych obserwacji
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj najpierw metodę fit()")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        logger.info(f"Wykrywanie anomalii w danych o kształcie {data.shape}")
        return self.model.predict(data)

    def score_samples(self, data):
        """
        Zwraca wyniki anomalii dla każdej próbki (niższy wynik = większe prawdopodobieństwo anomalii).

        Args:
            data (np.ndarray): Dane do analizy

        Returns:
            np.ndarray: Wyniki anomalii
        """
        if self.model is None:
            raise ValueError("Model nie został wytrenowany. Wywołaj najpierw metodę fit()")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        if hasattr(self.model, "score_samples"):
            return self.model.score_samples(data)
        else:
            # Dla modeli bez metody score_samples
            return self.model.decision_function(data)

    def detect_anomalies(self, data, threshold=None):
        """
        Wykrywa anomalie w danych z opcjonalnym progiem.

        Args:
            data (np.ndarray): Dane do analizy
            threshold (float, optional): Próg anomalii, domyślnie None (używa modelu)

        Returns:
            tuple: (indeksy anomalii, wartości wyników)
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        scores = self.score_samples(data)

        if threshold is None:
            predictions = self.predict(data)
            anomaly_indices = np.where(predictions == -1)[0]
        else:
            anomaly_indices = np.where(scores < threshold)[0]

        return anomaly_indices, scores

    def analyze_price_series(self, prices, window_size=20, features=None):
        """
        Analizuje szereg czasowy cen pod kątem anomalii.

        Args:
            prices (np.ndarray): Szereg czasowy cen
            window_size (int): Rozmiar okna do ekstrakcji cech
            features (list): Lista funkcji do ekstrakcji cech

        Returns:
            tuple: (indeksy anomalii, wyniki anomalii)
        """
        if features is None:
            # Domyślne cechy: zwroty, zmienność, itp.
            features = self._default_features()

        # Przygotowanie danych cech
        feature_data = np.zeros((len(prices) - window_size, len(features)))

        for i in range(len(prices) - window_size):
            window = prices[i:i + window_size]
            for j, feature_func in enumerate(features):
                feature_data[i, j] = feature_func(window)

        # Trenowanie modelu jeśli nie był już trenowany
        if hasattr(self.model, "fit_predict"):
            self.fit(feature_data)

        # Wykrywanie anomalii
        anomaly_indices, scores = self.detect_anomalies(feature_data)

        # Korekta indeksów, aby odpowiadały oryginalnym danym
        adjusted_indices = anomaly_indices + window_size

        return adjusted_indices, scores

    def _default_features(self):
        """Zwraca domyślne funkcje ekstrakcji cech dla danych cenowych."""
        return [
            lambda x: np.mean(x),  # Średnia
            lambda x: np.std(x),   # Odchylenie standardowe
            lambda x: np.max(x) - np.min(x),  # Zakres
            lambda x: np.mean(np.diff(x)),  # Średnia zmiana
            lambda x: np.std(np.diff(x)),   # Zmienność zmian
        ]


def simple_anomaly_test():
    """Prosta funkcja testująca moduł wykrywania anomalii."""
    # Generowanie danych testowych
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    anomalies = np.random.normal(5, 1, 50)

    # Wstawienie anomalii w losowe miejsca
    indices = np.random.choice(1000, 50, replace=False)
    data = normal_data.copy()
    data[indices] = anomalies

    # Wykrywanie anomalii
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(data.reshape(-1, 1))
    anomaly_indices, _ = detector.detect_anomalies(data.reshape(-1, 1))

    # Ocena skuteczności
    correct_detections = np.intersect1d(anomaly_indices, indices)
    precision = len(correct_detections) / len(anomaly_indices) if len(anomaly_indices) > 0 else 0
    recall = len(correct_detections) / len(indices)

    print(f"Wykryto {len(anomaly_indices)} anomalii, z czego {len(correct_detections)} prawidłowo")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

    return precision, recall


if __name__ == "__main__":
    # Konfiguracja logowania
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Uruchomienie prostego testu
    precision, recall = simple_anomaly_test()