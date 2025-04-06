"""
Moduł wykrywania anomalii przy użyciu metod AI/ML
Zoptymalizowana wersja dla środowiska Replit
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetectionModel:
    """Model wykrywania anomalii w danych rynkowych."""

    def __init__(self, contamination=0.05):
        """
        Inicjalizacja modelu wykrywania anomalii.

        Parameters:
            contamination (float): oczekiwana proporcja anomalii w danych (0.0 - 0.5)
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1  # Wykorzystaj wszystkie dostępne rdzenie
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        logging.info("Model wykrywania anomalii zainicjalizowany")

    def fit(self, data):
        """
        Dopasowuje model do danych.

        Parameters:
            data (pd.DataFrame/np.ndarray): Dane do treningu modelu
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Usunięcie wartości NaN
                data = data.dropna()

                # Konwersja do tablicy NumPy jeśli to DataFrame
                X = data.values if len(data.shape) > 1 else data.values.reshape(-1, 1)
            else:
                X = data if len(data.shape) > 1 else data.reshape(-1, 1)

            # Skalowanie danych
            X_scaled = self.scaler.fit_transform(X)

            # Trenowanie modelu
            self.model.fit(X_scaled)
            self.is_fitted = True
            logging.info(f"Model wykrywania anomalii wytrenowany na {X.shape[0]} próbkach")
            return True
        except Exception as e:
            logging.error(f"Błąd podczas trenowania modelu wykrywania anomalii: {e}")
            return False

    def detect(self, data, threshold=-0.5):
        """
        Wykrywa anomalie w danych.

        Parameters:
            data (pd.DataFrame/np.ndarray): Dane do analizy
            threshold (float): Próg decyzyjny (niższe wartości -> więcej anomalii)

        Returns:
            np.ndarray: Indeksy wykrytych anomalii
        """
        if not self.is_fitted:
            logging.warning("Model nie jest wytrenowany. Najpierw wywołaj metodę fit().")
            return np.array([])

        try:
            if isinstance(data, pd.DataFrame):
                # Zachowanie indeksów DataFrame do późniejszego zwrócenia
                original_index = data.index

                # Usunięcie wartości NaN
                data = data.dropna()
                data_index = data.index

                # Konwersja do tablicy NumPy
                X = data.values if len(data.shape) > 1 else data.values.reshape(-1, 1)
            else:
                X = data if len(data.shape) > 1 else data.reshape(-1, 1)
                original_index = np.arange(len(X))
                data_index = original_index

            # Skalowanie danych
            X_scaled = self.scaler.transform(X)

            # Przewidywanie anomalii (niższe wartości = większe prawdopodobieństwo anomalii)
            scores = self.model.decision_function(X_scaled)

            # Znalezienie indeksów anomalii
            anomaly_indices = np.where(scores < threshold)[0]

            # Mapowanie indeksów z powrotem na oryginalne indeksy
            if isinstance(data, pd.DataFrame):
                anomaly_indices = data_index[anomaly_indices]

            logging.info(f"Wykryto {len(anomaly_indices)} anomalii wśród {len(X)} próbek")
            return anomaly_indices
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania anomalii: {e}")
            return np.array([])

    def predict(self, data):
        """
        Przewiduje, które punkty są anomaliami (-1) lub normalne (1).

        Parameters:
            data (pd.DataFrame/np.ndarray): Dane do analizy

        Returns:
            np.ndarray: Etykiety dla każdego punktu danych (-1: anomalia, 1: normalne)
        """
        if not self.is_fitted:
            logging.warning("Model nie jest wytrenowany. Najpierw wywołaj metodę fit().")
            return np.array([])

        try:
            if isinstance(data, pd.DataFrame):
                # Usunięcie wartości NaN
                data = data.dropna()

                # Konwersja do tablicy NumPy
                X = data.values if len(data.shape) > 1 else data.values.reshape(-1, 1)
            else:
                X = data if len(data.shape) > 1 else data.reshape(-1, 1)

            # Skalowanie danych
            X_scaled = self.scaler.transform(X)

            # Przewidywanie anomalii
            predictions = self.model.predict(X_scaled)
            logging.info(f"Wykonano predykcję anomalii dla {len(X)} próbek")
            return predictions
        except Exception as e:
            logging.error(f"Błąd podczas przewidywania anomalii: {e}")
            return np.array([])

    def detect_multiple_features(self, data_dict, thresholds=None):
        """
        Wykrywa anomalie w wielu cechach/seriach danych.

        Parameters:
            data_dict (dict): Słownik {nazwa_cechy: dane}
            thresholds (dict, optional): Słownik {nazwa_cechy: próg}

        Returns:
            dict: Słownik {nazwa_cechy: indeksy_anomalii}
        """
        results = {}

        if thresholds is None:
            thresholds = {key: -0.5 for key in data_dict.keys()}

        for feature_name, data in data_dict.items():
            # Trenowanie nowego modelu dla każdej cechy
            feature_model = AnomalyDetectionModel()
            if feature_model.fit(data):
                threshold = thresholds.get(feature_name, -0.5)
                anomalies = feature_model.detect(data, threshold)
                results[feature_name] = anomalies
                logging.info(f"Wykryto {len(anomalies)} anomalii dla cechy '{feature_name}'")
            else:
                results[feature_name] = np.array([])
                logging.warning(f"Nie udało się wykryć anomalii dla cechy '{feature_name}'")

        return results

# Przykład użycia
if __name__ == "__main__":
    # Generowanie przykładowych danych
    np.random.seed(42)
    n_samples = 200
    n_outliers = 10

    # Normalne dane
    normal_data = np.random.normal(0, 1, size=(n_samples - n_outliers, 1))

    # Anomalie
    outliers = np.random.uniform(5, 10, size=(n_outliers, 1))

    # Połączenie danych
    X = np.vstack([normal_data, outliers])
    np.random.shuffle(X)

    # Tworzenie i trenowanie modelu
    model = AnomalyDetectionModel(contamination=n_outliers/n_samples)
    model.fit(X)

    # Wykrywanie anomalii
    anomaly_indices = model.detect(X)
    print(f"Wykryte anomalie (indeksy): {anomaly_indices}")

    # Predykcja wszystkich punktów
    predictions = model.predict(X)
    n_detected_anomalies = np.sum(predictions == -1)
    print(f"Liczba wykrytych anomalii: {n_detected_anomalies}")


# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Uruchomienie prostego testu (zachowana z oryginalnego kodu)
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
    detector = AnomalyDetectionModel(contamination=0.05)
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
    precision, recall = simple_anomaly_test()