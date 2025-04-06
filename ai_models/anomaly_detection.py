"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych używający różnych technik 
statystycznych i uczenia maszynowego.
"""

import logging
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/anomaly_detector.log"),
        logging.StreamHandler()
    ]
)

# Upewnij się, że katalog logs istnieje
os.makedirs("logs", exist_ok=True)

class AnomalyDetectionModel:
    """
    Klasa implementująca różne metody wykrywania anomalii w danych finansowych.
    """

    def __init__(self, contamination=0.05, random_state=42):
        """
        Inicjalizacja modelu wykrywania anomalii.

        Args:
            contamination (float): Oczekiwany procent anomalii w danych (0.0-0.5)
            random_state (int): Ziarno losowości dla powtarzalnych wyników
        """
        self.logger = logging.getLogger(__name__)
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.logger.info("Zainicjalizowano model wykrywania anomalii")

    def fit(self, data):
        """
        Trenuje model na podanych danych.

        Args:
            data (pd.DataFrame/np.array): Dane do treningu

        Returns:
            self: Wytrenowany model
        """
        try:
            # Sprawdzenie typu danych
            if isinstance(data, pd.DataFrame):
                X = data.values
            else:
                X = data

            # Normalizacja danych
            X_scaled = self.scaler.fit_transform(X)

            # Trenowanie modelu
            self.model.fit(X_scaled)
            self.logger.info(f"Model wytrenowany na {X.shape[0]} przykładach")
            return self
        except Exception as e:
            self.logger.error(f"Błąd podczas trenowania modelu: {e}")
            raise

    def predict(self, data):
        """
        Przewiduje anomalie w danych.

        Args:
            data (pd.DataFrame/np.array): Dane do analizy

        Returns:
            np.array: Tablica etykiet (-1 dla anomalii, 1 dla normalnych danych)
        """
        try:
            # Sprawdzenie typu danych
            if isinstance(data, pd.DataFrame):
                X = data.values
            else:
                X = data

            # Normalizacja danych
            X_scaled = self.scaler.transform(X)

            # Predykcja
            predictions = self.model.predict(X_scaled)
            self.logger.info(f"Wykonano predykcję na {X.shape[0]} przykładach")
            return predictions
        except Exception as e:
            self.logger.error(f"Błąd podczas predykcji: {e}")
            raise

    def detect_anomalies(self, data, threshold=2.5):
        """
        Wykrywa anomalie w danych używając standardowego odchylenia.

        Args:
            data (pd.Series/np.array): Seria danych do analizy
            threshold (float): Liczba odchyleń standardowych powyżej której
                              punkt jest uznawany za anomalię

        Returns:
            np.array: Tablica wartości boolowskich (True dla anomalii)
        """
        try:
            # Konwersja do numpy array
            if isinstance(data, pd.Series):
                values = data.values
            else:
                values = data

            mean = np.mean(values)
            std = np.std(values)

            # Wykrywanie anomalii
            anomalies = np.abs(values - mean) > threshold * std

            anomaly_count = np.sum(anomalies)
            self.logger.info(f"Wykryto {anomaly_count} anomalii wśród {len(values)} punktów danych")
            return anomalies
        except Exception as e:
            self.logger.error(f"Błąd podczas wykrywania anomalii: {e}")
            raise

    def detect_price_anomalies(self, price_data, window_size=20, threshold=2.5):
        """
        Wykrywa anomalie cenowe na podstawie ruchomego średniego odchylenia.

        Args:
            price_data (pd.Series): Seria danych cenowych
            window_size (int): Rozmiar okna do obliczenia ruchomej średniej
            threshold (float): Próg odchylenia dla anomalii

        Returns:
            pd.Series: Seria boolowska (True dla anomalii)
        """
        try:
            # Obliczenie ruchomej średniej
            rolling_mean = price_data.rolling(window=window_size).mean()
            rolling_std = price_data.rolling(window=window_size).std()

            # Obliczenie odchylenia
            z_score = (price_data - rolling_mean) / rolling_std

            # Identyfikacja anomalii
            anomalies = abs(z_score) > threshold

            # Uzupełnienie wartości NaN
            anomalies = anomalies.fillna(False)

            anomaly_count = anomalies.sum()
            self.logger.info(f"Wykryto {anomaly_count} anomalii cenowych w szeregu czasowym")
            return anomalies
        except Exception as e:
            self.logger.error(f"Błąd podczas wykrywania anomalii cenowych: {e}")
            raise

    def visualize_anomalies(self, data, anomalies, title="Wykryte anomalie"):
        """
        Wizualizuje wykryte anomalie.

        Args:
            data (pd.Series): Oryginalne dane
            anomalies (pd.Series): Seria boolowska z wynikami detekcji
            title (str): Tytuł wykresu
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(data, label='Dane')
            plt.scatter(data[anomalies].index, data[anomalies], 
                      color='red', label='Anomalie')
            plt.title(title)
            plt.legend()
            plt.tight_layout()

            # Zapisanie wykresu
            os.makedirs("reports", exist_ok=True)
            plt.savefig(f"reports/{title.replace(' ', '_')}.png")
            plt.close()

            self.logger.info(f"Wygenerowano wizualizację anomalii: reports/{title.replace(' ', '_')}.png")
        except ImportError:
            self.logger.warning("Biblioteka matplotlib nie jest zainstalowana. Wizualizacja niedostępna.")
        except Exception as e:
            self.logger.error(f"Błąd podczas wizualizacji anomalii: {e}")


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
    anomaly_indices = model.detect_anomalies(X.flatten()) #Adapt to new method
    print(f"Wykryte anomalie (indeksy): {anomaly_indices}")

    # Predykcja wszystkich punktów
    predictions = model.predict(X)
    n_detected_anomalies = np.sum(predictions == -1)
    print(f"Liczba wykrytych anomalii: {n_detected_anomalies}")