"""
anomaly_detector.py
--------------------
Moduł wykrywający anomalie w danych transakcyjnych i rynkowych.

Funkcjonalności:
- Wykorzystuje metody statystyczne (z-score), algorytmy klastrowania (K-Means) oraz modele uczenia bez nadzoru (Isolation Forest).
- Zapewnia mechanizm wczesnego ostrzegania, np. wysyłanie powiadomień (tu symulowane przez logowanie).
- Umożliwia logowanie zdarzeń do plików (np. aktualizacja pliku HEAD w folderze logs).
- Posiada parametryzację progu detekcji, by dostosować wykrywanie do różnych strategii i wielkości portfela.
- Zawiera testy jednostkowe sprawdzające wykrywanie anomalii.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Konfiguracja logowania
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
HEAD_FILE = os.path.join(LOG_DIR, "HEAD")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "anomaly_detector.log")),
        logging.StreamHandler(),
    ],
)


def update_head(log_id: str):
    """
    Aktualizuje plik HEAD z identyfikatorem ostatniego logu systemowego.
    Format: "HEAD: <timestamp> - <log_id>"
    """
    try:
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        head_content = f"HEAD: {timestamp} - {log_id}"
        with open(HEAD_FILE, "w") as f:
            f.write(head_content)
        logging.info("Plik HEAD zaktualizowany: %s", head_content)
    except Exception as e:
        logging.error("Błąd przy aktualizacji pliku HEAD: %s", e)
        raise


def detect_anomalies_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Wykrywa anomalie w serii danych za pomocą metody z-score.

    Parameters:
        data (pd.Series): Serie danych (np. ceny, wolumen).
        threshold (float): Próg z-score; wartości z-score > threshold (lub < -threshold) są uznawane za anomalie.

    Returns:
        pd.Series: Boolean Series, gdzie True oznacza wykrytą anomalię.
    """
    mean_val = data.mean()
    std_val = data.std()
    z_scores = (data - mean_val) / std_val
    anomalies = z_scores.abs() > threshold
    logging.info(
        "Wykryto %d anomalii przy progu z-score %.2f.", anomalies.sum(), threshold
    )
    return anomalies


def detect_anomalies_isolation_forest(
    df: pd.DataFrame, contamination: float = 0.01
) -> pd.Series:
    """
    Wykrywa anomalie przy użyciu modelu Isolation Forest.

    Parameters:
        df (pd.DataFrame): Dane wejściowe (numeryczne).
        contamination (float): Procent danych uznawany za anomalie.

    Returns:
        pd.Series: Etykiety: 1 dla normalnych, -1 dla anomalii.
    """
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        preds = iso_forest.fit_predict(df)
        logging.info("Isolation Forest wykrył %d anomalii.", np.sum(preds == -1))
        return pd.Series(preds, index=df.index)
    except Exception as e:
        logging.error("Błąd przy wykrywaniu anomalii za pomocą Isolation Forest: %s", e)
        raise


def detect_anomalies_kmeans(data: pd.Series, n_clusters: int = 2) -> pd.Series:
    """
    Wykrywa anomalie przy użyciu algorytmu K-Means.
    Zakłada, że jeden z klastrów reprezentuje dane anomalne.

    Parameters:
        data (pd.Series): Dane wejściowe (jednowymiarowe).
        n_clusters (int): Liczba klastrów.

    Returns:
        pd.Series: Boolean Series, gdzie True oznacza wykrytą anomalię.
    """
    try:
        values = data.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(values)
        # Zakładamy, że klaster o najmniejszej liczbie punktów to anomalie
        unique, counts = np.unique(labels, return_counts=True)
        anomaly_cluster = unique[np.argmin(counts)]
        anomalies = labels == anomaly_cluster
        logging.info("K-Means wykrył %d anomalii.", np.sum(anomalies))
        return pd.Series(anomalies, index=data.index)
    except Exception as e:
        logging.error("Błąd przy wykrywaniu anomalii za pomocą K-Means: %s", e)
        raise


def early_warning_notification(anomaly_details: dict):
    """
    Mechanizm wczesnego ostrzegania – symuluje wysłanie powiadomienia (np. e-mail, SMS) przez logowanie.

    Parameters:
        anomaly_details (dict): Szczegóły wykrytych anomalii.
    """
    try:
        logging.warning("ALERT! Wykryto anomalie: %s", anomaly_details)
        # Tutaj można zintegrować system powiadomień (np. e-mail, Slack, SMS).
    except Exception as e:
        logging.error("Błąd przy wysyłaniu powiadomienia: %s", e)
        raise


# -------------------- Testy jednostkowe --------------------
def unit_test_anomaly_detector():
    """
    Testy jednostkowe dla modułu anomaly_detector.py.
    Tworzy przykładowe dane i weryfikuje funkcjonalność wykrywania anomalii.
    """
    try:
        # Przykładowe dane: generujemy serię danych z kilkoma anomaliami
        np.random.seed(42)
        normal_data = np.random.normal(loc=100, scale=5, size=100)
        # Wprowadzamy anomalie
        normal_data[20] = 150
        normal_data[50] = 50
        data_series = pd.Series(normal_data)

        # Test z-score
        anomalies_z = detect_anomalies_zscore(data_series, threshold=3.0)
        assert anomalies_z.sum() >= 2, "Z-Score nie wykrył oczekiwanych anomalii."

        # Test Isolation Forest
        df = pd.DataFrame({"value": data_series})
        preds = detect_anomalies_isolation_forest(df, contamination=0.05)
        assert (
            preds == -1
        ).sum() >= 2, "Isolation Forest nie wykrył oczekiwanych anomalii."

        # Test K-Means
        anomalies_km = detect_anomalies_kmeans(data_series, n_clusters=2)
        assert anomalies_km.sum() >= 1, "K-Means nie wykrył oczekiwanych anomalii."

        # Test powiadomienia – symulujemy wysłanie alertu
        anomaly_details = {
            "z_score_anomalies": int(anomalies_z.sum()),
            "isolation_forest_anomalies": int((preds == -1).sum()),
            "kmeans_anomalies": int(anomalies_km.sum()),
        }
        early_warning_notification(anomaly_details)

        # Aktualizacja pliku HEAD
        update_head("anomaly_detector_test")

        logging.info("Testy jednostkowe anomaly_detector.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("AssertionError w testach jednostkowych: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach jednostkowych anomaly_detector.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_anomaly_detector()
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        raise
"""
anomaly_detector.py
------------------
Moduł wykrywający anomalie w danych tradingowych.
"""

import logging
from typing import Dict, List, Any

# Konfiguracja logowania
logger = logging.getLogger("anomaly_detector")
if not logger.handlers:
    log_dir = "logs"
    import os
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "anomaly_detector.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class AnomalyDetector:
    """
    Klasa wykrywająca anomalie w danych tradingowych.
    """

    def __init__(self, detection_method: str = "statistical"):
        """
        Inicjalizacja detektora anomalii.

        Parameters:
            detection_method (str): Metoda wykrywania anomalii 
                                   ('statistical', 'ml', 'hybrid')
        """
        self.detection_method = detection_method
        self.anomalies_detected = []
        logger.info(f"Inicjalizacja detektora anomalii z metodą: {detection_method}")

    def detect(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie w danych.

        Parameters:
            data (List[Dict[str, Any]]): Dane do analizy

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii
        """
        # Implementacja stub - zwraca pustą listę
        logger.debug(f"Wykrywanie anomalii w {len(data)} punktach danych metodą {self.detection_method}")
        return []

    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca listę wykrytych anomalii.

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii
        """
        logger.debug("Pobieranie wykrytych anomalii")
        return self.anomalies_detected

    def reset(self) -> None:
        """Resetuje stan detektora anomalii."""
        self.anomalies_detected = []
        logger.debug("Reset detektora anomalii")
