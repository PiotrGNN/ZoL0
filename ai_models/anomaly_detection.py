"""
anomaly_detection.py - Moduł do wykrywania anomalii w danych rynkowych
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional

# Eksport klasy AnomalyDetector
__all__ = ["AnomalyDetector"]

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Klasa do wykrywania anomalii w danych rynkowych.
    Wykorzystuje różne metody detekcji anomalii, takie jak Z-score, IQR, itp.
    """

    def __init__(self, method: str = "z_score", threshold: float = 3.0):
        """
        Inicjalizacja detektora anomalii.

        Parameters:
            method (str): Metoda wykrywania anomalii ('z_score', 'iqr', 'local_outlier')
            threshold (float): Próg uznania obserwacji za anomalię
        """
        self.method = method
        self.threshold = threshold
        self.anomalies = []
        self.detection_method = method
        logging.info(f"Zainicjalizowano AnomalyDetector (metoda: {method}, próg: {threshold})")

    def detect(self, data, column=None):
        """
        Wykrywa anomalie w danych.

        Parameters:
            data: Dane do analizy (np. numpy array, pandas DataFrame, lista)
            column: Opcjonalna nazwa kolumny, jeśli data to DataFrame

        Returns:
            List[int]: Indeksy wykrytych anomalii
        """
        if self.method == "z_score":
            return self._detect_zscore(data, column)
        elif self.method == "iqr":
            return self._detect_iqr(data, column)
        else:
            logging.warning(f"Nieznana metoda detekcji anomalii: {self.method}. Użycie domyślnej z_score.")
            return self._detect_zscore(data, column)

    def _detect_zscore(self, data, column=None):
        """
        Wykrywa anomalie metodą Z-score.

        Parameters:
            data: Dane do analizy
            column: Opcjonalna nazwa kolumny

        Returns:
            List[int]: Indeksy wykrytych anomalii
        """
        try:
            # Konwersja na numpy array
            if column is not None and hasattr(data, 'loc'):
                values = data[column].values
            else:
                values = np.array(data)

            # Obliczenie Z-score
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                return []

            z_scores = np.abs((values - mean) / std)

            # Znalezienie anomalii
            anomaly_indices = np.where(z_scores > self.threshold)[0]

            # Logowanie wykrytych anomalii
            if len(anomaly_indices) > 0:
                logging.info(f"Wykryto {len(anomaly_indices)} anomalii metodą Z-score")
                for idx in anomaly_indices:
                    self.log_anomaly({
                        'index': int(idx),
                        'value': float(values[idx]),
                        'z_score': float(z_scores[idx]),
                        'method': 'z_score'
                    })

            return anomaly_indices.tolist()
        except Exception as e:
            logging.error(f"Błąd podczas detekcji anomalii metodą Z-score: {e}")
            return []

    def _detect_iqr(self, data, column=None):
        """
        Wykrywa anomalie metodą IQR (Interquartile Range).

        Parameters:
            data: Dane do analizy
            column: Opcjonalna nazwa kolumny

        Returns:
            List[int]: Indeksy wykrytych anomalii
        """
        try:
            # Konwersja na numpy array
            if column is not None and hasattr(data, 'loc'):
                values = data[column].values
            else:
                values = np.array(data)

            # Obliczenie IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower_bound = q1 - (self.threshold * iqr)
            upper_bound = q3 + (self.threshold * iqr)

            # Znalezienie anomalii
            anomalies_lower = values < lower_bound
            anomalies_upper = values > upper_bound
            anomaly_indices = np.where(anomalies_lower | anomalies_upper)[0]

            # Logowanie wykrytych anomalii
            if len(anomaly_indices) > 0:
                logging.info(f"Wykryto {len(anomaly_indices)} anomalii metodą IQR")
                for idx in anomaly_indices:
                    value = values[idx]
                    bound_type = "lower" if value < lower_bound else "upper"
                    bound_value = lower_bound if bound_type == "lower" else upper_bound
                    self.log_anomaly({
                        'index': int(idx),
                        'value': float(value),
                        'bound_type': bound_type,
                        'bound_value': float(bound_value),
                        'method': 'iqr'
                    })

            return anomaly_indices.tolist()
        except Exception as e:
            logging.error(f"Błąd podczas detekcji anomalii metodą IQR: {e}")
            return []

    def analyze_market_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analizuje dane rynkowe w poszukiwaniu anomalii.

        Parameters:
            data (Dict[str, Any]): Dane do analizy.

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        # Implementacja detekcji anomalii
        # W szablonie zwracamy pustą listę
        return []

    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca wykryte anomalie z pamięci podręcznej.

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        return self.anomalies

    def log_anomaly(self, anomaly: Dict[str, Any]) -> None:
        """
        Loguje wykrytą anomalię.

        Parameters:
            anomaly (Dict[str, Any]): Dane anomalii.
        """
        logging.warning(f"Wykryto anomalię: {anomaly}")
        self.anomalies.append({
            **anomaly,
            "timestamp": time.time(),
            "detection_method": self.detection_method
        })

        # Ograniczamy liczbę przechowywanych anomalii do 100
        if len(self.anomalies) > 100:
            self.anomalies = self.anomalies[-100:]