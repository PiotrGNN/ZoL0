"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import numpy as np
import time
import random
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Wykrywacz anomalii w danych rynkowych."""

    def __init__(self, method: str = "z_score", threshold: float = 2.5):
        """
        Inicjalizuje wykrywacz anomalii.

        Parameters:
            method (str): Metoda wykrywania anomalii ('z_score', 'iqr', 'isolation_forest')
            threshold (float): Próg wykrywania anomalii
        """
        self.method = method
        self.threshold = threshold
        self.history_size = 1000  # Domyślny rozmiar historii
        self.price_history = []
        self.volume_history = []
        self.detected_anomalies = []
        self.last_detection_time = time.time()
        logger.info(f"Zainicjalizowano AnomalyDetector (metoda: {method}, próg: {threshold})")

    def detect(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wykrywa anomalie w danych.

        Parameters:
            data_point (Dict[str, Any]): Punkt danych do analizy

        Returns:
            Dict[str, Any]: Wynik detekcji anomalii
        """
        # Przygotuj dane
        price = data_point.get("price", 0)
        volume = data_point.get("volume", 0)
        timestamp = data_point.get("timestamp", time.time())

        # Dodaj dane do historii
        self.price_history.append(price)
        self.volume_history.append(volume)

        # Ogranicz rozmiar historii
        if len(self.price_history) > self.history_size:
            self.price_history = self.price_history[-self.history_size:]
            self.volume_history = self.volume_history[-self.history_size:]

        # Wykryj anomalie
        is_price_anomaly = False
        is_volume_anomaly = False
        anomaly_score = 0.0

        if len(self.price_history) > 5:  # Potrzebujemy przynajmniej kilku punktów danych
            if self.method == "z_score":
                is_price_anomaly, price_score = self._detect_z_score_anomaly(self.price_history, price)
                is_volume_anomaly, volume_score = self._detect_z_score_anomaly(self.volume_history, volume)
                anomaly_score = max(price_score, volume_score)
            elif self.method == "iqr":
                is_price_anomaly, price_score = self._detect_iqr_anomaly(self.price_history, price)
                is_volume_anomaly, volume_score = self._detect_iqr_anomaly(self.volume_history, volume)
                anomaly_score = max(price_score, volume_score)
            else:
                # Domyślna metoda
                is_price_anomaly, price_score = self._detect_z_score_anomaly(self.price_history, price)
                is_volume_anomaly, volume_score = self._detect_z_score_anomaly(self.volume_history, volume)
                anomaly_score = max(price_score, volume_score)

        # Jeśli wykryto anomalię, dodaj ją do listy
        if is_price_anomaly or is_volume_anomaly:
            anomaly = {
                "timestamp": timestamp,
                "price": price,
                "volume": volume,
                "is_price_anomaly": is_price_anomaly,
                "is_volume_anomaly": is_volume_anomaly,
                "score": anomaly_score,
                "method": self.method
            }

            self.detected_anomalies.append(anomaly)

            # Ogranicz rozmiar historii anomalii
            if len(self.detected_anomalies) > 100:
                self.detected_anomalies = self.detected_anomalies[-100:]

            logger.info(f"Wykryto anomalię: {anomaly}")

        self.last_detection_time = time.time()

        return {
            "is_anomaly": is_price_anomaly or is_volume_anomaly,
            "price_anomaly": is_price_anomaly,
            "volume_anomaly": is_volume_anomaly,
            "score": anomaly_score,
            "timestamp": timestamp
        }

    def _detect_z_score_anomaly(self, data: List[float], value: float) -> Tuple[bool, float]:
        """
        Wykrywa anomalie metodą z-score.

        Parameters:
            data (List[float]): Historia danych
            value (float): Wartość do sprawdzenia

        Returns:
            Tuple[bool, float]: (czy_anomalia, wynik_z_score)
        """
        if len(data) < 2:
            return False, 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return False, 0.0

        z_score = abs((value - mean) / std)

        return z_score > self.threshold, z_score

    def _detect_iqr_anomaly(self, data: List[float], value: float) -> Tuple[bool, float]:
        """
        Wykrywa anomalie metodą IQR (Interquartile Range).

        Parameters:
            data (List[float]): Historia danych
            value (float): Wartość do sprawdzenia

        Returns:
            Tuple[bool, float]: (czy_anomalia, wynik_iqr)
        """
        if len(data) < 4:
            return False, 0.0

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        if iqr == 0:
            return False, 0.0

        lower_bound = q1 - (self.threshold * iqr)
        upper_bound = q3 + (self.threshold * iqr)

        is_anomaly = value < lower_bound or value > upper_bound

        # Oblicz "score" jako odległość od najbliższej granicy
        if value < lower_bound:
            score = (lower_bound - value) / iqr
        elif value > upper_bound:
            score = (value - upper_bound) / iqr
        else:
            score = 0.0

        return is_anomaly, score

    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca wykryte anomalie.

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii
        """
        # Symulowane dane dla celów demonstracyjnych
        if not self.detected_anomalies:
            current_time = time.time()
            for i in range(3):
                anomaly_time = current_time - random.randint(60, 3600)
                anomaly = {
                    "timestamp": anomaly_time,
                    "price": random.uniform(30000, 40000),
                    "volume": random.uniform(100, 500),
                    "is_price_anomaly": random.choice([True, False]),
                    "is_volume_anomaly": random.choice([True, False]),
                    "score": random.uniform(2.5, 5.0),
                    "method": self.method,
                    "description": f"Wykryto nietypową aktywność rynkową",
                    "type": random.choice(["price_spike", "volume_spike", "price_drop"])
                }
                self.detected_anomalies.append(anomaly)

        return self.detected_anomalies

    def set_method(self, method: str) -> bool:
        """
        Ustawia metodę wykrywania anomalii.

        Parameters:
            method (str): Metoda wykrywania anomalii

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            self.method = method
            logger.info(f"Zmieniono metodę wykrywania anomalii na: {method}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas zmiany metody wykrywania anomalii: {e}")
            return False

    def set_threshold(self, threshold: float) -> bool:
        """
        Ustawia próg wykrywania anomalii.

        Parameters:
            threshold (float): Próg wykrywania anomalii

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            self.threshold = threshold
            logger.info(f"Zmieniono próg wykrywania anomalii na: {threshold}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas zmiany progu wykrywania anomalii: {e}")
            return False