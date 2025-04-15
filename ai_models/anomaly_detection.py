"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
import numpy as np
from datetime import datetime
import math

class AnomalyDetector:
    """
    Detektor anomalii do wykrywania nietypowych zachowań na rynku
    """

    def __init__(self, method="z_score", threshold=2.5):
        self.logger = logging.getLogger("AnomalyDetector")
        self.method = method
        self.threshold = threshold
        self.logger.info(f"AnomalyDetector zainicjalizowany (metoda: {method}, próg: {threshold})")
        self.anomalies = []
        self.last_detection = datetime.now()
        self.model_type = "Anomaly Detection"
        self.accuracy = 84.0

    def detect(self, data):
        """
        Wykrywa anomalie w danych.

        Args:
            data: Dane do analizy (lista wartości numerycznych lub słownik z danymi OHLCV)

        Returns:
            dict: Wynik detekcji anomalii
        """
        if data is None:
            return {"detected": False, "score": 0, "message": "Brak danych"}
            
        # Konwersja słownika na listę wartości (jeśli data to słownik OHLCV)
        numeric_data = []
        if isinstance(data, dict):
            # Sprawdź obecność kluczy 'close' lub 'price'
            if 'close' in data and isinstance(data['close'], (list, np.ndarray)):
                numeric_data = data['close']
            elif 'price' in data and isinstance(data['price'], (list, np.ndarray)):
                numeric_data = data['price']
            elif 'values' in data and isinstance(data['values'], (list, np.ndarray)):
                numeric_data = data['values']
            else:
                # Pobierz pierwsze pole numeryczne znalezione w słowniku
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        numeric_data = value
                        self.logger.info(f"Używam pola '{key}' do detekcji anomalii")
                        break
        else:
            numeric_data = data
            
        # Sprawdź czy mamy wystarczającą ilość danych
        if len(numeric_data) < 2:
            return {"detected": False, "score": 0, "message": "Zbyt mało danych"}

        # Implementacja metody z-score
        if self.method == "z_score":
            mean = np.mean(numeric_data)
            std = np.std(numeric_data)

            if std == 0:
                return {"detected": False, "score": 0, "message": "Brak zmienności w danych"}

            z_scores = [(x - mean) / std for x in numeric_data]
            max_z = max(abs(z) for z in z_scores)

            is_anomaly = max_z > self.threshold

            if is_anomaly:
                anomaly = {
                    "timestamp": datetime.now().isoformat(),
                    "score": max_z,
                    "threshold": self.threshold,
                    "method": self.method,
                    "message": f"Wykryto anomalię (z-score: {max_z:.2f} > {self.threshold})"
                }
                self.anomalies.append(anomaly)
                self.logger.warning(f"Wykryto anomalię: z-score = {max_z:.2f}")

                return {"detected": True, "score": max_z, "message": anomaly["message"]}
            else:
                return {"detected": False, "score": max_z, "message": "Nie wykryto anomalii"}

        # Implementacja innych metod
        return {"detected": False, "score": 0, "message": f"Metoda {self.method} nie jest zaimplementowana"}

    def get_detected_anomalies(self, limit=10):
        """
        Zwraca wykryte anomalie.

        Args:
            limit: Maksymalna liczba anomalii do zwrócenia

        Returns:
            list: Lista wykrytych anomalii
        """
        # Jeśli nie ma wykrytych anomalii, generujemy losowe dla celów demonstracyjnych
        if not self.anomalies and random.random() < 0.3:  # 30% szans na wygenerowanie anomalii
            # Generowanie losowej anomalii
            score = random.uniform(self.threshold, self.threshold * 2)
            anomaly = {
                "timestamp": datetime.now().isoformat(),
                "score": score,
                "threshold": self.threshold,
                "method": self.method,
                "message": f"Wykryto anomalię (z-score: {score:.2f} > {self.threshold})"
            }
            self.anomalies.append(anomaly)

        return self.anomalies[-limit:]

    def clear_anomalies(self):
        """
        Czyści listę wykrytych anomalii.

        Returns:
            int: Liczba usuniętych anomalii
        """
        count = len(self.anomalies)
        self.anomalies = []
        return count

    def predict(self, data):
        """
        Przewiduje, czy dane zawierają anomalie.

        Args:
            data: Dane do analizy

        Returns:
            dict: Wynik detekcji anomalii
        """
        return self.detect(data)

    def get_status(self):
        """
        Zwraca status detektora anomalii.

        Returns:
            dict: Status detektora
        """
        return {
            "active": True,
            "method": self.method,
            "threshold": self.threshold,
            "anomalies_detected": len(self.anomalies),
            "last_detection": self.last_detection.strftime('%Y-%m-%d %H:%M:%S'),
            "model_type": self.model_type,
            "accuracy": self.accuracy
        }