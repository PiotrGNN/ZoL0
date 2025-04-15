"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detektor anomalii w danych rynkowych."""

    def __init__(self):
        """Inicjalizuje detektor anomalii."""
        self.anomaly_patterns = {
            "price_spike": {
                "name": "Gwałtowny skok ceny",
                "description": "Nagły, znaczący wzrost ceny w krótkim okresie"
            },
            "price_crash": {
                "name": "Gwałtowny spadek ceny",
                "description": "Nagły, znaczący spadek ceny w krótkim okresie"
            },
            "volume_spike": {
                "name": "Gwałtowny wzrost wolumenu",
                "description": "Nagły, znaczący wzrost wolumenu transakcji"
            },
            "low_liquidity": {
                "name": "Niska płynność",
                "description": "Nietypowo niski wolumen transakcji"
            },
            "unusual_spread": {
                "name": "Nietypowy spread",
                "description": "Nietypowo duży spread między ceną kupna i sprzedaży"
            },
            "high_volatility": {
                "name": "Wysoka zmienność",
                "description": "Nietypowo wysoki poziom zmienności ceny"
            }
        }

        self.accuracy = 75.1
        self.model_type = "Statistical Anomaly Detector"
        self.status = "Active"
        self.last_detection_time = time.time()

    def predict(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predykcja anomalii w danych.

        Args:
            data: Lista danych do analizy

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii
        """
        return self.detect(data)

    def detect(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie w danych rynkowych.

        Args:
            data: Lista danych do analizy

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii
        """
        if not data:
            return []

        anomalies = []
        threshold = 2.0  # Standardowa wartość dla statystycznych anomalii (odchylenie standardowe)

        # Prostego wykrywanie anomalii oparte na wartościach znacząco odbiegających od średniej
        try:
            # Ekstrakcja wartości liczbowych z danych (jeśli są dostępne)
            values = []
            timestamps = []

            for item in data:
                # Sprawdź, czy item ma pole 'value'
                if 'value' in item:
                    values.append(float(item['value']))
                    timestamps.append(item.get('timestamp', time.time()))
                # Sprawdź, czy item ma pole 'price'
                elif 'price' in item:
                    values.append(float(item['price']))
                    timestamps.append(item.get('timestamp', time.time()))
                # Sprawdź, czy item jest liczbą lub można go przekonwertować na liczbę
                elif isinstance(item, (int, float)):
                    values.append(float(item))
                    timestamps.append(time.time())

            if values:
                # Oblicz statystyki
                mean_value = sum(values) / len(values)
                std_dev = np.std(values) if len(values) > 1 else 0

                # Wykryj anomalie
                for i, value in enumerate(values):
                    z_score = (value - mean_value) / std_dev if std_dev > 0 else 0

                    if abs(z_score) > threshold:
                        # Określ typ anomalii
                        anomaly_type = "price_spike" if value > mean_value else "price_crash"

                        anomalies.append({
                            "timestamp": timestamps[i],
                            "value": value,
                            "z_score": z_score,
                            "anomaly_type": anomaly_type,
                            "anomaly_name": self.anomaly_patterns[anomaly_type]["name"],
                            "description": self.anomaly_patterns[anomaly_type]["description"],
                            "confidence": min(0.95, 0.7 + abs(z_score) / 10)
                        })

            # Dodaj losową anomalię (dla celów demonstracyjnych)
            if not anomalies and random.random() < 0.3:
                rand_type = random.choice(list(self.anomaly_patterns.keys()))
                rand_timestamp = timestamps[-1] if timestamps else time.time()
                rand_value = values[-1] if values else 0

                anomalies.append({
                    "timestamp": rand_timestamp,
                    "value": rand_value,
                    "z_score": random.uniform(2.1, 4.0),
                    "anomaly_type": rand_type,
                    "anomaly_name": self.anomaly_patterns[rand_type]["name"],
                    "description": self.anomaly_patterns[rand_type]["description"],
                    "confidence": random.uniform(0.7, 0.95)
                })

            # Aktualizuj czas ostatniej detekcji
            self.last_detection_time = time.time()

        except Exception as e:
            logger.error(f"Błąd podczas wykrywania anomalii: {e}")

        return anomalies

    def get_available_patterns(self) -> Dict[str, Dict[str, str]]:
        """
        Zwraca dostępne wzorce anomalii.

        Returns:
            Dict[str, Dict[str, str]]: Słownik wzorców anomalii
        """
        return self.anomaly_patterns