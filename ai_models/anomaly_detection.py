"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
from datetime import datetime

class AnomalyDetector:
    """
    Klasa wykrywająca anomalie w danych rynkowych.
    W bieżącej implementacji generuje przykładowe anomalie dla celów demonstracyjnych.
    """

    def __init__(self):
        """
        Inicjalizacja detektora anomalii.
        """
        self.threshold = 2.5  # Domyślny próg dla anomalii (z-score)
        self.last_update = datetime.now()
        self.detected_anomalies = []
        logging.info("Inicjalizacja detektora anomalii")

    def get_detected_anomalies(self):
        """
        Zwraca listę wykrytych anomalii.
        W tej implementacji generuje przykładowe anomalie dla demonstracji.

        Returns:
            list: Lista wykrytych anomalii
        """
        # Aktualizujemy listę anomalii dla celów demonstracyjnych
        now = datetime.now()

        # Losowo generujemy anomalie (przykład)
        if random.random() < 0.2:  # 20% szans na nową anomalię
            anomaly_types = [
                "Nagły wzrost wolumenu", 
                "Nietypowa zmienność ceny",
                "Korelacja międzyrynkowa",
                "Wzorzec świecowy",
                "Odbicie od poziomów wsparcia/oporu"
            ]

            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"]

            new_anomaly = {
                "type": random.choice(anomaly_types),
                "symbol": random.choice(symbols),
                "severity": round(random.uniform(0.3, 0.95), 2),
                "timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
                "details": "Wykryto anomalię na podstawie modelu statystycznego"
            }

            self.detected_anomalies.append(new_anomaly)

            # Ograniczamy listę do 5 ostatnich anomalii
            if len(self.detected_anomalies) > 5:
                self.detected_anomalies = self.detected_anomalies[-5:]

        return self.detected_anomalies


# Przykład użycia
if __name__ == "__main__":
    # Konfiguracja loggera dla bezpośredniego uruchomienia
    logging.basicConfig(level=logging.INFO)

    # Inicjalizacja i testowanie detektora
    detector = AnomalyDetector()
    anomalies = detector.get_detected_anomalies()

    print("Wykryte anomalie:")
    for anomaly in anomalies:
        print(anomaly)