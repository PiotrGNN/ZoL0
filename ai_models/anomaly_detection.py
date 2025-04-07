"""
anomaly_detection.py
--------------------
Moduł do wykrywania anomalii w danych rynkowych przy użyciu różnych metod statystycznych i algorytmów ML.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Any

# Konfiguracja logowania - retained from original
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/anomaly_detector.log", mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detektor anomalii rynkowych."""

    def __init__(self):
        """Inicjalizacja detektora anomalii."""
        self.last_update = datetime.now()
        self.detected_anomalies = []
        logger.info("Zainicjalizowano detektor anomalii")

    def detect(self, data: Any = None) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie w danych rynkowych.
        W wersji demonstracyjnej generuje losowe anomalie.

        Parameters:
            data: Dane do analizy.

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        # Symulacja wykrywania anomalii
        anomaly_types = ["Price Spike", "Volume Surge", "Volatility Increase", "Pattern Break"]
        anomaly_count = random.randint(0, 3)

        anomalies = []
        for _ in range(anomaly_count):
            anomaly_type = random.choice(anomaly_types)
            score = random.uniform(2.0, 5.0)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            anomaly = {
                "type": anomaly_type,
                "score": round(score, 2),
                "timestamp": timestamp,
                "description": f"Wykryto {anomaly_type.lower()} z wynikiem {score:.2f}"
            }
            anomalies.append(anomaly)

        self.detected_anomalies = anomalies
        self.last_update = datetime.now()

        if anomalies:
            logger.info(f"Wykryto {len(anomalies)} anomalii")

        return anomalies

    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca listę ostatnio wykrytych anomalii.

        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        # Odśwież anomalie co jakiś czas
        if random.random() > 0.7:
            self.detect()
        return self.detected_anomalies

if __name__ == "__main__":
    detector = AnomalyDetector()
    anomalies = detector.detect()
    print(f"Wykryto {len(anomalies)} anomalii")
    for anomaly in anomalies:
        print(f"- {anomaly['type']}: {anomaly['description']}")
"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
from datetime import datetime, timedelta

class AnomalyDetector:
    """
    Klasa do wykrywania anomalii w danych rynkowych.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicjalizacja detektora anomalii")
        self.detected_anomalies = []
        
    def detect_anomalies(self, data=None):
        """
        Wykrywa anomalie w podanych danych.
        
        Args:
            data (dict, optional): Dane do analizy. Defaults to None.
            
        Returns:
            list: Lista wykrytych anomalii
        """
        # Symulowana detekcja anomalii
        if random.random() < 0.3:  # 30% szansa na wykrycie anomalii
            anomaly = {
                "id": len(self.detected_anomalies) + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": random.choice(["Price Spike", "Volume Anomaly", "Pattern Break"]),
                "symbol": random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
                "severity": random.choice(["Low", "Medium", "High"]),
                "description": "Unusual market behavior detected"
            }
            self.detected_anomalies.append(anomaly)
            self.logger.info(f"Wykryto anomalię: {anomaly['type']} na {anomaly['symbol']}")
        
        # Limituj listę anomalii do 10 najnowszych
        if len(self.detected_anomalies) > 10:
            self.detected_anomalies = self.detected_anomalies[-10:]
            
        return self.detected_anomalies
    
    def get_detected_anomalies(self):
        """
        Zwraca listę wykrytych anomalii.
        
        Returns:
            list: Lista wykrytych anomalii
        """
        # Symulacja wykrywania anomalii przy każdym zapytaniu
        self.detect_anomalies()
        return self.detected_anomalies
