
"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
from typing import Dict, List, Any, Optional
import os

class AnomalyDetector:
    """
    Klasa wykrywająca anomalie w danych rynkowych.
    """
    
    def __init__(self, method: str = "isolation_forest", threshold: float = 2.5):
        """
        Inicjalizuje detektor anomalii.
        
        Parameters:
            method (str): Metoda detekcji anomalii ('isolation_forest', 'one_class_svm', 'lof').
            threshold (float): Próg wykrywania anomalii.
        """
        self.method = method
        self.threshold = threshold
        self.detected_anomalies = []
        
        # Konfiguracja logowania
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "anomaly_detector.log")
        
        self.logger = logging.getLogger("anomaly_detector")
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Inicjalizacja detektora anomalii z metodą: {method}")
    
    def detect(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie w danych.
        
        Parameters:
            data (List[Dict[str, Any]]): Dane do analizy.
            
        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        try:
            # Symulacja detekcji anomalii (w rzeczywistym systemie powinno używać faktycznych algorytmów)
            anomalies = []
            
            if not data:
                self.logger.warning("Brak danych do analizy anomalii")
                return []
            
            # Symulujemy wykrywanie anomalii
            for i, point in enumerate(data):
                # Symulacja anomalii: 10% szans na anomalię
                if random.random() < 0.1:
                    anomaly = {
                        "index": i,
                        "timestamp": point.get("timestamp", f"point_{i}"),
                        "value": point.get("value", 0),
                        "score": random.uniform(self.threshold, self.threshold * 2),
                        "type": random.choice(["price_spike", "volume_anomaly", "pattern_break"]),
                        "severity": random.choice(["low", "medium", "high"])
                    }
                    anomalies.append(anomaly)
            
            self.detected_anomalies = anomalies
            self.logger.info(f"Wykryto {len(anomalies)} anomalii")
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Błąd podczas detekcji anomalii: {e}")
            return []
    
    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca listę wszystkich wykrytych anomalii.
        
        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii.
        """
        # Jeśli nie wykryto żadnych anomalii, zwracamy przykładowe dla celów demonstracyjnych
        if not self.detected_anomalies:
            self.logger.info("Wywołano informacje o detektorze anomalii (metoda: %s)", self.method)
            return [
                {
                    "timestamp": "2025-04-07 10:05:00",
                    "symbol": "BTC/USDT",
                    "value": 78345.5,
                    "score": 3.2,
                    "type": "price_spike",
                    "severity": "high"
                },
                {
                    "timestamp": "2025-04-07 11:15:00",
                    "symbol": "ETH/USDT",
                    "value": 4456.75,
                    "score": 2.7,
                    "type": "volume_anomaly",
                    "severity": "medium"
                }
            ]
        return self.detected_anomalies
    
    def set_threshold(self, new_threshold: float) -> None:
        """
        Ustawia nowy próg wykrywania anomalii.
        
        Parameters:
            new_threshold (float): Nowy próg wykrywania anomalii.
        """
        if new_threshold <= 0:
            self.logger.warning(f"Nieprawidłowy próg: {new_threshold}. Musi być dodatni.")
            return
        
        self.threshold = new_threshold
        self.logger.info(f"Zaktualizowano próg detekcji anomalii: {new_threshold}")
    
    def change_method(self, new_method: str) -> bool:
        """
        Zmienia metodę detekcji anomalii.
        
        Parameters:
            new_method (str): Nowa metoda detekcji anomalii.
            
        Returns:
            bool: True jeśli zmiana się powiodła, False w przeciwnym razie.
        """
        valid_methods = ["isolation_forest", "one_class_svm", "lof", "dbscan", "autoencoder"]
        
        if new_method not in valid_methods:
            self.logger.warning(f"Nieprawidłowa metoda: {new_method}. Dozwolone metody: {valid_methods}")
            return False
        
        self.method = new_method
        self.logger.info(f"Zmieniono metodę detekcji anomalii na: {new_method}")
        return True
