
"""
anomaly_detection.py
--------------------
Moduł wykrywający anomalie w danych tradingowych.

Funkcjonalności:
- Wykrywanie anomalii w danych rynkowych
- Wsparcie dla różnych metod detekcji (z-score, isolation forest)
- Generowanie alertów przy wykryciu podejrzanych wzorców
"""

import logging
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Klasa wykrywająca anomalie w danych rynkowych przy użyciu różnych metod statystycznych.
    """
    def __init__(self, method: str = "z_score", threshold: float = 2.5, window_size: int = 20):
        """
        Inicjalizuje detektor anomalii.
        
        Args:
            method (str): Metoda detekcji ('z_score', 'isolation_forest', 'kmeans')
            threshold (float): Próg uznania obserwacji za anomalię
            window_size (int): Wielkość okna analizy
        """
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.historical_data = []
        self.detected_anomalies = []
        
        logger.info(f"AnomalyDetector zainicjalizowany (metoda: {method}, próg: {threshold})")
    
    def detect(self, data: Union[List, Dict, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie w danych.
        
        Args:
            data: Dane do analizy - lista, słownik lub tablica numpy
            
        Returns:
            List[Dict]: Lista wykrytych anomalii
        """
        # Konwersja różnych formatów danych do listy wartości
        values = self._convert_input_to_values(data)
        
        # Sprawdzenie czy mamy wystarczająco danych
        if len(values) < 3:
            logger.warning("Za mało danych do wykrycia anomalii")
            return []
        
        anomalies = []
        timestamps = [time.time()] * len(values) if isinstance(data, (list, np.ndarray)) else None
        
        # Analiza danych odpowiednią metodą
        if self.method == "z_score":
            anomalies = self._detect_with_zscore(values, timestamps)
        else:
            # Domyślnie używamy z-score jeśli metoda nie jest obsługiwana
            logger.warning(f"Metoda {self.method} nie jest zaimplementowana, użycie z-score")
            anomalies = self._detect_with_zscore(values, timestamps)
        
        # Dodaj wykryte anomalie do historii
        self.detected_anomalies.extend(anomalies)
        
        # Zachowaj tylko ostatnie 100 anomalii
        if len(self.detected_anomalies) > 100:
            self.detected_anomalies = self.detected_anomalies[-100:]
        
        return anomalies
    
    def _convert_input_to_values(self, data: Union[List, Dict, np.ndarray]) -> List[float]:
        """
        Konwertuje różne formaty danych wejściowych do listy wartości.
        
        Args:
            data: Dane wejściowe (lista, słownik, np.ndarray)
            
        Returns:
            List[float]: Lista wartości
        """
        if isinstance(data, list):
            # Jeśli to lista słowników, spróbuj wyciągnąć wartości
            if all(isinstance(item, dict) for item in data) and len(data) > 0:
                if 'value' in data[0]:
                    return [item.get('value', 0.0) for item in data]
                elif 'close' in data[0]:
                    return [item.get('close', 0.0) for item in data]
                else:
                    # Bierzemy pierwszą wartość liczbową z każdego słownika
                    return [next((v for v in item.values() if isinstance(v, (int, float))), 0.0) for item in data]
            # Jeśli to zwykła lista
            else:
                return [float(x) if isinstance(x, (int, float)) else 0.0 for x in data]
        
        elif isinstance(data, dict):
            # Jeśli słownik zawiera szeregi czasowe
            if any(key in data for key in ['close', 'open', 'high', 'low']):
                return data.get('close', [])
            # Inaczej bierzemy wszystkie wartości liczbowe
            else:
                return [v for v in data.values() if isinstance(v, (int, float))]
        
        elif isinstance(data, np.ndarray):
            # Konwersja z numpy array
            if data.ndim == 1:
                return data.tolist()
            elif data.ndim == 2:
                # Dla 2D bierzemy ostatnią kolumnę (zakładamy, że to ceny close)
                return data[:, -1].tolist()
        
        # W razie niepowodzenia zwracamy pustą listę
        logger.warning(f"Nieobsługiwany format danych: {type(data)}")
        return []
    
    def _detect_with_zscore(self, values: List[float], timestamps: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie metodą z-score.
        
        Args:
            values: Lista wartości do analizy
            timestamps: Lista timestampów dla każdej wartości
            
        Returns:
            List[Dict]: Lista wykrytych anomalii
        """
        if not timestamps:
            timestamps = [time.time()] * len(values)
        
        # Obliczamy statystyki
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 1.0
        
        # Zabezpieczenie przed dzieleniem przez zero
        if std == 0:
            std = 1.0
        
        anomalies = []
        
        # Szukamy anomalii
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std)
            
            if z_score > self.threshold:
                anomaly = {
                    "timestamp": timestamps[i],
                    "value": value,
                    "score": z_score,
                    "threshold": self.threshold,
                    "method": self.method,
                    "detection_time": time.time()
                }
                anomalies.append(anomaly)
                logger.info(f"Wykryto anomalię: wartość {value}, z-score {z_score:.2f}")
        
        return anomalies
    
    def predict(self, data: Union[List, Dict, np.ndarray]) -> Dict[str, Any]:
        """
        Przewiduje czy nowe dane zawierają anomalie.
        
        Args:
            data: Dane do analizy
            
        Returns:
            Dict: Wynik predykcji
        """
        anomalies = self.detect(data)
        
        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "detection_method": self.method,
            "threshold": self.threshold
        }
    
    def get_detected_anomalies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Zwraca listę wykrytych anomalii.
        
        Args:
            limit: Maksymalna liczba zwracanych anomalii
            
        Returns:
            List[Dict]: Lista wykrytych anomalii
        """
        # Sortuj anomalie od najnowszych
        sorted_anomalies = sorted(
            self.detected_anomalies, 
            key=lambda x: x["detection_time"], 
            reverse=True
        )
        return sorted_anomalies[:limit]
    
    def clear_anomalies(self) -> int:
        """
        Czyści listę wykrytych anomalii i zwraca ich liczbę.
        
        Returns:
            int: Liczba usuniętych anomalii
        """
        count = len(self.detected_anomalies)
        self.detected_anomalies = []
        return count
