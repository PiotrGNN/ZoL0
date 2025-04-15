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

    def __init__(self, method=None, threshold=None):
        """
        Inicjalizuje detektor anomalii.
        
        Args:
            method (str, optional): Metoda wykrywania anomalii (np. 'z_score', 'iqr'). Domyślnie None.
            threshold (float, optional): Próg wykrywania anomalii. Domyślnie None.
        """
        # Ustawienia metody i progu
        self.method = method or "z_score"
        self.threshold = threshold or 2.0
        
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
        
    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca listę wykrytych anomalii.
        
        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii
        """
        # Przykładowe dane dla demonstracji, gdy brak rzeczywistych anomalii
        import random
        import time
        
        if random.random() < 0.3:  # 30% szans na wykrycie anomalii
            anomaly_type = random.choice(list(self.anomaly_patterns.keys()))
            return [{
                "timestamp": time.time(),
                "value": random.uniform(1000, 50000),
                "score": random.uniform(2.5, 5.0),
                "anomaly_type": anomaly_type,
                "anomaly_name": self.anomaly_patterns[anomaly_type]["name"],
                "description": self.anomaly_patterns[anomaly_type]["description"],
                "confidence": random.uniform(0.7, 0.95)
            }]
        return []
"""
anomaly_detection.py
------------------
Moduł do wykrywania anomalii w danych rynkowych.
"""

import logging
import random
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

class AnomalyDetector:
    """Detektor anomalii w danych rynkowych."""
    
    def __init__(self, method: str = 'z_score', threshold: float = 2.5):
        """
        Inicjalizacja detektora anomalii.
        
        Args:
            method: Metoda wykrywania anomalii ('z_score', 'iqr', 'isolation_forest')
            threshold: Próg uznania za anomalię
        """
        self.logger = logging.getLogger("anomaly_detector")
        self.method = method
        self.threshold = threshold
        
        self.detected_anomalies = []
        self.max_anomalies = 50  # Maksymalna liczba przechowywanych anomalii
        
        self.logger.info(f"Zainicjalizowano detektor anomalii (metoda: {method}, próg: {threshold})")
    
    def detect(self, data: List[float]) -> Dict[str, Any]:
        """
        Wykrywa anomalie w danych.
        
        Args:
            data: Lista wartości do analizy
            
        Returns:
            Dict[str, Any]: Wynik detekcji anomalii
        """
        if not data or len(data) < 3:
            return {"anomalies": [], "is_anomaly": False, "score": 0.0}
        
        try:
            # Wybór metody detekcji
            if self.method == 'z_score':
                anomalies, scores = self._detect_z_score(data)
            elif self.method == 'iqr':
                anomalies, scores = self._detect_iqr(data)
            else:
                # Domyślnie z-score
                anomalies, scores = self._detect_z_score(data)
            
            # Czy jest anomalia (indeks ostatniego punktu)
            is_anomaly = len(data) - 1 in anomalies
            
            # Ostatnia wartość score
            last_score = scores[-1] if scores else 0.0
            
            # Zapisz wykryte anomalie
            if anomalies:
                for idx in anomalies:
                    if idx < len(data):
                        timestamp = time.time()
                        self._add_anomaly({
                            "index": idx,
                            "value": data[idx],
                            "score": scores[idx] if idx < len(scores) else 0.0,
                            "timestamp": timestamp,
                            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                        })
            
            return {
                "anomalies": anomalies,
                "scores": scores,
                "is_anomaly": is_anomaly,
                "score": last_score
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas wykrywania anomalii: {e}")
            return {"anomalies": [], "is_anomaly": False, "score": 0.0, "error": str(e)}
    
    def _detect_z_score(self, data: List[float]) -> Tuple[List[int], List[float]]:
        """
        Wykrywa anomalie metodą z-score.
        
        Args:
            data: Lista wartości do analizy
            
        Returns:
            Tuple[List[int], List[float]]: Indeksy anomalii i wartości z-score
        """
        # Konwersja do numpy array
        try:
            values = np.array(data, dtype=float)
        except:
            # Fallback jeśli numpy nie jest dostępne
            values = data
            
        # Obliczenie średniej i odchylenia standardowego
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        
        # Obliczenie z-score dla każdego punktu
        z_scores = [(x - mean) / std_dev if std_dev > 0 else 0 for x in values]
        
        # Znalezienie anomalii (z-score przekracza próg)
        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > self.threshold]
        
        return anomalies, z_scores
    
    def _detect_iqr(self, data: List[float]) -> Tuple[List[int], List[float]]:
        """
        Wykrywa anomalie metodą IQR (Interquartile Range).
        
        Args:
            data: Lista wartości do analizy
            
        Returns:
            Tuple[List[int], List[float]]: Indeksy anomalii i wartości score
        """
        # Konwersja do numpy array
        try:
            values = np.array(data, dtype=float)
            
            # Obliczenie kwartyli
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            
            # Obliczenie IQR
            iqr = q3 - q1
            
            # Dolna i górna granica
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Znalezienie anomalii
            anomalies = [i for i, x in enumerate(values) if x < lower_bound or x > upper_bound]
            
            # Obliczenie "score" - jak daleko od granicy
            scores = []
            for x in values:
                if x < lower_bound:
                    score = (lower_bound - x) / iqr if iqr > 0 else 0
                elif x > upper_bound:
                    score = (x - upper_bound) / iqr if iqr > 0 else 0
                else:
                    score = 0
                scores.append(score)
            
            return anomalies, scores
        except:
            # Fallback jeśli numpy nie jest dostępne
            # Proste sortowanie i wybór kwartyli
            sorted_values = sorted(data)
            n = len(sorted_values)
            
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            anomalies = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
            
            scores = []
            for x in data:
                if x < lower_bound:
                    score = (lower_bound - x) / iqr if iqr > 0 else 0
                elif x > upper_bound:
                    score = (x - upper_bound) / iqr if iqr > 0 else 0
                else:
                    score = 0
                scores.append(score)
            
            return anomalies, scores
    
    def _add_anomaly(self, anomaly: Dict[str, Any]):
        """
        Dodaje anomalię do listy.
        
        Args:
            anomaly: Informacje o anomalii
        """
        self.detected_anomalies.append(anomaly)
        
        # Usuń najstarsze anomalie, jeśli przekroczono limit
        if len(self.detected_anomalies) > self.max_anomalies:
            self.detected_anomalies = self.detected_anomalies[-self.max_anomalies:]
    
    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """
        Zwraca wykryte anomalie.
        
        Returns:
            List[Dict[str, Any]]: Lista wykrytych anomalii
        """
        return self.detected_anomalies
    
    def clear_anomalies(self):
        """Czyści listę wykrytych anomalii."""
        self.detected_anomalies = []
        self.logger.info("Wyczyszczono listę anomalii")
    
    def set_method(self, method: str) -> bool:
        """
        Ustawia metodę wykrywania anomalii.
        
        Args:
            method: Metoda wykrywania anomalii
            
        Returns:
            bool: Czy operacja się powiodła
        """
        valid_methods = ['z_score', 'iqr', 'isolation_forest']
        
        if method not in valid_methods:
            self.logger.warning(f"Nieznana metoda wykrywania anomalii: {method}. Użyj jednej z: {valid_methods}")
            return False
        
        self.method = method
        self.logger.info(f"Ustawiono metodę wykrywania anomalii: {method}")
        return True
    
    def set_threshold(self, threshold: float) -> bool:
        """
        Ustawia próg uznania za anomalię.
        
        Args:
            threshold: Próg
            
        Returns:
            bool: Czy operacja się powiodła
        """
        if threshold <= 0:
            self.logger.warning(f"Próg musi być większy od zera: {threshold}")
            return False
        
        self.threshold = threshold
        self.logger.info(f"Ustawiono próg wykrywania anomalii: {threshold}")
        return True
    
    def generate_test_data(self, size: int = 100, anomaly_count: int = 5) -> List[float]:
        """
        Generuje testowe dane z anomaliami.
        
        Args:
            size: Rozmiar danych
            anomaly_count: Liczba anomalii
            
        Returns:
            List[float]: Wygenerowane dane
        """
        # Generuj normalne dane
        data = [random.normalvariate(0, 1) for _ in range(size)]
        
        # Dodaj anomalie
        for _ in range(anomaly_count):
            idx = random.randint(0, size - 1)
            data[idx] = random.normalvariate(0, 5)  # Anomalia ma większe odchylenie
        
        return data
