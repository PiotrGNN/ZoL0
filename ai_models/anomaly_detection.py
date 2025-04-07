"""
anomaly_detection.py
--------------------
Moduł do wykrywania anomalii w danych rynkowych przy użyciu różnych metod statystycznych i algorytmów ML.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Any, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/anomaly_detector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Klasa implementująca wykrywanie anomalii w danych rynkowych.
    Wspiera różne metody wykrywania, w tym statystyczne i oparte na ML.
    """

    def __init__(self, method: str = "isolation_forest", config: Optional[Dict[str, Any]] = None):
        """
        Inicjalizacja detektora anomalii.

        Args:
            method (str): Metoda wykrywania anomalii (isolation_forest, z_score, iqr, etc.)
            config (dict, optional): Dodatkowa konfiguracja dla wybranej metody
        """
        self.method = method
        self.config = config or {}
        self.detected_anomalies = []
        self.last_analysis_time = None

        logger.info(f"Inicjalizacja detektora anomalii z metodą: {method}")

        # Tutaj może być inicjalizacja konkretnego modelu na podstawie metody
        # W wersji demonstracyjnej używamy prostych metod statystycznych

    def detect(self, data: Union[pd.DataFrame, np.ndarray, List[float]]) -> Dict[str, Any]:
        """
        Wykrywa anomalie w podanych danych.

        Args:
            data: Dane do analizy (DataFrame, ndarray lub lista wartości)

        Returns:
            dict: Wynik analizy z informacjami o wykrytych anomaliach
        """
        if isinstance(data, list):
            data = np.array(data)

        try:
            if self.method == "z_score":
                anomalies = self._detect_z_score(data)
            elif self.method == "isolation_forest":
                anomalies = self._detect_isolation_forest(data)
            elif self.method == "iqr":
                anomalies = self._detect_iqr(data)
            else:
                logger.warning(f"Nieznana metoda wykrywania anomalii: {self.method}, używam z_score")
                anomalies = self._detect_z_score(data)

            # Zapisanie wyników
            self.last_analysis_time = datetime.now()

            # Dodanie nowych anomalii do listy (max 100 ostatnich)
            for anomaly in anomalies:
                self.detected_anomalies.append({
                    "timestamp": self.last_analysis_time.isoformat(),
                    "value": float(anomaly["value"]) if "value" in anomaly else None,
                    "score": float(anomaly["score"]) if "score" in anomaly else None,
                    "details": anomaly.get("details", "")
                })

            # Ograniczenie liczby przechowywanych anomalii
            if len(self.detected_anomalies) > 100:
                self.detected_anomalies = self.detected_anomalies[-100:]

            return {
                "method": self.method,
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies,
                "analysis_time": self.last_analysis_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Błąd podczas wykrywania anomalii: {str(e)}")
            return {
                "method": self.method,
                "anomalies_detected": 0,
                "anomalies": [],
                "error": str(e),
                "analysis_time": datetime.now().isoformat()
            }

    def _detect_z_score(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie używając metody Z-score.

        Args:
            data: Dane do analizy

        Returns:
            list: Lista znalezionych anomalii
        """
        threshold = self.config.get("threshold", 3.0)

        # Obliczanie Z-score
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return []

        z_scores = np.abs((data - mean) / std)
        anomalies = []

        for i, z in enumerate(z_scores):
            if z > threshold:
                anomalies.append({
                    "index": i,
                    "value": float(data[i]),
                    "z_score": float(z),
                    "score": float(z),
                    "details": f"Z-score: {z:.2f} (próg: {threshold})"
                })

        return anomalies

    def _detect_isolation_forest(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie używając metody Isolation Forest.
        W tej wersji demonstracyjnej symulujemy działanie tej metody.

        Args:
            data: Dane do analizy

        Returns:
            list: Lista znalezionych anomalii
        """
        try:
            # Symulacja wyników dla celów demonstracyjnych
            # W rzeczywistej implementacji użylibyśmy sklearn.ensemble.IsolationForest

            # Uproszczona symulacja - używamy Z-score z mniejszym progiem jako przybliżenie
            anomalies = self._detect_z_score(data)

            # Dodanie informacji specyficznych dla Isolation Forest
            for anomaly in anomalies:
                anomaly["details"] = f"Isolation Forest score: {anomaly['score']:.4f}"

            return anomalies

        except Exception as e:
            logger.error(f"Błąd podczas wykrywania anomalii metodą Isolation Forest: {str(e)}")
            return []

    def _detect_iqr(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """
        Wykrywa anomalie używając metody IQR (Interquartile Range).

        Args:
            data: Dane do analizy

        Returns:
            list: Lista znalezionych anomalii
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomalies = []

        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                score = abs((value - np.median(data)) / iqr) if iqr > 0 else 0
                anomalies.append({
                    "index": i,
                    "value": float(value),
                    "score": float(score),
                    "details": f"Wartość {value:.2f} poza zakresem IQR: ({lower_bound:.2f}, {upper_bound:.2f})"
                })

        return anomalies

    def get_detected_anomalies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Zwraca listę ostatnio wykrytych anomalii.

        Args:
            limit (int): Maksymalna liczba zwracanych anomalii

        Returns:
            list: Lista ostatnich anomalii
        """
        anomalies = sorted(
            self.detected_anomalies, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        return anomalies[:limit]

    def analyze(self, data: pd.DataFrame, column: str = "close") -> Dict[str, Any]:
        """
        Analizuje dane cenowe pod kątem anomalii.

        Args:
            data (DataFrame): Dane cenowe
            column (str): Kolumna do analizy

        Returns:
            dict: Wyniki analizy
        """
        if column not in data.columns:
            logger.error(f"Kolumna {column} nie istnieje w danych")
            return {
                "success": False,
                "error": f"Kolumna {column} nie istnieje w danych",
                "anomalies": []
            }

        values = data[column].values
        detection_result = self.detect(values)

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "analysis": detection_result
        }

    def reset(self) -> None:
        """
        Resetuje stan detektora anomalii.
        """
        self.detected_anomalies = []
        self.last_analysis_time = None
        logger.info(f"Reset detektora anomalii z metodą: {self.method}")


# Przykład użycia
if __name__ == "__main__":
    # Generowanie przykładowych danych
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'price': np.random.normal(100, 10, n_samples),
        'volume': np.random.exponential(1000, n_samples)
    })

    # Dodanie kilku anomalii
    anomaly_indices = np.random.choice(n_samples, 10, replace=False)
    data.loc[anomaly_indices, 'price'] = np.random.normal(150, 20, 10)
    data.loc[anomaly_indices, 'volume'] = np.random.exponential(5000, 10)

    # Inicjalizacja i testowanie detektora
    detector = AnomalyDetector(method="isolation_forest", config={"threshold": 2.5})
    result = detector.analyze(data, column='price')
    print(result)
    result = detector.analyze(data, column='volume')
    print(result)
    print(detector.get_detected_anomalies())

    detector.reset()
    print(detector.get_detected_anomalies())