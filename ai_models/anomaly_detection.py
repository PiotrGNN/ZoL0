"""
anomaly_detection.py
-------------------
Moduł odpowiedzialny za wykrywanie anomalii w danych finansowych
z wykorzystaniem różnych algorytmów uczenia maszynowego.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# Konfiguracja loggera
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Klasa wykrywająca anomalie w danych rynkowych.
    Wykorzystuje różne metody: statystyczne, ML, itp.
    """

    def __init__(self, method="isolation_forest", threshold=3.0, contamination=0.05):
        """
        Inicjalizacja detektora anomalii.

        Args:
            method (str): Metoda detekcji ('isolation_forest', 'zscore', 'kmeans')
            threshold (float): Próg dla metod statystycznych (np. z-score)
            contamination (float): Parametr dla Isolation Forest (oczekiwany % anomalii)
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.model = None
        self.is_fitted = False
        logging.info(f"Inicjalizacja detektora anomalii z metodą: {method}")

        if method == "isolation_forest":
            self.model = IsolationForest(contamination=contamination, random_state=42)
        elif method not in ["zscore", "kmeans"]:
            logging.warning(f"Nieznana metoda detekcji anomalii: {method}, używam domyślnej (isolation_forest)")
            self.method = "isolation_forest"
            self.model = IsolationForest(contamination=contamination, random_state=42)

    def __str__(self):
        """Zwraca opis detektora anomalii."""
        status = "Wytrenowany" if self.is_fitted else "Gotowy do użycia"
        return f"Anomaly Detector - Metoda: {self.method}\nParametry: Próg={self.threshold}, Zanieczyszczenie={self.contamination}\nStatus: {status}"

    def detect(self, data):
        """
        Wykrywa anomalie w danych.

        Args:
            data (pd.DataFrame lub pd.Series): Dane do analizy

        Returns:
            pd.Series: Indeksy anomalii (True) lub wartości normalne (False)
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, pd.Series) and len(data) == 0):
            logging.warning("Przekazano puste dane do wykrywania anomalii")
            return pd.Series([], dtype=bool)

        if isinstance(data, pd.DataFrame):
            # Jeśli dataframe, sprawdzamy czy jest 1-kolumnowy
            if data.shape[1] == 1:
                data = data.iloc[:, 0]
            else:
                # Jeśli wielowymiarowy, używamy Isolation Forest
                if self.method != "isolation_forest":
                    logging.warning("Dla danych wielowymiarowych używam Isolation Forest.")
                    return self._detect_isolation_forest(data)

        try:
            if self.method == "zscore":
                return self._detect_zscore(data)
            elif self.method == "kmeans":
                return self._detect_kmeans(data)
            else:  # isolation_forest
                return self._detect_isolation_forest(data)
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania anomalii: {str(e)}")
            return pd.Series([False] * len(data), index=data.index)

    def get_anomaly_score(self, data):
        """
        Zwraca wynik anomalii dla każdego punktu danych.

        Args:
            data (pd.DataFrame lub pd.Series): Dane do analizy

        Returns:
            pd.Series: Wyniki anomalii, wyższe wartości oznaczają większe prawdopodobieństwo anomalii
        """
        if self.method == "isolation_forest":
            if isinstance(data, pd.Series):
                data = data.values.reshape(-1, 1)
            if not self.is_fitted:
                self.model.fit(data)
                self.is_fitted = True
            scores = -self.model.score_samples(data)  # Wyższe wartości = większe prawdopodobieństwo anomalii
            return pd.Series(scores, index=data.index if hasattr(data, 'index') else None)
        elif self.method == "zscore":
            return pd.Series(np.abs(stats.zscore(data)), index=data.index)
        else:
            logging.warning(f"Metoda {self.method} nie obsługuje bezpośrednio wyników anomalii, zwracam wartości z-score")
            return pd.Series(np.abs(stats.zscore(data)), index=data.index)


    def _detect_isolation_forest(self, data):
        if not self.is_fitted:
            self.model.fit(data)
            self.is_fitted = True
        predictions = self.model.predict(data)
        return pd.Series(predictions == -1, index=data.index)

    def _detect_zscore(self, data):
        z_scores = np.abs(stats.zscore(data))
        return pd.Series(z_scores > self.threshold, index=data.index)

    def _detect_kmeans(self, data):
        # Implementacja k-means tutaj (zostawiam jako ćwiczenie)
        # ...
        return pd.Series([False] * len(data), index=data.index)  # Tymczasowe


    def info(self) -> None:
        """Wyświetla informacje o aktualnej konfiguracji detektora anomalii."""
        print(f"Anomaly Detector - Metoda: {self.method}")
        print(f"Parametry: Próg={self.threshold}, Zanieczyszczenie={self.contamination}")
        print("Status: Gotowy do użycia")
        logger.info(f"Wywołano informacje o detektorze anomalii (metoda: {self.method})")


# Przykład użycia
if __name__ == "__main__":
    # Konfiguracja loggera dla bezpośredniego uruchomienia
    logging.basicConfig(level=logging.INFO)

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
    detector = AnomalyDetector(method="isolation_forest", contamination=0.01)
    detector.train(data)
    anomalies = detector.detect(data)

    print(f"Wykryto {anomalies.sum()} anomalii")
    print("Indeksy anomalii:", anomalies[anomalies].index.tolist())