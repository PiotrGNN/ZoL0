"""
anomaly_detection.py
-------------------
Moduł do wykrywania anomalii rynkowych za pomocą technik statystycznych i ML.

Funkcjonalności:
- Detekcja nietypowych wzorców cenowych
- Identyfikacja odstających wolumenów
- Wykrywanie manipulacji rynkowych
- Alerty w czasie rzeczywistym
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("anomaly_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Klasa implementująca wykrywanie anomalii rynkowych."""

    def __init__(self, sensitivity: float = 0.05, window_size: int = 100):
        """
        Inicjalizuje detektor anomalii.

        Args:
            sensitivity: Czułość detektora (0.01-0.1)
            window_size: Rozmiar okna analizy
        """
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=self.sensitivity,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        logger.info(f"✅ Inicjalizacja detektora anomalii (czułość: {sensitivity})")

    @staticmethod
    def check_dependencies() -> bool:
        """Sprawdza, czy wszystkie zależności są dostępne."""
        try:
            import numpy
            import pandas
            import sklearn
            return True
        except ImportError:
            return False

    def train(self, data: pd.DataFrame) -> None:
        """
        Trenuje model na danych historycznych.

        Args:
            data: DataFrame z danymi historycznymi
        """
        try:
            if data.empty or len(data) < self.window_size:
                logger.warning("⚠️ Za mało danych do treningu detektora anomalii")
                return

            features = self._extract_features(data)
            scaled_features = self.scaler.fit_transform(features)

            logger.info(f"🧠 Trening modelu na {len(features)} próbkach")
            self.model.fit(scaled_features)
            self.is_trained = True
            logger.info("✅ Model detektora anomalii wytrenowany")
        except Exception as e:
            logger.error(f"❌ Błąd podczas treningu detektora anomalii: {e}")
            self.is_trained = False

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Wykrywa anomalie w danych.

        Args:
            data: DataFrame z danymi do analizy

        Returns:
            DataFrame z flagami anomalii
        """
        if not self.is_trained:
            logger.warning("⚠️ Model nie jest wytrenowany. Uruchom train() najpierw.")
            return pd.DataFrame()

        try:
            features = self._extract_features(data)
            scaled_features = self.scaler.transform(features)

            # -1 oznacza anomalię, 1 oznacza normalną próbkę
            predictions = self.model.predict(scaled_features)
            scores = self.model.decision_function(scaled_features)

            # Dodajemy wyniki detekcji do oryginalnych danych
            result = data.copy()
            result['is_anomaly'] = [1 if pred == -1 else 0 for pred in predictions]
            result['anomaly_score'] = scores

            num_anomalies = sum(result['is_anomaly'])
            logger.info(f"🔍 Wykryto {num_anomalies} anomalii w {len(data)} próbkach")

            return result
        except Exception as e:
            logger.error(f"❌ Błąd podczas detekcji anomalii: {e}")
            return pd.DataFrame()

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Wyodrębnia cechy z danych surowych.

        Args:
            data: DataFrame z danymi surowymi

        Returns:
            Tablica cech do analizy
        """
        # Przekształcamy dane do formatu odpowiedniego dla algorytmu
        # Przykładowe cechy: zwroty, zmienność, objętość
        features = []

        # Podstawowe funkcje
        if 'close' in data.columns:
            # Zwroty procentowe
            returns = data['close'].pct_change().fillna(0).values.reshape(-1, 1)
            features.append(returns)

            # Zmienność (rolling)
            volatility = data['close'].pct_change().rolling(window=20).std().fillna(0).values.reshape(-1, 1)
            features.append(volatility)

        # Cechy wolumenowe
        if 'volume' in data.columns:
            # Wolumen znormalizowany
            volume = data['volume'].values.reshape(-1, 1)
            volume_ma = data['volume'].rolling(window=20).mean().fillna(data['volume'].mean()).values.reshape(-1, 1)
            volume_ratio = (volume / volume_ma)
            features.append(volume_ratio)

        # Spready, jeśli dostępne
        if all(col in data.columns for col in ['high', 'low']):
            spread = (data['high'] - data['low']).values.reshape(-1, 1)
            features.append(spread)

        # Łączymy wszystkie cechy
        if features:
            return np.hstack(features)
        else:
            # Jeśli nie ma odpowiednich kolumn, używamy wszystkich dostępnych danych numerycznych
            return data.select_dtypes(include=[np.number]).values

# Przykład użycia (dla testów)
if __name__ == "__main__":
    # Tworzymy przykładowe dane
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
    prices = np.cumsum(np.random.normal(0, 1, 1000)) + 1000

    # Dodajemy kilka anomalii
    prices[500:510] += 50
    prices[700:705] -= 30
    prices[900] += 100

    volumes = np.random.normal(1000, 100, 1000)
    volumes[500:510] *= 5

    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': volumes,
        'high': prices + np.random.normal(0, 5, 1000),
        'low': prices - np.random.normal(0, 5, 1000)
    })

    # Inicjalizujemy i trenujemy detektor
    detector = AnomalyDetector(sensitivity=0.03)
    train_data = data.iloc[:800]
    test_data = data.iloc[800:]

    detector.train(train_data)
    results = detector.detect(test_data)

    if not results.empty:
        print(f"Wykryto {results['is_anomaly'].sum()} anomalii w {len(results)} próbkach")
        anomalies = results[results['is_anomaly'] == 1]
        if not anomalies.empty:
            print("\nPrzyklady wykrytych anomalii:")
            print(anomalies[['timestamp', 'close', 'volume', 'anomaly_score']].head())