
"""
anomaly_detection.py
-------------------
Moduł odpowiedzialny za wykrywanie anomalii w danych rynkowych z wykorzystaniem różnych technik.

Klasa AnomalyDetectionModel obsługuje wykrywanie anomalii w danych cenowych, wolumenowych
i innych istotnych wskaźnikach rynkowych.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Konfiguracja logowania
logging.basicConfig(
    filename="logs/anomaly_detection.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class AnomalyDetectionModel:
    """
    Klasa do wykrywania anomalii w danych rynkowych.
    Zawiera różne metody do wykrywania nietypowych wzorców w danych.
    """

    def __init__(self, config=None):
        """
        Inicjalizacja modelu wykrywania anomalii.
        
        Args:
            config (dict, optional): Konfiguracja modelu. Defaults to None.
        """
        self.config = config or {}
        self.threshold = float(self.config.get("ANOMALY_THRESHOLD", 2.5))
        self.scaler = StandardScaler()
        self.models = {
            "isolation_forest": IsolationForest(
                contamination=0.05, random_state=42, n_jobs=-1
            ),
            "lof": LocalOutlierFactor(n_neighbors=20, contamination=0.05, n_jobs=-1),
        }
        logging.info("Zainicjalizowano model wykrywania anomalii")

    def detect_price_anomalies(self, price_data, method="zscore"):
        """
        Wykrywa anomalie w danych cenowych.
        
        Args:
            price_data (pd.Series/np.array): Dane cenowe do analizy
            method (str, optional): Metoda wykrywania anomalii. Defaults to "zscore".
                Dostępne metody: "zscore", "isolation_forest", "lof"
                
        Returns:
            pd.Series: Boolean maska wskazująca na anomalie
        """
        try:
            logging.info(f"Wykrywanie anomalii cenowych metodą {method}")
            
            if not isinstance(price_data, (pd.Series, np.ndarray)):
                if isinstance(price_data, pd.DataFrame) and "close" in price_data.columns:
                    price_data = price_data["close"]
                else:
                    logging.error("Nieprawidłowy format danych cenowych")
                    return np.zeros(len(price_data), dtype=bool)
            
            if method == "zscore":
                return self._detect_with_zscore(price_data)
            elif method == "isolation_forest":
                return self._detect_with_isolation_forest(price_data)
            elif method == "lof":
                return self._detect_with_lof(price_data)
            else:
                logging.warning(f"Nieznana metoda {method}, używam z-score")
                return self._detect_with_zscore(price_data)
                
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania anomalii cenowych: {e}")
            return np.zeros(len(price_data), dtype=bool)

    def detect_volume_anomalies(self, volume_data, threshold=None):
        """
        Wykrywa anomalie w danych wolumenowych.
        
        Args:
            volume_data (pd.Series/np.array): Dane wolumenowe do analizy
            threshold (float, optional): Próg odchylenia standardowego. Defaults to None.
                
        Returns:
            pd.Series: Boolean maska wskazująca na anomalie
        """
        try:
            threshold = threshold or self.threshold
            logging.info(f"Wykrywanie anomalii wolumenowych (próg={threshold})")
            
            # Obliczenie logarytmicznej zmiany wolumenu
            log_volume = np.log1p(volume_data)
            rolling_mean = pd.Series(log_volume).rolling(window=20).mean()
            rolling_std = pd.Series(log_volume).rolling(window=20).std()
            
            # Obliczenie z-score
            z_scores = (log_volume - rolling_mean) / rolling_std
            
            # Identyfikacja anomalii
            anomalies = abs(z_scores) > threshold
            return anomalies.fillna(False)
            
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania anomalii wolumenowych: {e}")
            return pd.Series(np.zeros(len(volume_data), dtype=bool))

    def detect_pattern_anomalies(self, price_data, window_size=20):
        """
        Wykrywa anomalie wzorców cenowych.
        
        Args:
            price_data (pd.DataFrame): Dane cenowe (OHLC)
            window_size (int, optional): Rozmiar okna analizy. Defaults to 20.
                
        Returns:
            pd.Series: Boolean maska wskazująca na anomalie
        """
        try:
            logging.info(f"Wykrywanie anomalii wzorców (okno={window_size})")
            
            # Przygotowanie cech dla modelu
            features = self._prepare_pattern_features(price_data, window_size)
            
            # Zastosowanie Isolation Forest
            model = IsolationForest(contamination=0.05, random_state=42)
            anomalies = model.fit_predict(features)
            
            # Konwersja wyników (-1: anomalia, 1: normalne)
            result = np.zeros(len(price_data), dtype=bool)
            result[window_size:] = (anomalies == -1)
            
            return pd.Series(result, index=price_data.index if hasattr(price_data, 'index') else None)
            
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania anomalii wzorców: {e}")
            return pd.Series(np.zeros(len(price_data), dtype=bool))

    def _detect_with_zscore(self, data):
        """Wykrywanie anomalii metodą z-score."""
        # Obliczenie średniej kroczącej i odchylenia standardowego
        data_series = pd.Series(data)
        rolling_mean = data_series.rolling(window=20).mean()
        rolling_std = data_series.rolling(window=20).std()
        
        # Obliczenie z-score
        z_scores = (data_series - rolling_mean) / rolling_std
        
        # Identyfikacja anomalii
        anomalies = abs(z_scores) > self.threshold
        return anomalies.fillna(False)

    def _detect_with_isolation_forest(self, data):
        """Wykrywanie anomalii metodą Isolation Forest."""
        # Przygotowanie danych
        X = np.array(data).reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Dopasowanie modelu
        model = self.models["isolation_forest"]
        predictions = model.fit_predict(X_scaled)
        
        # Konwersja wyników (-1: anomalia, 1: normalne)
        return predictions == -1

    def _detect_with_lof(self, data):
        """Wykrywanie anomalii metodą Local Outlier Factor."""
        # Przygotowanie danych
        X = np.array(data).reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Dopasowanie modelu
        model = self.models["lof"]
        predictions = model.fit_predict(X_scaled)
        
        # Konwersja wyników (-1: anomalia, 1: normalne)
        return predictions == -1

    def _prepare_pattern_features(self, price_data, window_size):
        """Przygotowuje cechy dla wykrywania anomalii we wzorcach."""
        features = []
        
        # Upewnienie się, że mamy dane OHLC
        if isinstance(price_data, pd.DataFrame):
            close_prices = price_data["close"] if "close" in price_data.columns else price_data.iloc[:, 0]
        else:
            close_prices = price_data
            
        close_prices = np.array(close_prices)
        
        # Utworzenie cech na podstawie okien czasowych
        for i in range(len(close_prices) - window_size):
            window = close_prices[i:i + window_size]
            normalized_window = (window - np.mean(window)) / np.std(window) if np.std(window) > 0 else window
            features.append(normalized_window)
            
        return np.array(features)

    def get_anomaly_statistics(self, data, anomalies):
        """
        Zwraca statystyki dotyczące wykrytych anomalii.
        
        Args:
            data (pd.Series/np.array): Analizowane dane
            anomalies (pd.Series/np.array): Maska anomalii
                
        Returns:
            dict: Słownik ze statystykami
        """
        try:
            data_array = np.array(data)
            anomaly_mask = np.array(anomalies)
            
            # Przefiltrowanie danych
            normal_data = data_array[~anomaly_mask]
            anomaly_data = data_array[anomaly_mask]
            
            if len(anomaly_data) == 0:
                return {
                    "anomaly_count": 0,
                    "anomaly_percentage": 0,
                    "stats": None
                }
                
            # Obliczenie statystyk
            stats = {
                "anomaly_count": len(anomaly_data),
                "anomaly_percentage": 100 * len(anomaly_data) / len(data_array),
                "stats": {
                    "normal_mean": np.mean(normal_data),
                    "normal_std": np.std(normal_data),
                    "anomaly_mean": np.mean(anomaly_data),
                    "anomaly_std": np.std(anomaly_data),
                    "min_anomaly": np.min(anomaly_data),
                    "max_anomaly": np.max(anomaly_data)
                }
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Błąd podczas obliczania statystyk anomalii: {e}")
            return {"error": str(e)}

# Przykład użycia modułu
if __name__ == "__main__":
    # Utworzenie przykładowych danych
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, 100)) + 100
    # Dodanie anomalii
    prices[30] += 10
    prices[60] -= 15
    
    df = pd.DataFrame({
        "timestamp": dates,
        "close": prices
    })
    df.set_index("timestamp", inplace=True)
    
    # Inicjalizacja modelu
    model = AnomalyDetectionModel()
    
    # Wykrywanie anomalii
    anomalies = model.detect_price_anomalies(df["close"])
    
    # Wyświetlenie wyników
    print(f"Wykryto {anomalies.sum()} anomalii cenowych")
    print(f"Anomalie wystąpiły w następujących dniach: {df.index[anomalies].strftime('%Y-%m-%d').tolist()}")
    
    # Statystyki anomalii
    stats = model.get_anomaly_statistics(df["close"], anomalies)
    print(f"Statystyki anomalii: {stats}")
