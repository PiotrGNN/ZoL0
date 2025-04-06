
"""
anomaly_detection.py
-----------------
Moduł do wykrywania anomalii rynkowych za pomocą różnych metod.

Główne funkcjonalności:
- Izolacja lasów (Isolation Forest) do wykrywania nietypowych wzorców cenowych
- Analiza odchyleń od średniej kroczącej
- Wykrywanie gwałtownych zmian wolumenu
- Integracja z systemem powiadomień
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Konfiguracja logowania
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Klasa do wykrywania anomalii rynkowych za pomocą różnych technik
    statystycznych i uczenia maszynowego.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicjalizuje detektor anomalii z podaną konfiguracją.
        
        Args:
            config: Słownik z konfiguracją, może zawierać:
                - threshold: próg dla anomalii (domyślnie 2.5)
                - window_size: rozmiar okna dla średnich kroczących (domyślnie 20)
                - contamination: parametr dla Isolation Forest (domyślnie 0.05)
        """
        self.config = config or {}
        self.threshold = self.config.get("threshold", 2.5)
        self.window_size = self.config.get("window_size", 20)
        self.contamination = self.config.get("contamination", 0.05)
        
        # Inicjalizacja modelu Isolation Forest
        self.isolation_forest = None
        
        logger.info(f"AnomalyDetector zainicjowany z progiem {self.threshold} i rozmiarem okna {self.window_size}")
    
    @staticmethod
    def check_dependencies() -> bool:
        """
        Sprawdza, czy wszystkie wymagane zależności są dostępne.
        
        Returns:
            bool: True jeśli wszystkie zależności są dostępne, False w przeciwnym przypadku
        """
        try:
            import numpy
            import pandas
            import sklearn.ensemble
            return True
        except ImportError:
            return False
    
    def detect_price_anomalies(self, prices: pd.Series) -> pd.Series:
        """
        Wykrywa anomalie cenowe przy użyciu techniki Z-score.
        
        Args:
            prices: Seria cen, indeksowana datami
            
        Returns:
            pd.Series: Seria wartości boolowskich, True oznacza anomalię
        """
        if len(prices) < self.window_size:
            logger.warning(f"Za mało danych dla wykrycia anomalii (min. {self.window_size})")
            return pd.Series(False, index=prices.index)
        
        # Obliczenie średniej kroczącej i odchylenia standardowego
        rolling_mean = prices.rolling(window=self.window_size).mean()
        rolling_std = prices.rolling(window=self.window_size).std()
        
        # Obliczenie Z-score
        z_scores = np.abs((prices - rolling_mean) / rolling_std)
        
        # Oznaczenie anomalii
        anomalies = z_scores > self.threshold
        
        num_anomalies = anomalies.sum()
        if num_anomalies > 0:
            logger.info(f"Wykryto {num_anomalies} anomalii cenowych")
        
        return anomalies.fillna(False)
    
    def detect_volume_anomalies(self, volumes: pd.Series) -> pd.Series:
        """
        Wykrywa anomalie wolumenowe przy użyciu techniki Z-score.
        
        Args:
            volumes: Seria wolumenów, indeksowana datami
            
        Returns:
            pd.Series: Seria wartości boolowskich, True oznacza anomalię
        """
        if len(volumes) < self.window_size:
            logger.warning(f"Za mało danych dla wykrycia anomalii (min. {self.window_size})")
            return pd.Series(False, index=volumes.index)
        
        # Logarytmiczna transformacja wolumenu dla bardziej normalnego rozkładu
        log_volumes = np.log1p(volumes)
        
        # Obliczenie średniej kroczącej i odchylenia standardowego
        rolling_mean = log_volumes.rolling(window=self.window_size).mean()
        rolling_std = log_volumes.rolling(window=self.window_size).std()
        
        # Obliczenie Z-score
        z_scores = np.abs((log_volumes - rolling_mean) / rolling_std)
        
        # Oznaczenie anomalii
        anomalies = z_scores > self.threshold
        
        num_anomalies = anomalies.sum()
        if num_anomalies > 0:
            logger.info(f"Wykryto {num_anomalies} anomalii wolumenowych")
        
        return anomalies.fillna(False)
    
    def fit_isolation_forest(self, data: pd.DataFrame) -> None:
        """
        Trenuje model Isolation Forest na podanych danych.
        
        Args:
            data: DataFrame z cechami rynkowymi (ceny, wolumeny, wskaźniki, itp.)
        """
        if len(data) < 100:  # Minimalny rozmiar danych do treningu
            logger.warning("Za mało danych do treningu modelu Isolation Forest")
            return
        
        try:
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1  # Użycie wszystkich dostępnych rdzeni
            )
            self.isolation_forest.fit(data)
            logger.info("Model Isolation Forest został wytrenowany")
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu Isolation Forest: {e}")
            self.isolation_forest = None
    
    def detect_multivariate_anomalies(self, data: pd.DataFrame) -> pd.Series:
        """
        Wykrywa wielowymiarowe anomalie przy użyciu modelu Isolation Forest.
        
        Args:
            data: DataFrame z cechami rynkowymi
            
        Returns:
            pd.Series: Seria wartości boolowskich, True oznacza anomalię
        """
        if self.isolation_forest is None:
            logger.warning("Model Isolation Forest nie został wytrenowany")
            return pd.Series(False, index=data.index)
        
        try:
            # Predykcja: -1 oznacza anomalię, 1 oznacza normalną obserwację
            predictions = self.isolation_forest.predict(data)
            anomalies = pd.Series(predictions == -1, index=data.index)
            
            num_anomalies = anomalies.sum()
            if num_anomalies > 0:
                logger.info(f"Wykryto {num_anomalies} wielowymiarowych anomalii")
            
            return anomalies
        except Exception as e:
            logger.error(f"Błąd podczas wykrywania wielowymiarowych anomalii: {e}")
            return pd.Series(False, index=data.index)
    
    def generate_anomaly_report(self, ohlcv_data: pd.DataFrame) -> Dict:
        """
        Generuje raport o wykrytych anomaliach na podstawie danych OHLCV.
        
        Args:
            ohlcv_data: DataFrame z danymi Open, High, Low, Close, Volume
            
        Returns:
            Dict: Raport z informacjami o wykrytych anomaliach
        """
        if len(ohlcv_data) < self.window_size:
            return {"status": "error", "message": "Za mało danych"}
        
        try:
            # Wykrycie anomalii cenowych
            price_anomalies = self.detect_price_anomalies(ohlcv_data['close'])
            
            # Wykrycie anomalii wolumenowych
            volume_anomalies = self.detect_volume_anomalies(ohlcv_data['volume'])
            
            # Przygotowanie wielowymiarowych cech
            features = pd.DataFrame({
                'close_returns': ohlcv_data['close'].pct_change(),
                'volume_change': ohlcv_data['volume'].pct_change(),
                'high_low_diff': (ohlcv_data['high'] - ohlcv_data['low']) / ohlcv_data['low'],
                'close_open_diff': (ohlcv_data['close'] - ohlcv_data['open']) / ohlcv_data['open']
            }).fillna(0)
            
            # Trenowanie i wykrywanie wielowymiarowych anomalii
            if self.isolation_forest is None:
                self.fit_isolation_forest(features)
            
            multivar_anomalies = self.detect_multivariate_anomalies(features)
            
            # Przygotowanie raportu
            total_price_anomalies = price_anomalies.sum()
            total_volume_anomalies = volume_anomalies.sum()
            total_multivar_anomalies = multivar_anomalies.sum()
            
            # Znalezienie dat z anomaliami
            price_anomaly_dates = price_anomalies[price_anomalies].index.tolist()
            volume_anomaly_dates = volume_anomalies[volume_anomalies].index.tolist()
            multivar_anomaly_dates = multivar_anomalies[multivar_anomalies].index.tolist()
            
            return {
                "status": "success",
                "summary": {
                    "price_anomalies": int(total_price_anomalies),
                    "volume_anomalies": int(total_volume_anomalies),
                    "multivariate_anomalies": int(total_multivar_anomalies)
                },
                "details": {
                    "price_anomaly_dates": price_anomaly_dates,
                    "volume_anomaly_dates": volume_anomaly_dates,
                    "multivariate_anomaly_dates": multivar_anomaly_dates
                }
            }
        except Exception as e:
            logger.error(f"Błąd podczas generowania raportu o anomaliach: {e}")
            return {"status": "error", "message": str(e)}


def unit_test_anomaly_detector():
    """
    Funkcja testująca moduł wykrywania anomalii.
    """
    # Generowanie danych testowych
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    
    # Dane normalne
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    normal_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
    normal_volumes = np.random.lognormal(6, 0.5, 100)
    
    # Wstawienie anomalii
    normal_prices[50] = normal_prices[49] * 1.2  # Nagły wzrost o 20%
    normal_volumes[70] = normal_volumes[69] * 5  # Nagły wzrost wolumenu
    
    # Tworzenie DataFrame
    test_data = pd.DataFrame({
        'open': normal_prices - np.random.normal(0, 1, 100),
        'high': normal_prices + np.random.normal(2, 1, 100),
        'low': normal_prices - np.random.normal(2, 1, 100),
        'close': normal_prices,
        'volume': normal_volumes
    }, index=dates)
    
    # Inicjalizacja detektora
    detector = AnomalyDetector(config={
        "threshold": 2.5,
        "window_size": 10,
        "contamination": 0.05
    })
    
    # Testowanie wykrywania anomalii cenowych
    price_anomalies = detector.detect_price_anomalies(test_data['close'])
    assert isinstance(price_anomalies, pd.Series), "Wynik powinien być pd.Series"
    assert price_anomalies.sum() > 0, "Powinny zostać wykryte anomalie cenowe"
    assert price_anomalies[50], "Anomalia cenowa na indeksie 50 powinna zostać wykryta"
    
    # Testowanie wykrywania anomalii wolumenowych
    volume_anomalies = detector.detect_volume_anomalies(test_data['volume'])
    assert isinstance(volume_anomalies, pd.Series), "Wynik powinien być pd.Series"
    assert volume_anomalies.sum() > 0, "Powinny zostać wykryte anomalie wolumenowe"
    
    # Testowanie generowania raportu
    report = detector.generate_anomaly_report(test_data)
    assert isinstance(report, dict), "Raport powinien być słownikiem"
    assert report["status"] == "success", f"Status raportu powinien być 'success', otrzymano: {report['status']}"
    assert "summary" in report, "Raport powinien zawierać podsumowanie"
    assert "details" in report, "Raport powinien zawierać szczegóły"
    
    print("✅ Wszystkie testy modułu wykrywania anomalii zakończone pomyślnie!")


if __name__ == "__main__":
    # Konfiguracja logowania
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='anomaly_detection.log'
    )
    
    # Uruchomienie testów
    unit_test_anomaly_detector()
