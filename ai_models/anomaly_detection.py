
"""
anomaly_detection.py
-------------------
Model do wykrywania anomalii cenowych i wzorców zachowań rynkowych.

Funkcjonalności:
- Wykrywanie nagłych skoków/spadków cen.
- Identyfikacja nietypowych wzorców wolumenu.
- Analiza odstających wartości wskaźników technicznych.
- Możliwość trenowania na historycznych danych z oznaczonymi anomaliami.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AnomalyDetectionModel:
    """
    Model wykrywania anomalii w danych cenowych i wskaźnikach.
    Wykorzystuje algorytmy Isolation Forest i statystyczne metody
    wykrywania odstających wartości.
    """
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        """
        Inicjalizacja modelu wykrywania anomalii.
        
        Parameters:
            contamination (float): Szacowany procent anomalii w danych (0.0-0.5).
            n_estimators (int): Liczba drzew w lesie izolacyjnym.
            random_state (int): Ziarno losowości dla powtarzalnych wyników.
        """
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        logger.info(f"Zainicjalizowano model wykrywania anomalii (contam={contamination})")
    
    def detect_price_anomalies(self, price_data, columns=None):
        """
        Wykrywa anomalie w danych cenowych.
        
        Parameters:
            price_data (DataFrame): DataFrame z danymi cenowymi.
            columns (list): Lista kolumn do analizy. Jeśli None, użyj wszystkich.
            
        Returns:
            DataFrame: DataFrame z dodatkową kolumną 'is_anomaly' (1 dla anomalii, 0 dla normalnych).
        """
        try:
            if price_data is None or len(price_data) == 0:
                logger.warning("Brak danych do wykrycia anomalii")
                return pd.DataFrame()
                
            if columns is None:
                # Użyj wszystkich kolumn liczbowych
                columns = price_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Przygotowanie danych
            X = price_data[columns].copy()
            X = X.fillna(method='ffill')
            
            # Normalizacja danych
            X_scaled = self.scaler.fit_transform(X)
            
            # Wykrywanie anomalii
            predictions = self.model.fit_predict(X_scaled)
            anomaly_score = self.model.decision_function(X_scaled)
            
            # Konwersja wyników (-1: anomalia, 1: normalne) na (1: anomalia, 0: normalne)
            is_anomaly = np.where(predictions == -1, 1, 0)
            
            # Tworzenie wyniku
            result = price_data.copy()
            result['is_anomaly'] = is_anomaly
            result['anomaly_score'] = anomaly_score
            
            anomaly_count = np.sum(is_anomaly)
            logger.info(f"Wykryto {anomaly_count} anomalii w danych ({anomaly_count/len(price_data)*100:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Błąd podczas wykrywania anomalii: {e}")
            return pd.DataFrame()
    
    def detect_volume_anomalies(self, volume_data, window=20):
        """
        Wykrywa anomalie w danych wolumenowych.
        
        Parameters:
            volume_data (Series): Seria danych wolumenowych.
            window (int): Okno czasowe do analizy.
            
        Returns:
            Series: Seria z oznaczonymi anomaliami wolumenu.
        """
        try:
            if volume_data is None or len(volume_data) == 0:
                logger.warning("Brak danych wolumenowych do analizy")
                return pd.Series()
            
            # Obliczanie średniej kroczącej i odchylenia standardowego
            rolling_mean = volume_data.rolling(window=window).mean()
            rolling_std = volume_data.rolling(window=window).std()
            
            # Identyfikacja anomalii (3 odchylenia standardowe od średniej)
            upper_band = rolling_mean + 3 * rolling_std
            
            # Oznaczenie anomalii
            volume_anomalies = (volume_data > upper_band).astype(int)
            
            anomaly_count = volume_anomalies.sum()
            logger.info(f"Wykryto {anomaly_count} anomalii wolumenu")
            
            return volume_anomalies
            
        except Exception as e:
            logger.error(f"Błąd podczas wykrywania anomalii wolumenu: {e}")
            return pd.Series()
    
    def detect_pattern_anomalies(self, ohlcv_data, lookback=5):
        """
        Wykrywa anomalie we wzorcach cenowych OHLCV.
        
        Parameters:
            ohlcv_data (DataFrame): DataFrame z danymi OHLCV.
            lookback (int): Liczba świeczek wstecz do analizy wzorca.
            
        Returns:
            DataFrame: DataFrame z dodatkową kolumną 'pattern_anomaly'.
        """
        try:
            if ohlcv_data is None or len(ohlcv_data) < lookback + 1:
                logger.warning(f"Za mało danych do analizy wzorców (min {lookback+1})")
                return pd.DataFrame()
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in ohlcv_data.columns]
            
            if missing_columns:
                logger.warning(f"Brakujące kolumny do analizy wzorców: {missing_columns}")
                return pd.DataFrame()
            
            # Przygotowanie danych
            features = []
            for i in range(len(ohlcv_data) - lookback):
                window = ohlcv_data.iloc[i:i+lookback]
                
                # Tworzenie cech opisujących wzorzec
                pattern_features = []
                
                # Względne zmiany cen
                for col in ['open', 'high', 'low', 'close']:
                    changes = window[col].pct_change().dropna().values
                    pattern_features.extend(changes)
                
                # Cechy wolumenu
                vol_changes = window['volume'].pct_change().dropna().values
                pattern_features.extend(vol_changes)
                
                # Zakres cenowy
                ranges = (window['high'] - window['low']).values / window['close'].values
                pattern_features.extend(ranges)
                
                features.append(pattern_features)
            
            if not features:
                logger.warning("Nie udało się wygenerować cech wzorców")
                return pd.DataFrame()
                
            # Wykrywanie anomalii we wzorcach
            X = np.array(features)
            X = np.nan_to_num(X)  # Obsługa NaN
            
            model = IsolationForest(contamination=self.contamination, random_state=42)
            predictions = model.fit_predict(X)
            
            # Konwersja wyników
            pattern_anomalies = np.where(predictions == -1, 1, 0)
            
            # Tworzenie wyniku
            result = ohlcv_data.copy()
            result['pattern_anomaly'] = 0
            result.iloc[lookback:]['pattern_anomaly'] = pattern_anomalies
            
            anomaly_count = np.sum(pattern_anomalies)
            logger.info(f"Wykryto {anomaly_count} anomalii wzorców cenowych")
            
            return result
            
        except Exception as e:
            logger.error(f"Błąd podczas wykrywania anomalii wzorców: {e}")
            return pd.DataFrame()

    def get_model_info(self):
        """
        Zwraca informacje o modelu.
        
        Returns:
            dict: Słownik z informacjami o modelu.
        """
        return {
            "model_name": "Isolation Forest Anomaly Detection",
            "contamination": self.contamination,
            "n_estimators": self.model.n_estimators,
            "features_supported": ["price", "volume", "pattern"]
        }


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    # Konfiguracja logowania
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("anomaly_detection.log"),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Generowanie przykładowych danych
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
        
        # Dane cenowe z anomaliami
        prices = np.cumsum(np.random.normal(0, 1, 100))
        # Dodanie kilku anomalii
        prices[30] += 10  # Nagły skok
        prices[60] -= 8   # Nagły spadek
        
        # Dane wolumenowe
        volume = np.random.gamma(shape=2, scale=100, size=100)
        # Anomalie wolumenu
        volume[45] = volume[45] * 5
        volume[75] = volume[75] * 4
        
        # Tworzenie DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 2, 100),
            'low': prices - np.random.uniform(0, 2, 100),
            'close': prices + np.random.normal(0, 0.5, 100),
            'volume': volume
        })
        df.set_index('date', inplace=True)
        
        # Inicjalizacja modelu
        model = AnomalyDetectionModel(contamination=0.05)
        
        # Wykrywanie anomalii cenowych
        price_anomalies = model.detect_price_anomalies(df, columns=['close'])
        
        # Wykrywanie anomalii wolumenu
        volume_anomalies = model.detect_volume_anomalies(df['volume'])
        
        # Wykrywanie anomalii wzorców
        pattern_anomalies = model.detect_pattern_anomalies(df)
        
        print(f"Wykryte anomalie cenowe: {price_anomalies['is_anomaly'].sum()}")
        print(f"Wykryte anomalie wolumenu: {volume_anomalies.sum()}")
        
        if 'pattern_anomaly' in pattern_anomalies.columns:
            print(f"Wykryte anomalie wzorców: {pattern_anomalies['pattern_anomaly'].sum()}")
        
    except Exception as e:
        logging.error(f"Błąd podczas testowania modelu: {e}")
