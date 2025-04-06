
"""
anomaly_detection.py
--------------------
Moduł implementujący zaawansowane algorytmy wykrywania anomalii cenowych i wolumenowych.
Wykorzystuje algorytmy statystyczne oraz uczenie maszynowe.
"""

import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    filename='anomaly_detection.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

class AnomalyDetectionModel:
    """
    Klasa dostarczająca metody do wykrywania anomalii w danych rynkowych.
    Możliwe metody: 
    - Isolation Forest
    - Local Outlier Factor
    - Z-score
    - Modified Z-score
    - DBSCAN
    """
    
    def __init__(self, method='isolation_forest', contamination=0.05):
        """
        Inicjalizacja modelu wykrywania anomalii.
        
        Args:
            method (str): Metoda wykrywania anomalii ('isolation_forest', 'lof', 'zscore', 'mzscore')
            contamination (float): Oczekiwany odsetek anomalii w danych (0.0 - 0.5)
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.threshold = None
        logging.info(f"Inicjalizacja modelu wykrywania anomalii: {method}, contamination={contamination}")
    
    def fit(self, data):
        """
        Trenuje model wykrywania anomalii.
        
        Args:
            data (np.array or pd.DataFrame): Dane do analizy
        
        Returns:
            self: Wytrenowany model
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if self.method == 'isolation_forest':
            self.model = IsolationForest(contamination=self.contamination, random_state=42)
            self.model.fit(data)
        elif self.method == 'lof':
            self.model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
            self.model.fit(data)
        elif self.method == 'zscore':
            # Z-Score nie wymaga trenowania, tylko obliczenie średniej i odchylenia standardowego
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.threshold = 3.0  # Typowy próg dla Z-score (3 odchylenia standardowe)
        elif self.method == 'mzscore':
            # Modified Z-Score używa mediany zamiast średniej
            self.median = np.median(data, axis=0)
            self.mad = np.median(np.abs(data - self.median), axis=0) * 1.4826  # MAD * c dla normalności
            self.threshold = 3.5  # Typowy próg dla Modified Z-score
        else:
            raise ValueError(f"Nieznana metoda wykrywania anomalii: {self.method}")
            
        logging.info(f"Model {self.method} został wytrenowany na danych o kształcie {data.shape}")
        return self
    
    def predict(self, data):
        """
        Wykrywa anomalie w danych.
        
        Args:
            data (np.array or pd.DataFrame): Dane do analizy
        
        Returns:
            np.array: 1 dla normalnych obserwacji, -1 dla anomalii
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if self.model is None and self.method not in ['zscore', 'mzscore']:
            raise ValueError("Model nie został wytrenowany. Wywołaj najpierw metodę fit()")
            
        if self.method == 'isolation_forest':
            return self.model.predict(data)
        elif self.method == 'lof':
            return self.model.predict(data)
        elif self.method == 'zscore':
            z_scores = np.abs((data - self.mean) / (self.std + 1e-10))  # Unikamy dzielenia przez zero
            return np.where(np.max(z_scores, axis=1) > self.threshold, -1, 1)
        elif self.method == 'mzscore':
            mz_scores = np.abs((data - self.median) / (self.mad + 1e-10))  # Unikamy dzielenia przez zero
            return np.where(np.max(mz_scores, axis=1) > self.threshold, -1, 1)
            
    def detect_anomalies(self, data, return_scores=False):
        """
        Wykrywa anomalie i opcjonalnie zwraca wyniki z punktacją.
        
        Args:
            data (np.array or pd.DataFrame): Dane do analizy
            return_scores (bool): Czy zwrócić punktację anomalii
            
        Returns:
            pd.DataFrame: DataFrame z wynikami detekcji anomalii
        """
        is_dataframe = isinstance(data, pd.DataFrame)
        index = data.index if is_dataframe else None
        
        if is_dataframe:
            features = data.values
        else:
            features = data
        
        results = pd.DataFrame(index=index)
        results['is_anomaly'] = self.predict(features) == -1
        
        if return_scores:
            if self.method == 'isolation_forest':
                # Dla Isolation Forest, wyższe wartości oznaczają niższe prawdopodobieństwo anomalii
                scores = -self.model.decision_function(features)
                results['anomaly_score'] = scores
            elif self.method == 'lof':
                # Dla LOF, wyższe wartości oznaczają wyższe prawdopodobieństwo anomalii
                scores = self.model.negative_outlier_factor_ * -1
                results['anomaly_score'] = scores
            elif self.method == 'zscore':
                scores = np.max(np.abs((features - self.mean) / (self.std + 1e-10)), axis=1)
                results['anomaly_score'] = scores
            elif self.method == 'mzscore':
                scores = np.max(np.abs((features - self.median) / (self.mad + 1e-10)), axis=1)
                results['anomaly_score'] = scores
                
        return results

    def detect_price_anomalies(self, df, price_col='close', window=20):
        """
        Wykrywa anomalie cenowe używając ruchomego okna.
        
        Args:
            df (pd.DataFrame): DataFrame z danymi cenowymi
            price_col (str): Nazwa kolumny z cenami
            window (int): Wielkość okna do analizy
            
        Returns:
            pd.Series: Seria z informacją czy dany punkt jest anomalią (True/False)
        """
        if self.method in ['zscore', 'mzscore']:
            # Dla metod statystycznych używamy rolling window
            if self.method == 'zscore':
                means = df[price_col].rolling(window=window).mean()
                stds = df[price_col].rolling(window=window).std()
                z_scores = np.abs((df[price_col] - means) / (stds + 1e-10))
                anomalies = z_scores > self.threshold
            else:  # mzscore
                rolling_median = df[price_col].rolling(window=window).median()
                rolling_mad = df[price_col].rolling(window=window).apply(
                    lambda x: np.median(np.abs(x - np.median(x))) * 1.4826
                )
                mz_scores = np.abs((df[price_col] - rolling_median) / (rolling_mad + 1e-10))
                anomalies = mz_scores > self.threshold
                
            return anomalies.astype(bool)
        else:
            # Dla metod uczenia maszynowego używamy podejścia z oknem przesuwnym
            anomalies = pd.Series(False, index=df.index)
            
            for i in range(window, len(df)):
                features = df[price_col].iloc[i-window:i].values.reshape(-1, 1)
                self.fit(features)
                prediction = self.predict(df[price_col].iloc[i].reshape(1, -1))
                anomalies.iloc[i] = prediction[0] == -1
                
            return anomalies

def detect_volatility_anomalies(price_data, window=20, n_std=2.5):
    """
    Wykrywa anomalie w zmienności cen.
    
    Args:
        price_data (pd.Series): Seria z danymi cenowymi
        window (int): Rozmiar okna do obliczania zmienności
        n_std (float): Liczba odchyleń standardowych powyżej której uznajemy zmienność za anomalię
        
    Returns:
        pd.Series: True dla punktów z anomalną zmiennością
    """
    # Obliczamy zmienność jako odchylenie standardowe zwrotów w oknie
    returns = price_data.pct_change().dropna()
    rolling_vol = returns.rolling(window=window).std()
    
    # Obliczamy średnią i odchylenie standardowe zmienności
    mean_vol = rolling_vol.mean()
    std_vol = rolling_vol.std()
    
    # Wykrywamy anomalie jako punkty, gdzie zmienność przekracza średnią o n_std odchyleń standardowych
    anomalies = rolling_vol > (mean_vol + n_std * std_vol)
    
    logging.info(f"Wykryto {anomalies.sum()} anomalii zmienności z {len(anomalies)} punktów")
    return anomalies

def detect_volume_anomalies(volume_data, window=20, n_std=3.0):
    """
    Wykrywa anomalie w wolumenie.
    
    Args:
        volume_data (pd.Series): Seria z danymi o wolumenie
        window (int): Rozmiar okna do analizy
        n_std (float): Liczba odchyleń standardowych powyżej której uznajemy wolumen za anomalię
        
    Returns:
        pd.Series: True dla punktów z anomalnym wolumenem
    """
    # Normalizujemy wolumen za pomocą logarytmu
    log_volume = np.log1p(volume_data)
    
    # Obliczamy średni log-wolumen w oknie
    rolling_mean = log_volume.rolling(window=window).mean()
    rolling_std = log_volume.rolling(window=window).std()
    
    # Wykrywamy anomalie
    anomalies = (log_volume > (rolling_mean + n_std * rolling_std)) | (log_volume < (rolling_mean - n_std * rolling_std))
    
    logging.info(f"Wykryto {anomalies.sum()} anomalii wolumenu z {len(anomalies)} punktów")
    return anomalies

# Przykładowe użycie
if __name__ == "__main__":
    try:
        # Generujemy przykładowe dane
        np.random.seed(42)
        n = 1000
        
        # Normalne dane z kilkoma anomaliami
        data = np.random.randn(n, 2)
        # Dodajemy kilka anomalii
        anomalies_idx = np.random.choice(n, 50, replace=False)
        data[anomalies_idx] = data[anomalies_idx] * 5
        
        # Tworzymy i trenujemy model
        model = AnomalyDetectionModel(method='isolation_forest', contamination=0.05)
        model.fit(data)
        
        # Wykrywamy anomalie
        predictions = model.predict(data)
        anomalies_count = (predictions == -1).sum()
        
        logging.info(f"Wykryto {anomalies_count} anomalii z {n} punktów")
        print(f"Wykryto {anomalies_count} anomalii.")
        
    except Exception as e:
        logging.error(f"Błąd: {e}")
        print(f"Wystąpił błąd: {e}")
