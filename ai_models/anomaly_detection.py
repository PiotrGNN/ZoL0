
"""
anomaly_detection.py
---------------------
Moduł wykrywania anomalii cenowych i wolumenowych w danych giełdowych.
Wykorzystuje algorytmy statystyczne oraz uczenie maszynowe do wykrywania
nietypowych wzorców, które mogą wskazywać na manipulacje rynkowe,
nagłe zmiany trendów lub inne istotne wydarzenia rynkowe.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/anomaly_detector.log"),
        logging.StreamHandler()
    ]
)

class AnomalyDetectionModel:
    """
    Klasa implementująca metody wykrywania anomalii w danych finansowych.
    Wykorzystuje algorytmy izolacji anomalii oraz analizę statystyczną.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Inicjalizacja modelu wykrywania anomalii.

        Parameters:
            contamination (float): Szacowana frakcja anomalii w danych (0.0-0.5).
            random_state (int): Ziarno losowości dla powtarzalności wyników.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # Wykorzystaj wszystkie dostępne rdzenie
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        logging.info("Zainicjalizowano model wykrywania anomalii (Isolation Forest)")

    def fit(self, data: pd.DataFrame) -> "AnomalyDetectionModel":
        """
        Trenuje model na podstawie dostarczonych danych.

        Parameters:
            data (pd.DataFrame): DataFrame zawierający cechy finansowe.

        Returns:
            self: Wytrenowany model.
        """
        try:
            # Utworzenie kopii danych aby uniknąć ostrzeżeń o operacjach in-place
            X = data.copy()
            
            # Sprawdzenie i obsługa brakujących wartości
            if X.isnull().values.any():
                logging.warning("Wykryto brakujące wartości. Zastępowanie średnią.")
                X.fillna(X.mean(), inplace=True)
            
            # Standaryzacja danych
            X_scaled = self.scaler.fit_transform(X)
            
            # Trenowanie modelu
            self.model.fit(X_scaled)
            self.is_fitted = True
            
            logging.info(f"Model został wytrenowany na {X.shape[0]} próbkach z {X.shape[1]} cechami.")
            return self
        except Exception as e:
            logging.error(f"Błąd podczas trenowania modelu: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Przewiduje anomalie w nowych danych.

        Parameters:
            data (pd.DataFrame): DataFrame z nowymi danymi do analizy.

        Returns:
            np.ndarray: Tablica z wynikami (-1 dla anomalii, 1 dla normalnych).
        """
        if not self.is_fitted:
            raise ValueError("Model nie został wytrenowany. Najpierw wywołaj metodę fit().")
        
        try:
            # Utworzenie kopii danych
            X = data.copy()
            
            # Obsługa brakujących wartości
            if X.isnull().values.any():
                logging.warning("Wykryto brakujące wartości. Zastępowanie średnią.")
                X.fillna(X.mean(), inplace=True)
            
            # Standaryzacja danych
            X_scaled = self.scaler.transform(X)
            
            # Predykcja
            predictions = self.model.predict(X_scaled)
            logging.info(f"Wykonano predykcję dla {X.shape[0]} próbek.")
            
            return predictions
        except Exception as e:
            logging.error(f"Błąd podczas predykcji: {e}")
            raise

    def detect_anomalies(self, data: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Wykrywa anomalie w danych i zwraca DataFrame z flagami anomalii.

        Parameters:
            data (pd.DataFrame): Dane do analizy.
            threshold (float, optional): Próg decyzyjny dla anomalii.

        Returns:
            pd.DataFrame: Dane z dodatkową kolumną 'is_anomaly'.
        """
        try:
            # Skopiuj dane wejściowe
            result_df = data.copy()
            
            # Utwórz podgląd danych tylko z kolumnami numerycznymi
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("Brak kolumn numerycznych w danych wejściowych.")
                
            X = result_df[numeric_cols]
            
            # Wykryj anomalie
            if not self.is_fitted:
                self.fit(X)
            
            # Predykcja anomalii (-1) lub normalnych danych (1)
            result_df['anomaly_score'] = self.model.decision_function(self.scaler.transform(X))
            result_df['is_anomaly'] = self.predict(X)
            
            # Zamieniamy wartości na bardziej intuicyjne: 1 dla anomalii, 0 dla normalnych
            result_df['is_anomaly'] = np.where(result_df['is_anomaly'] == -1, 1, 0)
            
            num_anomalies = sum(result_df['is_anomaly'])
            logging.info(f"Wykryto {num_anomalies} anomalii w {len(result_df)} rekordach.")
            
            return result_df
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania anomalii: {e}")
            raise

    def save_model(self, path: str = "saved_models") -> str:
        """
        Zapisuje wytrenowany model do pliku.

        Parameters:
            path (str): Ścieżka do katalogu, w którym ma być zapisany model.

        Returns:
            str: Pełna ścieżka do zapisanego pliku modelu.
        """
        import joblib
        from datetime import datetime
        
        if not self.is_fitted:
            raise ValueError("Model nie został wytrenowany. Najpierw wywołaj metodę fit().")
            
        try:
            # Utwórz katalog, jeśli nie istnieje
            os.makedirs(path, exist_ok=True)
            
            # Generuj unikalną nazwę pliku
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{path}/anomaly_detection_model_{timestamp}.pkl"
            
            # Zapisz model
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'contamination': self.contamination,
                'random_state': self.random_state
            }, filename)
            
            logging.info(f"Model został zapisany do pliku: {filename}")
            return filename
        except Exception as e:
            logging.error(f"Błąd podczas zapisywania modelu: {e}")
            raise

    @classmethod
    def load_model(cls, filepath: str) -> "AnomalyDetectionModel":
        """
        Wczytuje model z pliku.

        Parameters:
            filepath (str): Ścieżka do pliku modelu.

        Returns:
            AnomalyDetectionModel: Wczytany model.
        """
        import joblib
        
        try:
            # Wczytaj model
            model_data = joblib.load(filepath)
            
            # Utwórz instancję klasy
            instance = cls(
                contamination=model_data['contamination'],
                random_state=model_data['random_state']
            )
            
            # Ustaw atrybuty
            instance.model = model_data['model']
            instance.scaler = model_data['scaler']
            instance.is_fitted = model_data['is_fitted']
            
            logging.info(f"Model został wczytany z pliku: {filepath}")
            return instance
        except Exception as e:
            logging.error(f"Błąd podczas wczytywania modelu: {e}")
            raise


# Przykład użycia modułu
if __name__ == "__main__":
    # Przygotowanie przykładowych danych
    np.random.seed(42)
    n_samples = 1000
    
    # Tworzenie normalnych danych
    normal_data = np.random.normal(0, 1, (n_samples, 3))
    
    # Tworzenie anomalii
    anomalies = np.random.uniform(-5, 5, (50, 3))
    
    # Łączenie danych
    all_data = np.vstack([normal_data, anomalies])
    
    # Konwersja do DataFrame
    columns = ['price', 'volume', 'volatility']
    df = pd.DataFrame(all_data, columns=columns)
    
    # Inicjalizacja i trenowanie modelu
    model = AnomalyDetectionModel(contamination=0.05)
    
    # Wykrywanie anomalii
    result = model.detect_anomalies(df)
    
    # Wyświetlanie wyników
    anomalies_count = result['is_anomaly'].sum()
    print(f"Wykryto {anomalies_count} anomalii w zbiorze {len(result)} próbek.")
    
    # Zapisywanie modelu
    model_path = model.save_model()
    print(f"Model zapisany w: {model_path}")
