
"""
anomaly_detection.py
------------------
Moduł zawierający klasę AnomalyDetector do wykrywania anomalii w danych cenowych.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

class AnomalyDetector:
    """
    Klasa do wykrywania anomalii w danych cenowych i sygnałach transakcyjnych.
    Użyteczna do identyfikacji potencjalnych błędów w strategiach lub manipulacji rynkiem.
    """
    
    def __init__(self, model_path: Optional[str] = None, sensitivity: float = 0.05):
        """
        Inicjalizacja detektora anomalii.
        
        Args:
            model_path: Opcjonalna ścieżka do zapisanego modelu
            sensitivity: Poziom czułości wykrywania anomalii (0.01-0.1, gdzie niższe wartości oznaczają większą czułość)
        """
        self.model = None
        self.sensitivity = max(0.01, min(0.1, sensitivity))  # Zakres 0.01-0.1
        
        if model_path and os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logging.info(f"Załadowano model detekcji anomalii z {model_path}")
            except Exception as e:
                logging.error(f"Nie udało się załadować modelu detekcji anomalii: {e}")
                self._initialize_default_model()
        else:
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Inicjalizuje domyślny model detekcji anomalii."""
        try:
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=self.sensitivity,  # Oczekiwany procent anomalii w danych
                random_state=42,
                n_estimators=100
            )
            logging.info("Zainicjalizowano domyślny model detekcji anomalii (Isolation Forest)")
        except ImportError:
            logging.warning("Nie można zaimportować scikit-learn. Używam prostego modelu zastępczego.")
            self.model = None
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Trenuje model detekcji anomalii na danych.
        
        Args:
            data: Dane treningowe (DataFrame lub np.ndarray)
        """
        if self.model is None:
            self._initialize_default_model()
            
        if self.model is None:
            logging.warning("Brak modelu do treningu (scikit-learn niedostępny)")
            return
            
        try:
            # Przygotuj dane do treningu
            X = self._prepare_data(data)
            
            # Trenuj model
            self.model.fit(X)
            logging.info(f"Wytrenowano model detekcji anomalii na {X.shape[0]} próbkach")
        except Exception as e:
            logging.error(f"Błąd podczas treningu modelu detekcji anomalii: {e}")
    
    def detect(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Wykrywa anomalie w danych.
        
        Args:
            data: Dane do analizy (DataFrame lub np.ndarray)
            
        Returns:
            Dict: Wyniki detekcji zawierające indeksy anomalii i ich wyniki
        """
        try:
            if self.model is None:
                return self._fallback_detect(data)
                
            # Przygotuj dane
            X = self._prepare_data(data)
            
            # Wykonaj predykcję (-1 to anomalia, 1 to normalne dane)
            predictions = self.model.predict(X)
            scores = self.model.decision_function(X)
            
            # Znajdź indeksy anomalii
            anomaly_indices = np.where(predictions == -1)[0]
            
            # Jeśli dane zawierają indeks czasowy, użyj go
            timestamps = None
            if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
                timestamps = data.index[anomaly_indices].tolist()
            
            # Stwórz wynik
            result = {
                "anomaly_indices": anomaly_indices.tolist(),
                "anomaly_scores": scores[anomaly_indices].tolist(),
                "anomaly_count": len(anomaly_indices),
                "total_samples": X.shape[0],
                "anomaly_ratio": len(anomaly_indices) / X.shape[0] if X.shape[0] > 0 else 0,
                "timestamps": timestamps
            }
            
            return result
        except Exception as e:
            logging.error(f"Błąd podczas wykrywania anomalii: {e}")
            return self._fallback_detect(data)
    
    def _fallback_detect(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Prosta metoda zastępcza do wykrywania anomalii bez modelu.
        
        Args:
            data: Dane do analizy
            
        Returns:
            Dict: Wyniki detekcji
        """
        try:
            # Konwertuj dane do numpy array jeśli potrzeba
            if isinstance(data, pd.DataFrame):
                X = data.select_dtypes(include=['number']).values
            else:
                X = data
                
            # Metoda zastępcza - użyj prostej metody z-score do wykrywania anomalii
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0) + 1e-10  # Unikaj dzielenia przez 0
            
            # Oblicz z-scores
            z_scores = np.abs((X - mean) / std)
            
            # Użyj maksymalnego z-score każdej próbki jako wyniku anomalii
            max_z_scores = np.max(z_scores, axis=1)
            
            # Próg z-score odpowiadający czułości
            threshold = max(2.5, 3.0 - self.sensitivity * 10)
            
            # Znajdź indeksy anomalii
            anomaly_indices = np.where(max_z_scores > threshold)[0]
            
            # Jeśli dane zawierają indeks czasowy, użyj go
            timestamps = None
            if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
                timestamps = data.index[anomaly_indices].tolist()
            
            # Stwórz wynik
            result = {
                "anomaly_indices": anomaly_indices.tolist(),
                "anomaly_scores": max_z_scores[anomaly_indices].tolist(),
                "anomaly_count": len(anomaly_indices),
                "total_samples": X.shape[0],
                "anomaly_ratio": len(anomaly_indices) / X.shape[0] if X.shape[0] > 0 else 0,
                "timestamps": timestamps,
                "method": "z_score_fallback"
            }
            
            return result
        except Exception as e:
            logging.error(f"Błąd podczas zastępczego wykrywania anomalii: {e}")
            return {
                "anomaly_indices": [],
                "anomaly_scores": [],
                "anomaly_count": 0,
                "total_samples": 0,
                "anomaly_ratio": 0,
                "error": str(e),
                "method": "failed_fallback"
            }
    
    def _prepare_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Przygotowuje dane do analizy.
        
        Args:
            data: Dane wejściowe (DataFrame lub np.ndarray)
            
        Returns:
            np.ndarray: Przygotowane dane
        """
        if isinstance(data, pd.DataFrame):
            # Wybierz tylko kolumny numeryczne
            numeric_data = data.select_dtypes(include=['number'])
            
            # Dodaj cechy techniczne jeśli są dostępne odpowiednie kolumny
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                price_volatility = data['high'] - data['low']
                price_change = data['close'] - data['open']
                numeric_data['price_volatility'] = price_volatility
                numeric_data['price_change'] = price_change
            
            # Wypełnij braki danych
            numeric_data = numeric_data.fillna(method='ffill').fillna(0)
            
            return numeric_data.values
        else:
            # Zakładamy, że dane są już w formacie np.ndarray
            # Wypełnij braki danych
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=0.0)
            return data
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Kompatybilność ze standardowym interfejsem - przewiduje, czy dane są anomaliami.
        
        Args:
            data: Dane do analizy
            
        Returns:
            np.ndarray: Wynik detekcji (-1 dla anomalii, 1 dla normalnych danych)
        """
        try:
            if self.model is None:
                self._initialize_default_model()
                
            if self.model is None:
                # Fallback gdy model jest niedostępny
                X = self._prepare_data(data)
                result = np.ones(X.shape[0])
                detection = self._fallback_detect(data)
                result[detection["anomaly_indices"]] = -1
                return result
            
            # Przygotuj dane
            X = self._prepare_data(data)
            
            # Sprawdź czy model został wytrenowany, jeśli nie, trenuj na bieżących danych
            try:
                # Próba wywołania predict może spowodować wyjątek jeśli model nie jest wytrenowany
                self.model.predict(X[:1])
            except Exception as e:
                if "not fitted yet" in str(e):
                    logging.info("Model IsolationForest nie był wytrenowany. Trenuję na bieżących danych...")
                    self.model.fit(X)
                    logging.info(f"Model wytrenowany na {X.shape[0]} próbkach")
                else:
                    raise e
            
            # Wykonaj predykcję
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"Błąd podczas przewidywania anomalii: {e}")
            # W przypadku błędu zwróć wszystkie dane jako normalne
            if isinstance(data, pd.DataFrame):
                return np.ones(data.shape[0])
            else:
                return np.ones(data.shape[0] if len(data.shape) > 1 else len(data))
                
    def save_model(self, path: str) -> bool:
        """
        Zapisuje model do pliku.
        
        Args:
            path: Ścieżka do pliku
            
        Returns:
            bool: True jeśli zapis się powiódł, False w przeciwnym razie
        """
        try:
            if self.model is not None:
                import pickle
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                logging.info(f"Model detekcji anomalii zapisany w {path}")
                return True
            else:
                logging.warning("Brak modelu do zapisania")
                return False
        except Exception as e:
            logging.error(f"Błąd podczas zapisywania modelu: {e}")
            return False
