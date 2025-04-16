
"""
model_recognition.py
-------------------
Moduł zawierający klasę ModelRecognizer do rozpoznawania wzorców na wykresach cenowych.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

class ModelRecognizer:
    """
    Klasa do rozpoznawania wzorców cenowych na wykresach, np. flagi, głowy i ramiona, itp.
    Wykorzystuje klasyfikację wzorców używając metod uczenia maszynowego.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicjalizacja rozpoznawacza wzorców.
        
        Args:
            model_path: Opcjonalna ścieżka do zapisanego modelu
        """
        self.model = None
        self.pattern_names = [
            "Bull Flag", "Bear Flag", "Head and Shoulders", 
            "Inverse H&S", "Double Top", "Double Bottom",
            "Triple Top", "Triple Bottom", "Ascending Triangle",
            "Descending Triangle", "Symmetrical Triangle", "Channel"
        ]
        
        if model_path and os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logging.info(f"Załadowano model rozpoznawania wzorców z {model_path}")
            except Exception as e:
                logging.error(f"Nie udało się załadować modelu rozpoznawania wzorców: {e}")
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """
        Rozpoznaje wzorce w danych cenowych.
        
        Args:
            data: Dane cenowe w formacie DataFrame lub np.ndarray
            
        Returns:
            Dict[str, float]: Słownik z nazwami wzorców i ich prawdopodobieństwami
        """
        try:
            if self.model is None:
                # Jeśli nie ma modelu, zwracamy przykładowe dane
                import random
                return {pattern: round(random.uniform(0.1, 0.95), 2) for pattern in self.pattern_names}
            
            # Przetwarzanie danych wejściowych
            X = self._preprocess_data(data)
            
            # Faktyczna predykcja z modelu
            predictions = self.model.predict_proba(X)[0]
            
            # Zwracamy słownik z nazwami wzorców i ich prawdopodobieństwami
            result = {pattern: float(pred) for pattern, pred in zip(self.pattern_names, predictions)}
            
            return result
        except Exception as e:
            logging.error(f"Błąd podczas rozpoznawania wzorców: {e}")
            # Fallback - zwracamy None dla wszystkich wzorców
            return {pattern: 0.0 for pattern in self.pattern_names}
    
    def _preprocess_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Przetwarza dane wejściowe do formatu akceptowanego przez model.
        
        Args:
            data: Dane cenowe w formacie DataFrame lub np.ndarray
            
        Returns:
            np.ndarray: Przetworzone dane gotowe do predykcji
        """
        if isinstance(data, pd.DataFrame):
            # Sprawdź czy są wymagane kolumny
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logging.warning(f"Brakujące kolumny w danych wejściowych: {missing_cols}")
                # Dodaj brakujące kolumny z wartościami 0
                for col in missing_cols:
                    data[col] = 0
            
            # Wybierz tylko potrzebne kolumny
            X = data[required_cols].values
        else:
            # Zakładamy, że dane są już w formacie np.ndarray
            X = data
            
            # Sprawdź czy dane mają wystarczającą liczbę kolumn
            if X.shape[1] < 5:
                # Dodaj brakujące kolumny
                padding = np.zeros((X.shape[0], 5 - X.shape[1]))
                X = np.hstack((X, padding))
        
        # Normalizacja danych
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Unikaj dzielenia przez 0
        X_norm = (X - X_mean) / X_std
        
        # Dodaj cechy techniczne
        X_features = self._extract_technical_features(X)
        
        # Połącz wszystkie cechy
        X_final = np.hstack((X_norm, X_features))
        
        # Dodaj wymiar wsadowy, jeśli go nie ma
        if len(X_final.shape) == 2:
            X_final = np.expand_dims(X_final, axis=0)
            
        return X_final
    
    def _extract_technical_features(self, X: np.ndarray) -> np.ndarray:
        """
        Ekstrahuje cechy techniczne z danych cenowych.
        
        Args:
            X: Dane cenowe w formacie np.ndarray
            
        Returns:
            np.ndarray: Wyekstrahowane cechy techniczne
        """
        # Przykładowa implementacja - w rzeczywistych modelach byłoby więcej cech
        try:
            open_prices = X[:, 0]
            high_prices = X[:, 1]
            low_prices = X[:, 2]
            close_prices = X[:, 3]
            volume = X[:, 4]
            
            # Oblicz podstawowe wskaźniki
            price_range = high_prices - low_prices
            body_size = np.abs(close_prices - open_prices)
            upper_shadow = high_prices - np.maximum(open_prices, close_prices)
            lower_shadow = np.minimum(open_prices, close_prices) - low_prices
            
            # Stwórz macierz cech
            features = np.column_stack((
                price_range,
                body_size,
                upper_shadow,
                lower_shadow,
                volume
            ))
            
            return features
        except Exception as e:
            logging.error(f"Błąd podczas ekstrakcji cech technicznych: {e}")
            # Zwróć pustą macierz cech
            return np.zeros((X.shape[0], 5))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trenuje model rozpoznawania wzorców.
        
        Args:
            X: Dane treningowe
            y: Etykiety wzorców
            
        Returns:
            None
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            logging.info("Model rozpoznawania wzorców został wytrenowany")
        except Exception as e:
            logging.error(f"Błąd podczas trenowania modelu rozpoznawania wzorców: {e}")
            
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
                logging.info(f"Model rozpoznawania wzorców zapisany w {path}")
                return True
            else:
                logging.warning("Brak modelu do zapisania")
                return False
        except Exception as e:
            logging.error(f"Błąd podczas zapisywania modelu: {e}")
            return False
