"""
model_recognition.py
-------------------
Moduł do rozpoznawania typów modeli rynkowych.
"""

import logging
import random
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ModelRecognizer:
    """Rozpoznawacz modeli rynkowych."""

    def __init__(self):
        """Inicjalizuje rozpoznawacz modeli."""
        self.logger = logging.getLogger("ModelRecognizer")
        self.logger.info("ModelRecognizer zainicjalizowany")
        self.known_patterns = {
            "trend": {"name": "Trend Following", "confidence": 0.85, "type": "trend"},
            "reversal": {"name": "Mean Reversion", "confidence": 0.78, "type": "reversal"},
            "breakout": {"name": "Breakout", "confidence": 0.82, "type": "breakout"},
            "volatility": {"name": "Volatility", "confidence": 0.75, "type": "volatility"}
        }

        self.model_types = [
            {
                "id": "trending_market",
                "name": "Trend rosnący/malejący",
                "description": "Rynek w wyraźnym trendzie jednokierunkowym"
            },
            {
                "id": "ranging_market",
                "name": "Rynek w konsolidacji",
                "description": "Cena oscyluje w określonym zakresie"
            },
            {
                "id": "breakout_pattern",
                "name": "Wybicie z formacji",
                "description": "Cena przebija ważny poziom wsparcia lub oporu"
            },
            {
                "id": "reversal_pattern",
                "name": "Formacja odwrócenia",
                "description": "Wzór sugerujący zmianę kierunku trendu"
            },
            {
                "id": "high_volatility",
                "name": "Wysoka zmienność",
                "description": "Rynek charakteryzuje się dużymi wahaniami cen"
            },
            {
                "id": "low_liquidity",
                "name": "Niska płynność",
                "description": "Rynek z niskim wolumenem i dużymi spreadami"
            }
        ]
        self.last_recognition_time = time.time()
        self.accuracy = 78.5
        self.model_type = "Pattern Recognition System"
        self.status = "Active"


    def predict(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Przewiduje typ modelu rynkowego na podstawie danych.

        Parameters:
            data (Optional[Dict[str, Any]]): Dane rynkowe

        Returns:
            Dict[str, Any]: Rozpoznany model
        """
        return self.identify_model_type(data)

    def identify_model_type(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identyfikuje typ modelu rynkowego.

        Parameters:
            data (Optional[Dict[str, Any]]): Dane rynkowe

        Returns:
            Dict[str, Any]: Rozpoznany model
        """
        # Rzeczywiste rozpoznawanie modeli rynkowych na podstawie danych
        if data is None or not isinstance(data, dict) or 'price_data' not in data:
            logging.warning("Brak wymaganych danych cenowych dla rozpoznania modelu rynkowego")
            return {"error": "Niewystarczające dane do analizy", "timestamp": time.time()}

        price_data = data['price_data']

        # Analiza trendu
        is_trending = False
        trend_direction = 0

        if len(price_data) >= 10:
            # Prosta analiza trendu - kierunek ruchu ostatnich 10 świec
            up_moves = sum(1 for i in range(1, len(price_data)) if price_data[i] > price_data[i-1])
            down_moves = sum(1 for i in range(1, len(price_data)) if price_data[i] < price_data[i-1])

            # Jeśli ponad 70% ruchów w jednym kierunku - mamy trend
            if up_moves / len(price_data) > 0.7:
                is_trending = True
                trend_direction = 1  # wzrostowy
            elif down_moves / len(price_data) > 0.7:
                is_trending = True
                trend_direction = -1  # spadkowy

        # Analiza zmienności
        volatility = 0
        if len(price_data) >= 2:
            returns = [abs(price_data[i] / price_data[i-1] - 1) for i in range(1, len(price_data))]
            volatility = sum(returns) / len(returns)

        high_volatility = volatility > 0.01  # 1% zmienności jako próg

        # Określenie typu rynku
        if is_trending and trend_direction > 0:
            selected_model = next(m for m in self.model_types if m["id"] == "trending_market")
            confidence = 0.7 + (up_moves / len(price_data) * 0.3)  # 0.7-1.0 zależnie od siły trendu
        elif is_trending and trend_direction < 0:
            selected_model = next(m for m in self.model_types if m["id"] == "trending_market")
            confidence = 0.7 + (down_moves / len(price_data) * 0.3)
        elif high_volatility:
            selected_model = next(m for m in self.model_types if m["id"] == "high_volatility")
            confidence = 0.7 + min(volatility * 10, 0.3)  # Skalowanie 0.7-1.0
        else:
            # Domyślnie konsolidacja
            selected_model = next(m for m in self.model_types if m["id"] == "ranging_market")
            confidence = 0.8

        result = {
            "type": selected_model["id"],
            "name": selected_model["name"],
            "description": selected_model["description"],
            "confidence": confidence,
            "timestamp": time.time()
        }

        self.last_recognition_time = time.time()
        return result

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Zwraca listę dostępnych modeli rynkowych.

        Returns:
            List[Dict[str, Any]]: Lista modeli
        """
        return self.model_types

    def analyze_price_action(self, price_data: List[float]) -> Dict[str, Any]:
        """
        Analizuje akcję cenową.

        Parameters:
            price_data (List[float]): Dane cenowe

        Returns:
            Dict[str, Any]: Wynik analizy
        """
        # Przykładowa implementacja analizy akcji cenowej
        if len(price_data) < 10:
            return {"error": "Za mało danych do analizy"}

        # Symulowana analiza
        is_trending = random.random() > 0.5
        is_volatile = random.random() > 0.7
        is_ranging = not is_trending and random.random() > 0.5

        trend_strength = random.uniform(0.1, 1.0) if is_trending else 0.0
        volatility = random.uniform(0.1, 1.0) if is_volatile else random.uniform(0.0, 0.1)
        range_width = random.uniform(0.01, 0.05) if is_ranging else 0.0

        # Określ dominujący model
        if is_trending and trend_strength > 0.7:
            dominant_model = "trending_market"
        elif is_ranging and range_width > 0.03:
            dominant_model = "ranging_market"
        elif is_volatile and volatility > 0.8:
            dominant_model = "high_volatility"
        else:
            dominant_model = random.choice([m["id"] for m in self.model_types])

        # Znajdź pełne informacje o modelu
        model_info = next((m for m in self.model_types if m["id"] == dominant_model), None)

        return {
            "dominant_model": dominant_model,
            "model_name": model_info["name"] if model_info else dominant_model,
            "trend_strength": trend_strength,
            "volatility": volatility,
            "range_width": range_width,
            "confidence": random.uniform(0.7, 0.95),
            "timestamp": time.time()
        }

if __name__ == "__main__":
    # Przykładowe użycie
    recognizer = ModelRecognizer()

    # Identyfikacja typu modelu
    model_type = recognizer.identify_model_type({"price_data":[1,2,3,4,5,6,7,8,9,10]})
    print(f"Rozpoznany model: {model_type['name']} (typ: {model_type['type']}) z pewnością {model_type['confidence']:.2f}")

    # Analiza akcji cenowej
    price_data = [10, 12, 15, 14, 16, 18, 20, 19, 17, 19, 22, 25]
    price_analysis = recognizer.analyze_price_action(price_data)
    print(f"Analiza akcji cenowej: {price_analysis}")


    # Pobranie dostępnych modeli
    available_models = recognizer.get_available_models()
    print(f"Dostępne modele: {available_models}")
"""
Model Recognizer - rozpoznawanie modeli rynkowych na podstawie danych wejściowych.
"""
import os
import sys
import logging
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

class ModelRecognizer:
    """Klasa do rozpoznawania modeli rynkowych."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Inicjalizacja rozpoznawania modeli.
        
        Args:
            confidence_threshold: Próg pewności dla rozpoznawania modelu
        """
        self.confidence_threshold = confidence_threshold
        self.models_database = self._initialize_models_database()
        
        # Konfiguracja loggera
        self.logger = logging.getLogger('ModelRecognizer')
        self.logger.setLevel(logging.INFO)
        
        # Upewnij się, że katalog logów istnieje
        os.makedirs('logs', exist_ok=True)
        
        # Dodaj handler pliku
        file_handler = logging.FileHandler('logs/model_recognition.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Dodaj handler konsoli
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"ModelRecognizer zainicjalizowany z {len(self.models_database)} modelami w bazie")
    
    def _initialize_models_database(self) -> List[Dict[str, Any]]:
        """
        Inicjalizuje bazę znanych modeli rynkowych.
        
        Returns:
            List[Dict[str, Any]]: Baza modeli
        """
        return [
            {
                'id': 1,
                'name': 'Head and Shoulders',
                'type': 'Reversal Pattern',
                'properties': {
                    'shape': 'three_peaks',
                    'trend_before': 'uptrend',
                    'trend_after': 'downtrend'
                }
            },
            {
                'id': 2,
                'name': 'Double Top',
                'type': 'Reversal Pattern',
                'properties': {
                    'shape': 'two_peaks',
                    'trend_before': 'uptrend',
                    'trend_after': 'downtrend'
                }
            },
            {
                'id': 3,
                'name': 'Bull Flag',
                'type': 'Continuation Pattern',
                'properties': {
                    'shape': 'flag',
                    'trend_before': 'uptrend',
                    'trend_after': 'uptrend'
                }
            },
            {
                'id': 4,
                'name': 'Triangle',
                'type': 'Continuation Pattern',
                'properties': {
                    'shape': 'triangle',
                    'trend_before': 'mixed',
                    'trend_after': 'depends'
                }
            },
            {
                'id': 5,
                'name': 'Cup and Handle',
                'type': 'Continuation Pattern',
                'properties': {
                    'shape': 'cup_handle',
                    'trend_before': 'uptrend',
                    'trend_after': 'uptrend'
                }
            }
        ]
    
    def identify_model_type(self, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Identyfikuje model rynkowy na podstawie danych.
        
        Args:
            market_data: Dane rynkowe do analizy (ceny, wolumen, itp.)
            
        Returns:
            Dict[str, Any]: Zidentyfikowany model lub informacja o błędzie
        """
        if market_data is None or not market_data:
            self.logger.warning("Brak wymaganych danych cenowych dla rozpoznania modelu rynkowego")
            return {
                'success': False,
                'error': 'Brak wymaganych danych cenowych',
                'message': 'Potrzebne są dane OHLCV do rozpoznania modelu'
            }
        
        try:
            # Tutaj byłaby prawdziwa logika analizy danych i rozpoznawania modeli
            # Dla celów demonstracyjnych zwracamy losowy model z bazy
            model = random.choice(self.models_database)
            confidence = random.uniform(0.6, 0.95)
            
            if confidence >= self.confidence_threshold:
                self.logger.info(f"Rozpoznano model: {model['name']} z pewnością {confidence:.2f}")
                return {
                    'success': True,
                    'type': model['type'],
                    'name': model['name'],
                    'id': model['id'],
                    'confidence': confidence,
                    'properties': model['properties'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                self.logger.info(f"Pewność rozpoznania modelu ({confidence:.2f}) poniżej progu ({self.confidence_threshold})")
                return {
                    'success': False,
                    'error': 'Niska pewność rozpoznania',
                    'message': 'Nie można jednoznacznie rozpoznać modelu',
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            self.logger.error(f"Błąd podczas rozpoznawania modelu: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Wystąpił błąd podczas analizy modelu'
            }
