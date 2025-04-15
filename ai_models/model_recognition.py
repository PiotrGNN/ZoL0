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


    def prepare_input_data(self, data: Any) -> Dict[str, Any]:
        """
        Przygotowuje dane wejściowe do analizy.

        Parameters:
            data (Any): Dane wejściowe w różnych formatach

        Returns:
            Dict[str, Any]: Przygotowane dane w formacie słownika
        """
        import numpy as np
        import pandas as pd

        # Jeśli dane są None, zwróć pusty słownik z błędem
        if data is None:
            return {'error': 'Brak danych wejściowych'}

        # Jeśli dane są już słownikiem, sprawdź czy mają wymagane pola
        if isinstance(data, dict):
            # Jeśli brak pola price_data, sprawdź inne możliwe pola
            if 'price_data' not in data:
                if 'close' in data and isinstance(data['close'], (list, np.ndarray)):
                    return {'price_data': data['close']}
                elif 'prices' in data and isinstance(data['prices'], (list, np.ndarray)):
                    return {'price_data': data['prices']}
                elif 'data' in data and isinstance(data['data'], (list, np.ndarray)):
                    return {'price_data': data['data']}
                # Jeśli nie znaleziono odpowiedniego pola, użyj pierwszego pola liczbowego
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        if all(isinstance(x, (int, float)) for x in value):
                            self.logger.info(f"Używam pola '{key}' jako danych cenowych")
                            return {'price_data': value}
            return data

        # Jeśli dane są array lub listą, uznaj za price_data
        elif isinstance(data, (list, np.ndarray)):
            return {'price_data': data}

        # Jeśli dane są DataFrame, przekonwertuj na słownik
        elif isinstance(data, pd.DataFrame):
            try:
                if 'close' in data.columns:
                    return {'price_data': data['close'].tolist()}
                else:
                    # Znajdź pierwszą kolumnę liczbową
                    numeric_cols = data.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        return {'price_data': data[numeric_cols[0]].tolist()}
                    return {'price_data': data.iloc[:, 0].tolist()}
            except Exception as e:
                self.logger.warning(f"Nie udało się przekonwertować DataFrame na słownik: {e}")

        # Jeśli nic nie pasuje, spróbuj skonwertować dane do postaci numpy
        try:
            array_data = np.array(data)
            if array_data.size > 0:
                return {'price_data': array_data.flatten().tolist()}
        except:
            pass

        return {'error': 'Nieobsługiwany format danych wejściowych'}

    def predict(self, data: Optional[Any]) -> Dict[str, Any]:
        """
        Przewiduje typ modelu rynkowego na podstawie danych.

        Parameters:
            data (Optional[Any]): Dane rynkowe w dowolnym formacie

        Returns:
            Dict[str, Any]: Rozpoznany model
        """
        if data is not None:
            prepared_data = self.prepare_input_data(data)
            return self.identify_model_type(prepared_data)
        return self.identify_model_type(None)

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

    def predict(self, data):
        """
        Przewiduje jaki model rynkowy najlepiej pasuje do aktualnych danych.

        Args:
            data: Dane OHLCV w formacie słownika lub DataFrame

        Returns:
            dict: Wynik predykcji zawierający nazwę najlepiej pasującego modelu i pewność.
        """
        try:
            # Jeśli dane są None lub puste, używamy identify_model_type bez danych
            if data is None:
                return self.identify_model_type(None)

            # Przygotowujemy dane przed użyciem
            try:
                # Importujemy funkcję prepare_data_for_model z model_training
                from ai_models.model_training import prepare_data_for_model
                
                # Konwertujemy dane wejściowe do odpowiedniego formatu
                prepared_data = prepare_data_for_model(data)
                
                # Sprawdzamy czy dane są niepuste
                if prepared_data is not None and len(prepared_data) > 0:
                    if hasattr(self, 'model') and self.model is not None:
                        # Używamy modelu ML jeśli istnieje
                        prediction = self.model.predict(prepared_data)
                        return prediction  # Zakładając że prediction jest już w odpowiednim formacie słownika
                    else:
                        # Używamy prostszej metody identyfikacji
                        return self.identify_model_type({"price_data": prepared_data})
                else:
                    return self.identify_model_type(None)
            except ImportError:
                # Jeśli nie możemy zaimportować funkcji prepare_data_for_model
                return self.identify_model_type(data)
            except Exception as prep_error:
                logging.warning(f"Błąd podczas przygotowania danych: {prep_error}")
                return self.identify_model_type(data)

        except Exception as e:
            return {"error": f"Błąd podczas predykcji: {e}", "success": False}