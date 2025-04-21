"""
pattern_recognition.py
---------------------
Moduł do rozpoznawania wzorców cenowych na wykresach.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def recognize_pattern(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """
    Rozpoznaje wzorce cenowe w danych historycznych.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi OHLCV
        window (int): Okno czasowe do analizy wzorca

    Returns:
        Dict[str, Any]: Słownik zawierający informacje o rozpoznanym wzorcu
    """
    if len(df) < window:
        return {"type": "unknown", "confidence": 0.0}

    try:
        # Przygotuj dane
        close = df['close'].values[-window:]
        high = df['high'].values[-window:]
        low = df['low'].values[-window:]
        
        # Oblicz podstawowe charakterystyki
        trend = _calculate_trend(close)
        volatility = _calculate_volatility(close)
        pattern_strength = _calculate_pattern_strength(high, low, close)
        
        # Rozpoznaj wzorzec
        pattern_info = _identify_pattern(close, high, low, trend, volatility, pattern_strength)
        
        logging.info(f"Rozpoznano wzorzec: {pattern_info['type']} (pewność: {pattern_info['confidence']:.2f})")
        return pattern_info
        
    except Exception as e:
        logging.error(f"Błąd podczas rozpoznawania wzorca: {e}")
        return {"type": "error", "confidence": 0.0}

def _calculate_trend(prices: np.ndarray) -> float:
    """Oblicza siłę trendu."""
    try:
        # Używamy regresji liniowej do określenia trendu
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        return slope / np.mean(prices)  # Normalizacja
    except:
        return 0.0

def _calculate_volatility(prices: np.ndarray) -> float:
    """Oblicza zmienność cen."""
    try:
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    except:
        return 0.0

def _calculate_pattern_strength(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Oblicza siłę wzorca na podstawie zakresów cenowych."""
    try:
        ranges = high - low
        typical_range = np.mean(ranges)
        closing_positions = (close - low) / ranges
        return np.mean(closing_positions)
    except:
        return 0.0

def _identify_pattern(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    trend: float,
    volatility: float,
    pattern_strength: float
) -> Dict[str, Any]:
    """
    Identyfikuje konkretny wzorzec na podstawie charakterystyk cenowych.
    
    Returns:
        Dict zawierający typ wzorca i poziom pewności
    """
    # Definiujemy progi
    TREND_THRESHOLD = 0.001
    VOLATILITY_THRESHOLD = 0.02
    STRENGTH_THRESHOLD = 0.5
    
    if abs(trend) > TREND_THRESHOLD:
        if trend > 0:
            pattern_type = "BullishTrend"
            confidence = min(abs(trend) / TREND_THRESHOLD, 1.0)
        else:
            pattern_type = "BearishTrend"
            confidence = min(abs(trend) / TREND_THRESHOLD, 1.0)
    elif volatility > VOLATILITY_THRESHOLD:
        pattern_type = "HighVolatility"
        confidence = min(volatility / VOLATILITY_THRESHOLD, 1.0)
    elif pattern_strength > STRENGTH_THRESHOLD:
        pattern_type = "Consolidation"
        confidence = pattern_strength
    else:
        pattern_type = "NoPattern"
        confidence = max(0.1, pattern_strength)
    
    return {
        "type": pattern_type,
        "confidence": confidence,
        "metrics": {
            "trend": trend,
            "volatility": volatility,
            "pattern_strength": pattern_strength
        }
    }

# -------------------- Testy jednostkowe --------------------
def unit_test_pattern_recognition():
    """Testy jednostkowe dla modułu pattern_recognition."""
    try:
        # Generujemy przykładowe dane
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # Test 1: Trend wzrostowy
        uptrend = pd.DataFrame({
            'open': np.linspace(100, 150, 50) + np.random.normal(0, 1, 50),
            'high': np.linspace(102, 152, 50) + np.random.normal(0, 1, 50),
            'low': np.linspace(98, 148, 50) + np.random.normal(0, 1, 50),
            'close': np.linspace(100, 150, 50) + np.random.normal(0, 1, 50),
            'volume': np.random.randint(1000, 2000, 50)
        }, index=dates)
        
        result = recognize_pattern(uptrend)
        assert result['type'] == "BullishTrend", "Nie wykryto trendu wzrostowego"
        assert result['confidence'] > 0.5, "Zbyt niska pewność dla wyraźnego trendu"
        
        logging.info("Testy jednostkowe pattern_recognition.py zakończone sukcesem")
        
    except AssertionError as ae:
        logging.error(f"AssertionError w testach pattern_recognition.py: {ae}")
    except Exception as e:
        logging.error(f"Błąd w testach pattern_recognition.py: {e}")
        raise

if __name__ == "__main__":
    try:
        unit_test_pattern_recognition()
    except Exception as e:
        logging.error(f"Testy jednostkowe nie powiodły się: {e}")
        raise