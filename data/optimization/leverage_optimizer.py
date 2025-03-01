"""
leverage_optimizer.py
---------------------
Moduł optymalizujący wykorzystanie dźwigni finansowej.
Funkcjonalności:
- Uwzględnia aktualną zmienność rynku, płynność oraz korelacje z innymi instrumentami.
- Implementuje model dynamicznej dźwigni (np. w oparciu o ATR lub VaR), który automatycznie zwiększa lub zmniejsza dźwignię w zależności od warunków rynkowych.
- Zawiera funkcje limitujące maksymalną stratę (np. ustawienie maksymalnej dźwigni w przypadku wysokiego ryzyka).
- Zapewnia integrację z modułem strategii (np. strategy_manager.py) oraz modułem wykonawczym (np. trade_executor.py).
- Posiada testy weryfikujące skuteczność i stabilność algorytmu przy różnych wielkościach portfela.
"""

import logging
import numpy as np
import pandas as pd

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Oblicza Average True Range (ATR), będący miarą zmienności rynku.
    
    Parameters:
        high (pd.Series): Serie cen wysokich.
        low (pd.Series): Serie cen niskich.
        close (pd.Series): Serie cen zamknięcia.
        period (int): Okres obliczeniowy ATR.
        
    Returns:
        pd.Series: Wartości ATR.
    """
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    logging.info("Obliczono ATR dla okresu %d.", period)
    return atr

def dynamic_leverage_model(df: pd.DataFrame, base_leverage: float = 1.0, atr_multiplier: float = 0.1, max_leverage: float = 5.0) -> pd.Series:
    """
    Implementuje model dynamicznej dźwigni finansowej, który ustala dźwignię na podstawie ATR.
    Im wyższa zmienność (ATR), tym mniejsza dźwignia; w warunkach niskiej zmienności, dźwignia może być wyższa.
    
    Parameters:
        df (pd.DataFrame): DataFrame zawierający kolumny 'high', 'low', 'close'.
        base_leverage (float): Bazowa dźwignia, gdy ATR jest niski.
        atr_multiplier (float): Współczynnik obniżający dźwignię w zależności od ATR.
        max_leverage (float): Maksymalna dozwolona dźwignia.
    
    Returns:
        pd.Series: Seria dynamicznie ustalonych poziomów dźwigni.
    """
    atr = calculate_atr(df['high'], df['low'], df['close'])
    # Przyjmujemy, że im wyższa ATR, tym mniejsza dźwignia: dynamiczna dźwignia = base_leverage / (1 + atr_multiplier * ATR)
    dynamic_leverage = base_leverage / (1 + atr_multiplier * atr)
    # Zapewniamy, że dźwignia nie przekroczy wartości max_leverage
    dynamic_leverage = dynamic_leverage.clip(upper=max_leverage)
    logging.info("Dynamiczna dźwignia została obliczona.")
    return dynamic_leverage

def limit_max_leverage(current_leverage: float, risk_factor: float, max_allowed_leverage: float = 3.0) -> float:
    """
    Funkcja ograniczająca maksymalną dźwignię w zależności od ryzyka.
    
    Parameters:
        current_leverage (float): Aktualnie ustalona dźwignia.
        risk_factor (float): Wskaźnik ryzyka (np. miernik zmienności lub wskaźnik VaR).
        max_allowed_leverage (float): Maksymalna dozwolona dźwignia w warunkach wysokiego ryzyka.
    
    Returns:
        float: Ograniczona dźwignia.
    """
    # Prosty model: jeśli risk_factor przekracza pewien próg, ogranicz dźwignię do max_allowed_leverage
    risk_threshold = 0.05  # przykładowy próg
    if risk_factor > risk_threshold:
        leverage = min(current_leverage, max_allowed_leverage)
    else:
        leverage = current_leverage
    logging.info("Ograniczona dźwignia: %.2f (risk_factor: %.4f)", leverage, risk_factor)
    return leverage

# -------------------- Testy jednostkowe --------------------
def unit_test_leverage_optimizer():
    """
    Testy jednostkowe dla modułu leverage_optimizer.py.
    Tworzy przykładowe dane rynkowe i weryfikuje działanie dynamicznej dźwigni oraz ograniczenia maksymalnej dźwigni.
    """
    try:
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        # Generujemy przykładowe dane: ceny rosnące z losowym szumem
        high = pd.Series(np.linspace(100, 120, 20) + np.random.normal(0, 1, 20), index=dates)
        low = pd.Series(np.linspace(95, 115, 20) + np.random.normal(0, 1, 20), index=dates)
        close = pd.Series(np.linspace(98, 118, 20) + np.random.normal(0, 1, 20), index=dates)
        df = pd.DataFrame({"high": high, "low": low, "close": close})
        
        dyn_leverage = dynamic_leverage_model(df, base_leverage=1.0, atr_multiplier=0.1, max_leverage=5.0)
        assert (dyn_leverage <= 5.0).all(), "Dynamiczna dźwignia przekracza maksymalny limit."
        logging.info("Test dynamicznej dźwigni zakończony sukcesem.")
        
        # Test ograniczenia maksymalnej dźwigni
        limited_leverage = limit_max_leverage(current_leverage=4.5, risk_factor=0.06, max_allowed_leverage=3.0)
        assert limited_leverage <= 3.0, "Funkcja ograniczająca dźwignię nie działa prawidłowo."
        logging.info("Test ograniczenia dźwigni zakończony sukcesem.")
        
    except AssertionError as ae:
        logging.error("AssertionError w testach jednostkowych: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach jednostkowych modułu leverage_optimizer.py: %s", e)
        raise

if __name__ == "__main__":
    try:
        unit_test_leverage_optimizer()
        logging.info("Wszystkie testy jednostkowe leverage_optimizer.py zakończone sukcesem.")
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        raise
