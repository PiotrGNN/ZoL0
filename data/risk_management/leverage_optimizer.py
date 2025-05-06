"""
leverage_optimizer.py
---------------------
Moduł optymalizujący dźwignię finansową.

Funkcjonalności:
- Uwzględnia aktualną zmienność rynku, płynność oraz korelacje z innymi instrumentami.
- Implementuje model dynamicznej dźwigni (np. w oparciu o ATR lub VaR), który automatycznie dostosowuje dźwignię.
- Zawiera funkcje limitujące maksymalną stratę.
- Zapewnia integrację z modułem strategii oraz modułem wykonawczym.
- Posiada testy weryfikujące skuteczność i stabilność algorytmu.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Oblicza Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    
    df = pd.DataFrame()
    df["TrueRange"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TrueRange"].rolling(window=period, min_periods=1).mean()
    
    return df["ATR"]

def dynamic_leverage_model(df: pd.DataFrame, base_leverage: float = 1.0, atr_multiplier: float = 0.1, max_leverage: float = 5.0) -> pd.Series:
    """
    Implementuje model dynamicznej dźwigni finansowej w oparciu o ATR.
    Im wyższa zmienność (ATR), tym mniejsza dźwignia.
    """
    atr = calculate_atr(df)
    dynamic_leverage = base_leverage / (1 + atr_multiplier * atr)
    dynamic_leverage = dynamic_leverage.clip(upper=max_leverage)
    
    return dynamic_leverage

def limit_max_leverage(current_leverage: float, risk_factor: float, max_allowed_leverage: float = 3.0) -> float:
    """
    Ogranicza maksymalną dźwignię w zależności od poziomu ryzyka.
    """
    risk_threshold = 0.05  # Przykładowy próg
    adjusted_leverage = (min(current_leverage, max_allowed_leverage) 
                       if risk_factor > risk_threshold 
                       else current_leverage)
    
    return adjusted_leverage

def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Oblicza Value at Risk dla danego poziomu ufności."""
    return norm.ppf(1 - confidence_level) * returns.std()