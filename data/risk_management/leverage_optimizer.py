"""
leverage_optimizer.py
------------------
Moduł optymalizacji dźwigni finansowej.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def calculate_optimal_leverage(returns: pd.Series, max_leverage: float = 3.0) -> float:
    """
    Oblicza optymalną dźwignię na podstawie historycznych zwrotów.
    
    Args:
        returns: Seria historycznych zwrotów
        max_leverage: Maksymalna dozwolona dźwignia
        
    Returns:
        float: Optymalna dźwignia
    """
    try:
        # Obliczenie podstawowych statystyk
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Optymalna dźwignia według kryterium Sharpe'a
        if volatility > 0 and mean_return > 0:
            optimal_leverage = mean_return / (volatility * volatility)
            return min(optimal_leverage, max_leverage)
        return 1.0
    
    except Exception as e:
        logger.error(f"Błąd w obliczaniu optymalnej dźwigni: {e}")
        return 1.0

def adjust_leverage_for_risk(
    current_leverage: float,
    risk_metrics: Dict[str, float],
    max_leverage: float = 3.0
) -> float:
    """
    Dostosowuje dźwignię na podstawie metryk ryzyka.
    
    Args:
        current_leverage: Aktualna dźwignia
        risk_metrics: Słownik z metrykami ryzyka
        max_leverage: Maksymalna dozwolona dźwignia
        
    Returns:
        float: Dostosowana dźwignia
    """
    try:
        # Podstawowe ograniczenia
        if current_leverage <= 0:
            return 1.0
        if current_leverage > max_leverage:
            return max_leverage
            
        # Dostosowanie na podstawie VaR
        var = risk_metrics.get('var', 0)
        if var:
            adjustment = 1 - (abs(var) / 0.1)  # 10% VaR jako punkt odniesienia
            return min(current_leverage * adjustment, max_leverage)
            
        return current_leverage
    
    except Exception as e:
        logger.error(f"Błąd w dostosowywaniu dźwigni: {e}")
        return min(current_leverage, max_leverage)

def calculate_leverage_capacity(
    capital: float,
    position_value: float,
    risk_limit: float = 0.1
) -> float:
    """
    Oblicza dostępną pojemność dźwigni.
    
    Args:
        capital: Dostępny kapitał
        position_value: Wartość pozycji
        risk_limit: Limit ryzyka jako % kapitału
        
    Returns:
        float: Maksymalna dostępna dźwignia
    """
    try:
        if position_value <= 0 or capital <= 0:
            return 0.0
            
        # Podstawowe ograniczenie dźwigni
        max_position = capital / risk_limit
        return min(max_position / position_value, 5.0)  # Maksymalna dźwignia 5x
        
    except Exception as e:
        logger.error(f"Błąd w obliczaniu pojemności dźwigni: {e}")
        return 1.0