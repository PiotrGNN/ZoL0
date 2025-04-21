"""
volatility_strategy.py
--------------------
Implementacja strategii opartej na zmienności.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class VolatilityStrategy(BaseStrategy):
    """Strategia oparta na zmienności rynkowej."""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'volatility_window': 20,
            'volatility_threshold': 0.015,
            'mean_reversion_threshold': 0.5,
            'entry_multiplier': 1.5
        }
        if params:
            default_params.update(params)
        super().__init__("volatility", default_params)
        
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generuje sygnały handlowe na podstawie wzorców zmienności.
        
        Args:
            market_data: DataFrame z danymi rynkowymi
            
        Returns:
            Dict zawierający sygnał i szczegóły
        """
        if not self.validate_data(market_data):
            return {"signal": 0, "error": "Nieprawidłowe dane wejściowe"}
            
        try:
            # Obliczenie zmienności
            returns = market_data['close'].pct_change()
            current_volatility = returns.rolling(window=self.params['volatility_window']).std()
            historical_volatility = returns.rolling(window=self.params['volatility_window'] * 2).std()
            
            # Obliczenie względnej zmienności
            relative_volatility = current_volatility / historical_volatility
            
            # Detekcja ekstremów zmienności
            is_high_volatility = relative_volatility.iloc[-1] > self.params['entry_multiplier']
            is_low_volatility = relative_volatility.iloc[-1] < 1/self.params['entry_multiplier']
            
            # Generowanie sygnału
            signal = 0
            if is_high_volatility:
                if returns.iloc[-1] < -self.params['mean_reversion_threshold']:
                    signal = 1  # Sygnał kupna przy wysokiej zmienności i spadku
            elif is_low_volatility:
                if abs(returns.iloc[-1]) > self.params['volatility_threshold']:
                    signal = -1  # Sygnał sprzedaży przy niskiej zmienności i ruchu
                    
            return {
                "signal": signal,
                "current_volatility": current_volatility.iloc[-1],
                "historical_volatility": historical_volatility.iloc[-1],
                "relative_volatility": relative_volatility.iloc[-1],
                "confidence": abs(relative_volatility.iloc[-1] - 1)
            }
            
        except Exception as e:
            self.logger.error(f"Błąd w generowaniu sygnału zmienności: {e}")
            return {"signal": 0, "error": str(e)}