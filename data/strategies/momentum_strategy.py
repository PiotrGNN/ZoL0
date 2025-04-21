"""
momentum_strategy.py
------------------
Implementacja strategii momentum.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    """Strategia oparta na momentum cenowym."""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 20,
            'momentum_threshold': 0.02,
            'volatility_window': 10
        }
        if params:
            default_params.update(params)
        super().__init__("momentum", default_params)
        
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generuje sygnały handlowe na podstawie momentum.
        
        Args:
            market_data: DataFrame z danymi rynkowymi
            
        Returns:
            Dict zawierający sygnał i szczegóły
        """
        if not self.validate_data(market_data):
            return {"signal": 0, "error": "Nieprawidłowe dane wejściowe"}
            
        try:
            # Obliczenie momentum
            returns = market_data['close'].pct_change(self.params['lookback_period'])
            volatility = market_data['close'].pct_change().rolling(
                window=self.params['volatility_window']
            ).std()
            
            # Normalizacja momentum przez zmienność
            momentum = returns / volatility
            
            # Generowanie sygnału
            signal = 0
            if momentum.iloc[-1] > self.params['momentum_threshold']:
                signal = 1  # Sygnał kupna
            elif momentum.iloc[-1] < -self.params['momentum_threshold']:
                signal = -1  # Sygnał sprzedaży
                
            return {
                "signal": signal,
                "momentum": momentum.iloc[-1],
                "volatility": volatility.iloc[-1],
                "threshold": self.params['momentum_threshold'],
                "confidence": abs(momentum.iloc[-1]) / self.params['momentum_threshold']
            }
            
        except Exception as e:
            self.logger.error(f"Błąd w generowaniu sygnału momentum: {e}")
            return {"signal": 0, "error": str(e)}