"""
mean_reversion_strategy.py
------------------------
Implementacja strategii mean reversion.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """Strategia powrotu do średniej."""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'ma_period': 20,
            'std_dev': 2.0,
            'exit_std_dev': 0.5,
            'lookback_period': 50
        }
        if params:
            default_params.update(params)
        super().__init__("mean_reversion", default_params)
        
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generuje sygnały handlowe na podstawie odchylenia od średniej.
        
        Args:
            market_data: DataFrame z danymi rynkowymi
            
        Returns:
            Dict zawierający sygnał i szczegóły
        """
        if not self.validate_data(market_data):
            return {"signal": 0, "error": "Nieprawidłowe dane wejściowe"}
            
        try:
            # Obliczenie średniej kroczącej i odchylenia standardowego
            ma = market_data['close'].rolling(window=self.params['ma_period']).mean()
            std = market_data['close'].rolling(window=self.params['ma_period']).std()
            
            # Obliczenie liczby odchyleń standardowych od średniej
            z_score = (market_data['close'] - ma) / std
            
            # Generowanie sygnału
            signal = 0
            if z_score.iloc[-1] < -self.params['std_dev']:
                signal = 1  # Sygnał kupna gdy cena jest znacząco poniżej średniej
            elif z_score.iloc[-1] > self.params['std_dev']:
                signal = -1  # Sygnał sprzedaży gdy cena jest znacząco powyżej średniej
                
            # Obliczenie dodatkowych metryk
            historical_z_scores = z_score.rolling(window=self.params['lookback_period']).mean()
            mean_reversion_strength = 1 - abs(historical_z_scores.iloc[-1])
                
            return {
                "signal": signal,
                "z_score": z_score.iloc[-1],
                "ma": ma.iloc[-1],
                "current_price": market_data['close'].iloc[-1],
                "std_dev": std.iloc[-1],
                "mean_reversion_strength": mean_reversion_strength,
                "confidence": abs(z_score.iloc[-1]) / self.params['std_dev']
            }
            
        except Exception as e:
            self.logger.error(f"Błąd w generowaniu sygnału mean reversion: {e}")
            return {"signal": 0, "error": str(e)}