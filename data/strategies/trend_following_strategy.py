"""
trend_following_strategy.py
-------------------------
Implementacja strategii podążania za trendem.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """Strategia podążania za trendem."""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'short_ma': 20,
            'long_ma': 50,
            'atr_period': 14,
            'trend_strength_threshold': 0.02
        }
        if params:
            default_params.update(params)
        super().__init__("trend_following", default_params)
        
    def calculate_atr(self, market_data: pd.DataFrame, period: int) -> pd.Series:
        """Oblicza Average True Range."""
        high = market_data['high']
        low = market_data['low']
        close = market_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generuje sygnały handlowe na podstawie trendu.
        
        Args:
            market_data: DataFrame z danymi rynkowymi
            
        Returns:
            Dict zawierający sygnał i szczegóły
        """
        if not self.validate_data(market_data):
            return {"signal": 0, "error": "Nieprawidłowe dane wejściowe"}
            
        try:
            # Obliczenie średnich kroczących
            short_ma = market_data['close'].rolling(window=self.params['short_ma']).mean()
            long_ma = market_data['close'].rolling(window=self.params['long_ma']).mean()
            
            # Obliczenie ATR dla oceny zmienności
            atr = self.calculate_atr(market_data, self.params['atr_period'])
            
            # Obliczenie siły trendu
            trend_strength = (short_ma - long_ma) / long_ma
            
            # Generowanie sygnału
            signal = 0
            if (trend_strength.iloc[-1] > self.params['trend_strength_threshold'] and
                short_ma.iloc[-1] > long_ma.iloc[-1]):
                signal = 1  # Sygnał kupna w trendzie wzrostowym
            elif (trend_strength.iloc[-1] < -self.params['trend_strength_threshold'] and
                  short_ma.iloc[-1] < long_ma.iloc[-1]):
                signal = -1  # Sygnał sprzedaży w trendzie spadkowym
                
            return {
                "signal": signal,
                "trend_strength": trend_strength.iloc[-1],
                "short_ma": short_ma.iloc[-1],
                "long_ma": long_ma.iloc[-1],
                "atr": atr.iloc[-1],
                "confidence": abs(trend_strength.iloc[-1]) / self.params['trend_strength_threshold']
            }
            
        except Exception as e:
            self.logger.error(f"Błąd w generowaniu sygnału trend following: {e}")
            return {"signal": 0, "error": str(e)}