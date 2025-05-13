"""
Momentum Strategy Implementation
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

class MomentumStrategy:
    """
    Implements a simple momentum trading strategy.
    """
    def __init__(self, symbol: str, lookback: int = 20, entry_threshold: float = 1.0, exit_threshold: float = 0.5):
        self.symbol = symbol
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.last_signal = None

    def generate_signal(self, price_series: pd.Series) -> int:
        """
        Generate trading signal based on momentum logic.
        Returns:
            1 for long, -1 for short, 0 for no action
        """
        if len(price_series) < self.lookback:
            return 0  # Not enough data
        window = price_series[-self.lookback:]
        momentum = window.iloc[-1] - window.iloc[0]
        if self.position == 0:
            if momentum > self.entry_threshold:
                self.position = 1
                self.last_signal = 'long'
                return 1  # Long entry
            elif momentum < -self.entry_threshold:
                self.position = -1
                self.last_signal = 'short'
                return -1  # Short entry
        elif self.position == 1:
            if momentum < self.exit_threshold:
                self.position = 0
                self.last_signal = 'exit_long'
                return 0  # Exit long
        elif self.position == -1:
            if momentum > -self.exit_threshold:
                self.position = 0
                self.last_signal = 'exit_short'
                return 0  # Exit short
        return 0  # Hold

    def backtest(self, price_series: pd.Series) -> Dict[str, Any]:
        """
        Backtest the momentum strategy on a price series.
        Returns a dict with performance metrics.
        """
        signals = []
        positions = []
        returns = []
        self.position = 0
        for i in range(self.lookback, len(price_series)):
            window = price_series.iloc[i-self.lookback:i]
            current_price = price_series.iloc[i]
            signal = self.generate_signal(price_series.iloc[:i+1])
            signals.append(signal)
            positions.append(self.position)
            if len(positions) > 1:
                ret = (current_price - price_series.iloc[i-1]) / price_series.iloc[i-1]
                returns.append(ret * positions[-2])
            else:
                returns.append(0)
        total_return = np.sum(returns)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if np.std(returns) > 0 else 0
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'signals': signals,
            'positions': positions,
            'returns': returns
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'lookback': self.lookback,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'position': self.position,
            'last_signal': self.last_signal
        }
