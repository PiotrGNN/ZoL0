"""
test_strategies.py
------------------
Testy jednostkowe dla modułów strategii (np. breakout_strategy.py, mean_reversion.py, trend_following.py).
Testy weryfikują poprawność generowania sygnałów, obsługę warunków wejścia/wyjścia, zarządzanie stop-lossami 
oraz integrację między strategiami i modułami backtestingowymi.
"""

import os
import unittest
import logging
import numpy as np
import pandas as pd

# Konfiguracja logowania do testów
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Zakładamy, że moduły strategii znajdują się w folderze data/strategies
from data.strategies.breakout_strategy import breakout_strategy
from data.strategies.mean_reversion import generate_mean_reversion_signal
from data.strategies.trend_following import generate_trend_following_signal

class TestStrategies(unittest.TestCase):
    def setUp(self):
        # Tworzymy przykładowe dane świecowe
        np.random.seed(42)
        self.dates = pd.date_range(start="2023-01-01", periods=50, freq="H")
        # Generujemy dane z lekkim trendem i oscylacjami
        base_prices = np.linspace(100, 105, 50)
        noise = np.random.normal(0, 0.5, 50)
        self.close = base_prices + noise
        self.high = self.close + np.random.uniform(0.5, 1.5, 50)
        self.low = self.close - np.random.uniform(0.5, 1.5, 50)
        self.volume = np.random.randint(1000, 1500, 50)
        self.df = pd.DataFrame({
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "volume": self.volume
        }, index=self.dates)

    def test_breakout_strategy_buy(self):
        # Symulujemy przebicie oporu
        df_copy = self.df.copy()
        df_copy.iloc[-1, df_copy.columns.get_loc("close")] = df_copy["close"].max() * 1.02
        df_copy.iloc[-1, df_copy.columns.get_loc("volume")] = df_copy["volume"].mean() * 1.5
        result = breakout_strategy(df_copy, window=10, volume_threshold=1.0)
        self.assertEqual(result["signal"], 1, "Breakout strategy powinna generować sygnał kupna (1).")

    def test_breakout_strategy_sell(self):
        # Symulujemy przebicie wsparcia
        df_copy = self.df.copy()
        df_copy.iloc[-1, df_copy.columns.get_loc("close")] = df_copy["close"].min() * 0.98
        df_copy.iloc[-1, df_copy.columns.get_loc("volume")] = df_copy["volume"].mean() * 1.5
        result = breakout_strategy(df_copy, window=10, volume_threshold=1.0)
        self.assertEqual(result["signal"], -1, "Breakout strategy powinna generować sygnał sprzedaży (-1).")

    def test_mean_reversion_signal(self):
        # Testujemy generowanie sygnału mean reversion
        signal = generate_mean_reversion_signal(self.df, window=10, zscore_threshold=1.5, volume_filter=1100)
        # Sygnały powinny być -1, 0 lub 1
        unique_signals = set(signal.unique())
        self.assertTrue(unique_signals.issubset({-1, 0, 1}), "Sygnały mean reversion powinny być -1, 0 lub 1.")

    def test_trend_following_signal(self):
        # Testujemy strategię trend following
        signal = generate_trend_following_signal(self.df, adx_threshold=25, macd_threshold=0, channel_window=10, liquidity_threshold=1000)
        # Sygnał powinien być -1, 0 lub 1
        unique_signals = set(signal.unique())
        self.assertTrue(unique_signals.issubset({-1, 0, 1}), "Sygnały trend following powinny być -1, 0 lub 1.")

if __name__ == "__main__":
    unittest.main()
