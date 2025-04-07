import logging
import unittest

import numpy as np
import pandas as pd

from data.strategies.breakout_strategy import breakout_strategy
from data.strategies.mean_reversion import generate_mean_reversion_signal
from data.strategies.trend_following import generate_trend_following_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class TestStrategies(unittest.TestCase):
    """Testy jednostkowe dla strategii tradingowych."""
    
    def setUp(self):
        """Przygotowanie danych testowych."""
        self.dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)
        
    def test_mean_reversion(self):
        """Test strategii mean reversion."""
        # Generujemy dane z trendem rewersyjnym: ceny oscylują wokół 100
        close_prices = 100 + np.random.normal(0, 2, 50)
        volume = np.random.randint(1000, 1500, 50)
        df = pd.DataFrame({"close": close_prices, "volume": volume}, index=self.dates)
        
        signal = generate_mean_reversion_signal(
            df, window=10, zscore_threshold=1.5, volume_filter=1100
        )
        
        self.assertIsNotNone(signal)
        self.assertTrue(isinstance(signal, pd.Series))
        self.assertEqual(len(signal), len(df))
        self.assertTrue(signal.isin([-1, 0, 1]).all(), "Sygnały powinny mieć wartości -1, 0 lub 1")
        
    def test_trend_following(self):
        """Test strategii podążania za trendem."""
        # Dane z trendem wzrostowym
        close_prices = np.linspace(100, 150, 50) + np.random.normal(0, 1, 50)
        df = pd.DataFrame({"close": close_prices}, index=self.dates)
        
        signal = generate_trend_following_signal(df, short_window=5, long_window=20)
        
        self.assertIsNotNone(signal)
        self.assertTrue(isinstance(signal, pd.Series))
        self.assertEqual(len(signal), len(df))
        
    def test_breakout_strategy(self):
        """Test strategii przełamania."""
        # Dane z nagłym wzrostem
        close_prices = np.ones(50) * 100
        close_prices[30:] = 120  # Gwałtowny wzrost po 30 dniu
        df = pd.DataFrame({"close": close_prices}, index=self.dates)
        
        signal = breakout_strategy(df, window=10)
        
        self.assertIsNotNone(signal)
        self.assertTrue(isinstance(signal, pd.Series))
        self.assertEqual(len(signal), len(df))
        # Powinna być przynajmniej jedna wartość 1 (sygnał kupna)
        self.assertTrue((signal == 1).any())

if __name__ == "__main__":
    unittest.main()

class TestStrategies(unittest.TestCase):
    """Testy jednostkowe strategii inwestycyjnych."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.dates = pd.date_range("2023-01-01", periods=50, freq="h")
        base_prices = np.linspace(100, 105, 50)
        noise = np.random.normal(0, 0.5, 50)

        self.df = pd.DataFrame(
            {
                "close": base_prices + noise,
                "high": base_prices + noise + np.random.uniform(0.5, 1.5, 50),
                "low": base_prices + noise - np.random.uniform(0.5, 1.5, 50),
                "volume": np.random.randint(1000, 1500, 50),
            },
            index=self.dates,
        )

        # Wyraźnie obniż ceny, aby łatwiej osiągnąć breakout
        self.df.iloc[-10:, [0, 1, 2]] -= 10.0
        self.df.iloc[-10:, 3] = int(self.df["volume"].mean() // 3)

    def test_breakout_strategy_buy(self) -> None:
        df_copy = self.df.copy()
        window_size = 10
        recent_high = df_copy["high"].iloc[-window_size:].max()

        # Bardzo silne przebicie
        df_copy.at[df_copy.index[-1], "high"] = recent_high + 20.0
        df_copy.at[df_copy.index[-1], "close"] = recent_high + 19.0
        df_copy.at[df_copy.index[-1], "volume"] = int(df_copy["volume"].mean() * 50)

        result = breakout_strategy(df_copy, window=window_size, volume_threshold=1.0)

        self.assertEqual(
            result["signal"], 1, "Powinien zostać wygenerowany sygnał kupna (1)."
        )

    def test_breakout_strategy_sell(self) -> None:
        df_copy = self.df.copy()
        window_size = 10
        recent_low = df_copy["low"].iloc[-window_size:].min()

        # Bardzo silne przebicie w dół
        df_copy.at[df_copy.index[-1], "low"] = recent_low - 20.0
        df_copy.at[df_copy.index[-1], "close"] = recent_low - 19.0
        df_copy.at[df_copy.index[-1], "volume"] = int(df_copy["volume"].mean() * 50)

        result = breakout_strategy(df_copy, window=window_size, volume_threshold=1.0)

        self.assertEqual(
            result["signal"], -1, "Powinien zostać wygenerowany sygnał sprzedaży (-1)."
        )

    def test_mean_reversion_signal(self) -> None:
        signal = generate_mean_reversion_signal(
            self.df, window=10, zscore_threshold=1.5, volume_filter=1100
        )
        self.assertIsInstance(signal, pd.Series)
        self.assertFalse(signal.isnull().any())
        self.assertTrue(set(signal.unique()).issubset({-1, 0, 1}))

    def test_trend_following_signal(self) -> None:
        signal = generate_trend_following_signal(
            self.df, adx_threshold=25, macd_threshold=0, channel_window=10, liquidity_threshold=1000
        )
        self.assertIsInstance(signal, pd.Series)
        self.assertFalse(signal.isnull().any())
        self.assertTrue(set(signal.unique()).issubset({-1, 0, 1}))


if __name__ == "__main__":
    unittest.main()
