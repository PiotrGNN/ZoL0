import logging
import unittest
import numpy as np
import pandas as pd
from data.strategies.breakout_strategy import breakout_strategy
from data.strategies.mean_reversion import generate_mean_reversion_signal
from data.strategies.trend_following import generate_trend_following_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class TestStrategies(unittest.TestCase):
    """Testy jednostkowe strategii inwestycyjnych."""

    def setUp(self) -> None:
        """Przygotowanie danych testowych."""
        np.random.seed(42)
        self.dates = pd.date_range(start="2023-01-01", periods=50, freq="h")
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

        # Create breakout pattern in the data
        self.df.iloc[-10:, [0, 1, 2]] -= 10.0  # Drop in price for last 10 periods
        self.df.iloc[-10:, 3] = int(self.df["volume"].mean() * 2)  # Volume spike

    def test_breakout_strategy_buy(self) -> None:
        """Test sygnału kupna dla strategii przebicia."""
        signal = breakout_strategy(
            self.df, lookback_period=5, std_multiplier=2.0, volume_multiplier=1.5
        )
        self.assertIsNotNone(signal)
        self.assertTrue(any(signal > 0), "Powinien wystąpić przynajmniej jeden sygnał kupna")

    def test_breakout_strategy_sell(self) -> None:
        """Test sygnału sprzedaży dla strategii przebicia."""
        # Odwróć dane dla sygnału sprzedaży
        self.df = self.df * -1
        signal = breakout_strategy(
            self.df, lookback_period=5, std_multiplier=2.0, volume_multiplier=1.5
        )
        self.assertIsNotNone(signal)
        self.assertTrue(any(signal < 0), "Powinien wystąpić przynajmniej jeden sygnał sprzedaży")

    def test_mean_reversion_signal(self) -> None:
        """Test sygnałów dla strategii powrotu do średniej."""
        signal = generate_mean_reversion_signal(
            self.df, window=20, std_dev=2.0
        )
        self.assertIsNotNone(signal)
        self.assertTrue(isinstance(signal, pd.Series))
        self.assertEqual(len(signal), len(self.df))
        
    def test_trend_following_signal(self) -> None:
        """Test sygnałów dla strategii podążania za trendem."""
        signal = generate_trend_following_signal(
            self.df, short_window=5, long_window=20
        )
        self.assertIsNotNone(signal)
        self.assertTrue(isinstance(signal, pd.Series))
        self.assertEqual(len(signal), len(self.df))
        # Verify that signals are within valid range (-1, 0, 1)
        self.assertTrue(all(s in [-1, 0, 1] for s in signal.unique()))

if __name__ == "__main__":
    unittest.main()
