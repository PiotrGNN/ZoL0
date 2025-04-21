import logging
import unittest
import numpy as np
import pandas as pd
from typing import Any, Dict, List

# Import modułów zarządzania ryzykiem
from ..risk_management.portfolio_risk import PortfolioRiskManager
from ..risk_management.position_sizing import (
    dynamic_position_size,
    fixed_fractional_position_size,
    kelly_criterion_position_size,
    risk_parity_position_size
)
from ..risk_management.risk_metrics import (
    calculate_risk_metrics,
    calculate_var,
    calculate_cvar
)
from ..risk_management.stop_loss_manager import (
    atr_based_stop_loss,
    fixed_stop_loss,
    time_based_stop_loss,
    trailing_stop_loss
)
from ..risk_management.leverage_optimizer import (
    dynamic_leverage_model,
    limit_max_leverage
)

"""
test_risk_management.py
-----------------------
Testy jednostkowe dla modułów zarządzania ryzykiem:
  - stop_loss_manager.py,
  - portfolio_risk.py,
  - position_sizing.py,
  - leverage_optimizer.py.

Weryfikujemy poprawność obliczeń funkcji stop-loss (fixed, trailing, ATR-based, time-based)
oraz funkcji optymalizacji dźwigni finansowej.
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TestRiskManagementModules(unittest.TestCase):
    """
    Testy jednostkowe dla modułów zarządzania ryzykiem.
    Sprawdzamy funkcje stop-loss oraz optymalizacji dźwigni.
    """

    def setUp(self) -> None:
        np.random.seed(42)
        self.dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.df_prices = pd.DataFrame({
            "high": np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            "low": np.linspace(90, 140, 100) + np.random.normal(0, 1, 100),
            "close": np.linspace(95, 145, 100) + np.random.normal(0, 1, 100)
        }, index=self.dates)
        self.returns = self.df_prices["close"].pct_change().dropna()

    def test_stop_loss_functions(self) -> None:
        entry_price: float = 100
        fixed_distance: float = 10

        if fixed_stop_loss:
            self.assertAlmostEqual(
                fixed_stop_loss(entry_price, fixed_distance),
                90,
                places=2,
                msg="Fixed stop-loss powinien zwracać 90 przy entry_price=100 i fixed_distance=10."
            )
        else:
            logging.warning("Funkcja `fixed_stop_loss` nie została zaimportowana.")

        highest_price: float = 110
        trailing_pct: float = 0.05
        expected_sl: float = 109.95
        if trailing_stop_loss:
            self.assertAlmostEqual(
                trailing_stop_loss(100, highest_price, trailing_pct),
                expected_sl,
                places=2,
                msg="Trailing stop-loss powinien zwracać około 109.95 przy entry_price=100, highest_price=110 i trailing_pct=0.05."
            )
        else:
            logging.warning("Funkcja `trailing_stop_loss` nie została zaimportowana.")

        if atr_based_stop_loss:
            atr_adj = atr_based_stop_loss(self.df_prices, atr_multiplier=1.5)
            # Sprawdzamy, czy wynik to obiekt Series i wszystkie wartości są dodatnie.
            self.assertTrue(
                isinstance(atr_adj, pd.Series) and (atr_adj > 0).all(),
                "ATR-based stop-loss powinien zwracać Series z wartościami > 0."
            )
        else:
            logging.warning("Funkcja `atr_based_stop_loss` nie została zaimportowana.")

        if time_based_stop_loss:
            entry_time = pd.Timestamp("2023-01-01 09:00:00")
            current_time = pd.Timestamp("2023-01-01 10:00:00")
            max_hold_time = pd.Timedelta(minutes=60)
            self.assertTrue(
                time_based_stop_loss(entry_time, max_hold_time, current_time),
                "Time-based stop-loss powinien zwracać True, gdy czas od wejścia przekracza 60 minut."
            )
        else:
            logging.warning("Funkcja `time_based_stop_loss` nie została zaimportowana.")

    def test_leverage_optimizer_functions(self) -> None:
        if dynamic_leverage_model:
            dyn_leverage = dynamic_leverage_model(self.df_prices, base_leverage=1.0, atr_multiplier=0.1, max_leverage=5.0)
            self.assertTrue((dyn_leverage <= 5.0).all(), "Dynamic leverage model powinien zwracać wartości nie przekraczające 5.0.")
        else:
            logging.warning("Funkcja `dynamic_leverage_model` nie została zaimportowana.")

        if limit_max_leverage:
            limited_leverage = limit_max_leverage(current_leverage=4.5, risk_factor=0.06, max_allowed_leverage=3.0)
            self.assertLessEqual(limited_leverage, 3.0, "Limit max leverage powinien zwracać wartość <= 3.0.")
        else:
            logging.warning("Funkcja `limit_max_leverage` nie została zaimportowana.")


if __name__ == "__main__":
    unittest.main()
