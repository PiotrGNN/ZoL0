"""
test_risk_management.py
----------------------
Tests for risk management modules including position sizing,
leverage optimization, and stop-loss calculation.
"""

from typing import Optional
import numpy as np
import pandas as pd
from data.tests import BaseTestCase

from data.risk_management.stop_loss_manager import (
    calculate_stop_loss,
    calculate_trailing_stop,
)
from data.risk_management.position_sizing import (
    calculate_position_size,
    risk_parity_position_size,
)
from data.risk_management.leverage_optimizer import (
    dynamic_leverage_model,
    limit_max_leverage,
    value_at_risk
)

class TestRiskManagement(BaseTestCase):
    """Test risk management functionality."""

    def setUp(self):
        """Initialize test data."""
        super().setUp()
        self.df = self.generate_test_data(periods=50)
        # Add some volatility patterns
        self.df.loc[20:30, "close"] *= 1.1  # Price surge
        self.df.loc[35:40, "close"] *= 0.9  # Price drop
        self.returns = self.df["close"].pct_change().dropna()

    def test_stop_loss_calculation(self):
        """Test stop loss calculations."""
        if not all([calculate_stop_loss, calculate_trailing_stop]):
            self.skipTest("Stop loss functions not available")

        # Test basic stop loss calculations
        test_cases = [
            {"entry": 100.0, "risk": 0.02, "type": "long", "expected_min": 97.0, "expected_max": 98.5},
            {"entry": 100.0, "risk": 0.02, "type": "short", "expected_min": 101.5, "expected_max": 103.0},
            {"entry": 50.0, "risk": 0.05, "type": "long", "expected_min": 46.0, "expected_max": 48.0},
        ]
        
        for case in test_cases:
            with self.subTest(**case):
                stop_price = calculate_stop_loss(
                    entry_price=case["entry"],
                    risk_percent=case["risk"],
                    position_type=case["type"]
                )
                self.assertIsNotNone(stop_price)
                if case["type"] == "long":
                    self.assertLess(stop_price, case["entry"])
                    self.assertGreaterEqual(stop_price, case["expected_min"])
                    self.assertLessEqual(stop_price, case["expected_max"])
                else:
                    self.assertGreater(stop_price, case["entry"])
                    self.assertGreaterEqual(stop_price, case["expected_min"])
                    self.assertLessEqual(stop_price, case["expected_max"])

        # Test trailing stop with different market conditions
        price_scenarios = [
            pd.Series([100, 105, 110, 108, 112]),  # Uptrend
            pd.Series([100, 95, 92, 94, 90]),      # Downtrend
            pd.Series([100, 102, 98, 103, 97]),    # Choppy
        ]
        
        for scenario in price_scenarios:
            with self.subTest(scenario_type=self._get_scenario_type(scenario)):
                trailing_stop = calculate_trailing_stop(
                    price_series=scenario,
                    initial_stop=scenario.iloc[0] * 0.95,
                    trail_percent=0.05
                )
                self.assertEqual(len(trailing_stop), len(scenario))
                self.assertTrue(trailing_stop.is_monotonic_increasing)
                self.assertTrue((trailing_stop <= scenario).all())

        # Test edge cases
        with self.assertRaises(ValueError):
            calculate_stop_loss(entry_price=-100.0, risk_percent=0.02)
        with self.assertRaises(ValueError):
            calculate_stop_loss(entry_price=100.0, risk_percent=-0.02)
        with self.assertRaises(ValueError):
            calculate_stop_loss(entry_price=100.0, risk_percent=1.5)

    def test_position_sizing(self):
        """Test position size calculations."""
        if not all([calculate_position_size, risk_parity_position_size]):
            self.skipTest("Position sizing functions not available")

        # Test fixed risk position sizing with different scenarios
        position_scenarios = [
            {
                "balance": 10000.0,
                "risk": 0.01,
                "entry": 100.0,
                "stop": 95.0,
                "expected_min": 1.0,
                "expected_max": 25.0
            },
            {
                "balance": 50000.0,
                "risk": 0.02,
                "entry": 500.0,
                "stop": 450.0,
                "expected_min": 10.0,
                "expected_max": 50.0
            }
        ]
        
        for scenario in position_scenarios:
            with self.subTest(**scenario):
                position = calculate_position_size(
                    account_balance=scenario["balance"],
                    risk_per_trade=scenario["risk"],
                    entry_price=scenario["entry"],
                    stop_loss_price=scenario["stop"]
                )
                self.assertIsNotNone(position)
                self.assertGreater(position, scenario["expected_min"])
                self.assertLess(position, scenario["expected_max"])
                
                # Verify maximum loss constraint
                max_loss = abs(scenario["entry"] - scenario["stop"]) * position
                self.assertLessEqual(max_loss, scenario["balance"] * scenario["risk"])

        # Test risk parity sizing
        vol_scenarios = [
            {"target": 0.15, "expected_max_weight": 0.4},
            {"target": 0.10, "expected_max_weight": 0.3},
            {"target": 0.20, "expected_max_weight": 0.5}
        ]
        
        for scenario in vol_scenarios:
            with self.subTest(**scenario):
                position_sizes = risk_parity_position_size(
                    returns=self.returns,
                    account_balance=10000.0,
                    volatility_target=scenario["target"]
                )
                self.assertIsInstance(position_sizes, dict)
                self.assertGreaterEqual(min(position_sizes.values()), 0)
                total_weight = sum(position_sizes.values()) / 10000.0
                self.assertLessEqual(total_weight, scenario["expected_max_weight"])

        # Test error handling
        with self.assertRaises(ValueError):
            calculate_position_size(
                account_balance=-1000.0,
                risk_per_trade=0.01,
                entry_price=100.0,
                stop_loss_price=95.0
            )

    def test_leverage_optimization(self):
        """Test leverage optimization methods."""
        # Test dynamic leverage model with different market conditions
        volatility_scenarios = [
            {"desc": "low_vol", "multiplier": 0.5},
            {"desc": "normal_vol", "multiplier": 1.0},
            {"desc": "high_vol", "multiplier": 2.0}
        ]
        
        for scenario in volatility_scenarios:
            with self.subTest(volatility=scenario["desc"]):
                test_df = self.df.copy()
                test_df[["high", "low", "close"]] *= scenario["multiplier"]
                
                leverage = dynamic_leverage_model(
                    price_data=test_df,
                    base_leverage=1.0,
                    atr_multiplier=0.1,
                    max_leverage=5.0
                )
                self.assertTrue((leverage <= 5.0).all())
                self.assertTrue((leverage >= 0.0).all())
                
                if scenario["desc"] == "high_vol":
                    self.assertTrue((leverage <= 2.0).all())
                elif scenario["desc"] == "low_vol":
                    self.assertTrue((leverage >= 0.5).all())

        # Test leverage limiting under different volatility conditions
        vol_test_cases = [
            {"current": 0.4, "base": 0.2, "max": 3.0, "expected_max": 1.5},
            {"current": 0.2, "base": 0.2, "max": 3.0, "expected_max": 3.0},
            {"current": 0.1, "base": 0.2, "max": 3.0, "expected_max": 3.0}
        ]
        
        for case in vol_test_cases:
            with self.subTest(**case):
                limited_lev = limit_max_leverage(
                    proposed_leverage=case["max"],
                    max_leverage=case["max"],
                    current_volatility=case["current"],
                    base_volatility=case["base"]
                )
                self.assertLessEqual(limited_lev, case["expected_max"])

        # Test VaR calculation with different confidence levels
        conf_levels = [0.90, 0.95, 0.99]
        for conf in conf_levels:
            with self.subTest(confidence=conf):
                var_result = value_at_risk(self.returns, confidence_level=conf)
                self.assertLess(var_result, 0)  # VaR should be negative
                self.assertTrue(-1 < var_result < 0)  # Sanity check on magnitude

        # Test error cases
        with self.assertRaises(ValueError):
            dynamic_leverage_model(
                price_data=self.df,
                base_leverage=-1.0,
                atr_multiplier=0.1,
                max_leverage=5.0
            )

    def _get_scenario_type(self, prices: pd.Series) -> str:
        """Helper to identify price scenario type."""
        returns = prices.pct_change().dropna()
        if (returns > 0).sum() / len(returns) > 0.7:
            return "uptrend"
        elif (returns < 0).sum() / len(returns) > 0.7:
            return "downtrend"
        return "choppy"

if __name__ == "__main__":
    unittest.main()
