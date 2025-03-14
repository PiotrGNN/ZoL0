"""
test_risk_management.py
-----------------------
Testy jednostkowe dla modułów:
- stop_loss_manager.py,
- portfolio_risk.py,
- position_sizing.py,
- leverage_optimizer.py.

Testy mają na celu weryfikację poprawności obliczeń ryzyka, poziomów stop-loss, dźwigni finansowej oraz rozmiarów pozycji przy różnych warunkach rynkowych, w tym w warunkach skrajnych.
"""

import os
import unittest
import logging
import numpy as np
import pandas as pd

# Konfiguracja logowania do testów
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Importujemy moduły testowane
from stop_loss_manager import fixed_stop_loss, trailing_stop_loss, atr_based_stop_loss, time_based_stop_loss, simulate_stop_loss_strategy
from portfolio_risk import calculate_var, calculate_cvar, monte_carlo_var, herfindahl_index, recommend_rebalancing, assess_portfolio_risk
from position_sizing import fixed_fractional_position_size, kelly_criterion_position_size, risk_parity_position_size, dynamic_position_size
from leverage_optimizer import dynamic_leverage_model, limit_max_leverage

class TestRiskManagementModules(unittest.TestCase):
    def setUp(self):
        # Przygotowanie przykładowych danych rynkowych
        np.random.seed(42)
        self.dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.df_prices = pd.DataFrame({
            "high": np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            "low": np.linspace(90, 140, 100) + np.random.normal(0, 1, 100),
            "close": np.linspace(95, 145, 100) + np.random.normal(0, 1, 100)
        }, index=self.dates)
        self.returns = self.df_prices['close'].pct_change().dropna()

    def test_stop_loss_functions(self):
        # Test fixed stop loss
        entry_price = 100
        fixed_distance = 10
        sl_fixed = fixed_stop_loss(entry_price, fixed_distance)
        self.assertAlmostEqual(sl_fixed, 90, msg="Fixed stop loss calculation error.")
        
        # Test trailing stop loss
        highest_price = 110
        trailing_pct = 0.05
        sl_trailing = trailing_stop_loss(100, highest_price, trailing_pct)
        expected_sl = highest_price * (1 - trailing_pct)
        self.assertAlmostEqual(sl_trailing, expected_sl, msg="Trailing stop loss calculation error.")
        
        # Test ATR-based stop loss adjustment
        atr_adj = atr_based_stop_loss(self.df_prices, period=14, atr_multiplier=1.5)
        self.assertGreater(atr_adj, 0, "ATR-based stop loss adjustment should be positive.")
        
        # Test time-based stop loss condition
        entry_time = pd.Timestamp("2023-01-01 09:00:00")
        current_time = pd.Timestamp("2023-01-01 10:00:00")
        time_stop = time_based_stop_loss(entry_time, current_time, max_duration_minutes=60)
        self.assertTrue(time_stop, "Time-based stop loss should trigger after exceeding max duration.")

    def test_portfolio_risk_metrics(self):
        var_value = calculate_var(self.returns, confidence_level=0.95)
        self.assertLess(var_value, 0, "VaR should be a negative value indicating loss.")
        
        cvar_value = calculate_cvar(self.returns, confidence_level=0.95)
        self.assertLessEqual(cvar_value, var_value, "CVaR should be lower (more negative) than VaR.")
        
        var_mc = monte_carlo_var(self.returns, num_simulations=1000, horizon=1, confidence_level=0.95)
        self.assertLess(var_mc, 0, "Monte Carlo VaR should be negative.")
        
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        hh_index = herfindahl_index(weights)
        self.assertGreaterEqual(hh_index, 0)
        self.assertLessEqual(hh_index, 1)
        
        recommendation = recommend_rebalancing(hh_index, threshold=0.3)
        self.assertIn(recommendation, ["Rebalance", "Hold"])
        
        risk_df = assess_portfolio_risk(self.returns, rolling_window=30, confidence_level=0.95)
        self.assertIn("VaR", risk_df.columns)
        self.assertIn("CVaR", risk_df.columns)

    def test_position_sizing_functions(self):
        capital = 10000
        risk_per_trade = 0.02
        stop_loss_distance = 50
        pos_size_fixed = fixed_fractional_position_size(capital, risk_per_trade, stop_loss_distance)
        self.assertGreater(pos_size_fixed, 0)
        
        win_rate = 0.55
        win_loss_ratio = 2.0
        pos_size_kelly = kelly_criterion_position_size(win_rate, win_loss_ratio, capital)
        self.assertGreaterEqual(pos_size_kelly, 0)
        
        volatilities = np.array([0.1, 0.2, 0.15, 0.25])
        pos_sizes_risk_parity = risk_parity_position_size(volatilities, capital)
        self.assertAlmostEqual(np.sum(pos_sizes_risk_parity), capital)
        
        market_volatility = 0.05
        pos_size_dynamic = dynamic_position_size(capital, risk_per_trade, stop_loss_distance, market_volatility)
        self.assertGreater(pos_size_dynamic, 0)

    def test_leverage_optimizer_functions(self):
        dyn_leverage = dynamic_leverage_model(self.df_prices, base_leverage=1.0, atr_multiplier=0.1, max_leverage=5.0)
        self.assertTrue((dyn_leverage <= 5.0).all())
        
        limited_leverage = limit_max_leverage(current_leverage=4.5, risk_factor=0.06, max_allowed_leverage=3.0)
        self.assertLessEqual(limited_leverage, 3.0)

if __name__ == "__main__":
    unittest.main()
