"""
test_backtesting.py
------------------
Testy jednostkowe dla modułów backtestingu
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data.optimization.enhanced_backtesting import EnhancedBacktester
from data.optimization.strategy_backtest_runner import StrategyBacktestRunner

class TestEnhancedBacktester(unittest.TestCase):
    def setUp(self):
        """Przygotowanie danych testowych"""
        self.backtester = EnhancedBacktester(
            initial_capital=10000.0,
            commission=0.001,
            spread=0.0005,
            slippage=0.0005
        )
        
        # Generowanie przykładowych danych
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)
        volumes = np.random.randint(1000, 2000, 100)
        
        self.test_data = pd.DataFrame({
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Przykładowe transakcje
        self.test_trades = [
            {
                'entry_time': datetime(2023, 1, 1),
                'exit_time': datetime(2023, 1, 5),
                'entry_price': 100.0,
                'exit_price': 105.0,
                'position_size': 1.0,
                'profit': 5.0
            },
            {
                'entry_time': datetime(2023, 1, 10),
                'exit_time': datetime(2023, 1, 15),
                'entry_price': 105.0,
                'exit_price': 103.0,
                'position_size': 1.0,
                'profit': -2.0
            }
        ]
        
    def test_position_sizing(self):
        """Test metod określania wielkości pozycji"""
        capital = 10000.0
        price = 100.0
        volatility = 0.02
        
        # Test fixed sizing
        self.backtester.position_sizing_method = "fixed"
        fixed_size = self.backtester.calculate_position_size(capital, price, volatility)
        self.assertGreater(fixed_size, 0)
        
        # Test volatility sizing
        self.backtester.position_sizing_method = "volatility"
        vol_size = self.backtester.calculate_position_size(capital, price, volatility)
        self.assertGreater(vol_size, 0)
        self.assertLess(vol_size, fixed_size)  # Przy większej zmienności pozycja powinna być mniejsza
        
    def test_monte_carlo_simulation(self):
        """Test symulacji Monte Carlo"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        mc_results = self.backtester.run_monte_carlo_simulation(returns, n_simulations=100)
        
        self.assertIn("confidence_intervals", mc_results)
        self.assertIn("mean_terminal_value", mc_results)
        self.assertGreater(mc_results["best_case"], mc_results["worst_case"])
        
    def test_walk_forward_analysis(self):
        """Test analizy walk-forward"""
        def dummy_strategy(data):
            return {
                'equity_curve': [10000 + i*100 for i in range(len(data)+1)],
                'trades': self.test_trades
            }
            
        results = self.backtester.run_walk_forward_analysis(
            self.test_data,
            dummy_strategy,
            train_ratio=0.7,
            n_splits=3
        )
        
        self.assertIn('split_results', results)
        self.assertIn('average_sharpe', results)
        self.assertIn('average_max_dd', results)
        self.assertIn('robustness_score', results)
        
    def test_performance_metrics(self):
        """Test obliczania metryk wydajności"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        # Test Sharpe ratio
        sharpe = self.backtester.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)
        
        # Test Sortino ratio
        sortino = self.backtester.calculate_sortino_ratio(returns)
        self.assertIsInstance(sortino, float)
        
        # Test max drawdown
        equity = [100 * (1 + r) for r in returns]
        max_dd = self.backtester.calculate_max_drawdown(equity)
        self.assertGreaterEqual(max_dd, 0)
        self.assertLessEqual(max_dd, 1)
        
class TestStrategyBacktestRunner(unittest.TestCase):
    def setUp(self):
        """Przygotowanie danych testowych"""
        self.runner = StrategyBacktestRunner(
            initial_capital=10000.0,
            position_sizing_method="volatility"
        )
        
        # Generowanie przykładowych danych
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 1, 100),
            'high': np.random.normal(101, 1, 100),
            'low': np.random.normal(99, 1, 100),
            'close': np.random.normal(100, 1, 100),
            'volume': np.random.randint(1000, 2000, 100)
        }, index=dates)
        
    def test_strategy_execution(self):
        """Test wykonania pojedynczej strategii"""
        # Test strategii trend following
        tf_results = self.runner._wrap_trend_following_strategy(
            self.test_data,
            adx_threshold=20,
            macd_threshold=0
        )
        
        self.assertIn('equity_curve', tf_results)
        self.assertIn('trades', tf_results)
        
        # Test strategii mean reversion
        mr_results = self.runner._wrap_mean_reversion_strategy(
            self.test_data,
            window=20,
            zscore_threshold=2.0
        )
        
        self.assertIn('equity_curve', mr_results)
        self.assertIn('trades', mr_results)
        
    def test_strategy_comparison(self):
        """Test porównania strategii"""
        strategies = ['trend_following', 'mean_reversion', 'breakout']
        params = {
            'trend_following': {'adx_threshold': 20},
            'mean_reversion': {'window': 20},
            'breakout': {'window': 20}
        }
        
        results = self.runner.run_strategy_comparison(
            self.test_data,
            strategies,
            params
        )
        
        for strategy in strategies:
            self.assertIn(strategy, results)
            strategy_results = results[strategy]
            self.assertIn('performance', strategy_results)
            self.assertIn('walk_forward', strategy_results)
            self.assertIn('equity_curve', strategy_results)
            
    def test_plotting(self):
        """Test generowania wykresów"""
        # Przygotuj przykładowe wyniki
        mock_results = {
            'strategy1': {
                'equity_curve': [10000 + i*100 for i in range(100)],
                'trades': [{'profit': 100} for _ in range(10)],
                'performance': {
                    'overall_metrics': {
                        'total_return': 0.1,
                        'sharpe_ratio': 1.5,
                        'max_drawdown': 0.05,
                        'win_rate': 0.6
                    }
                }
            }
        }
        
        # Sprawdź, czy funkcja nie zgłasza błędów
        try:
            self.runner.plot_strategy_comparison(mock_results, save_path=None)
        except Exception as e:
            self.fail(f"Generowanie wykresów nie powinno zgłaszać błędów: {str(e)}")

if __name__ == '__main__':
    unittest.main()