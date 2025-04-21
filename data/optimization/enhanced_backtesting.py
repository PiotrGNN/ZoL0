"""
enhanced_backtesting.py
----------------------
Rozszerzony moduł do backtestingu strategii tradingowych z zaawansowanymi funkcjami analizy.
"""

import logging
from typing import Dict, List, Any, Tuple, Callable, Optional
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedBacktester:
    def __init__(
        self, 
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        spread: float = 0.0005,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
        position_sizing_method: str = "fixed"
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.spread = spread
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.position_sizing_method = position_sizing_method
        
        self.trades_history = []
        self.equity_curve = []
        self.position = 0
        self.entry_price = 0
        
    def calculate_position_size(self, capital: float, price: float, volatility: float) -> float:
        """
        Oblicza wielkość pozycji w zależności od wybranej metody
        """
        if self.position_sizing_method == "fixed":
            return 0.1 * capital / price
        elif self.position_sizing_method == "volatility":
            return (0.1 * capital) / (price * volatility)
        elif self.position_sizing_method == "kelly":
            # Uproszczona implementacja kryterium Kelly'ego
            win_rate = 0.5  # Można dostosować na podstawie historycznych wyników
            return (win_rate * 2 - 1) * capital / price
        return 0.1 * capital / price

    def run_monte_carlo_simulation(
        self, 
        returns: pd.Series, 
        n_simulations: int = 1000,
        n_days: int = 252
    ) -> Dict[str, Any]:
        """
        Przeprowadza symulację Monte Carlo dla wyników strategii
        """
        mean_return = returns.mean()
        std_return = returns.std()
        
        simulations = np.zeros((n_simulations, n_days))
        for i in range(n_simulations):
            simulations[i] = np.random.normal(mean_return, std_return, n_days)
        
        cumulative_returns = np.cumprod(1 + simulations, axis=1)
        final_values = cumulative_returns[:, -1]
        
        confidence_intervals = {
            "95%": np.percentile(final_values, [2.5, 97.5]),
            "99%": np.percentile(final_values, [0.5, 99.5])
        }
        
        return {
            "confidence_intervals": confidence_intervals,
            "mean_terminal_value": final_values.mean(),
            "worst_case": final_values.min(),
            "best_case": final_values.max()
        }

    def run_walk_forward_analysis(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        train_ratio: float = 0.7,
        n_splits: int = 5,
        optimize_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Przeprowadza analizę walk-forward z możliwością optymalizacji parametrów
        """
        results = []
        data_splits = []
        
        # Tworzenie podziałów danych
        split_size = len(df) // n_splits
        for i in range(n_splits):
            start_idx = i * split_size
            train_end_idx = start_idx + int(split_size * train_ratio)
            test_end_idx = start_idx + split_size
            
            train_data = df.iloc[start_idx:train_end_idx]
            test_data = df.iloc[train_end_idx:test_end_idx]
            data_splits.append((train_data, test_data))
        
        for train_data, test_data in data_splits:
            # Optymalizacja parametrów na danych treningowych
            if optimize_func:
                optimal_params = optimize_func(train_data)
                strategy_results = strategy_func(test_data, **optimal_params)
            else:
                strategy_results = strategy_func(test_data)
            
            # Obliczenie metryk dla tego splitu
            returns = pd.Series(strategy_results['equity_curve']).pct_change()
            sharpe = self.calculate_sharpe_ratio(returns)
            max_dd = self.calculate_max_drawdown(strategy_results['equity_curve'])
            
            results.append({
                'returns': returns,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'profit_factor': self.calculate_profit_factor(strategy_results['trades'])
            })
        
        return {
            'split_results': results,
            'average_sharpe': np.mean([r['sharpe_ratio'] for r in results]),
            'average_max_dd': np.mean([r['max_drawdown'] for r in results]),
            'robustness_score': self.calculate_robustness_score(results)
        }

    def calculate_robustness_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Oblicza wskaźnik odporności strategii na podstawie wyników z różnych okresów
        """
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        return stats.variation(sharpe_ratios)  # Współczynnik zmienności

    def calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """
        Oblicza współczynnik zyskowności (suma zysków / suma strat)
        """
        profits = sum(t['profit'] for t in trades if t['profit'] > 0)
        losses = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
        return profits / losses if losses != 0 else float('inf')

    def generate_performance_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generuje szczegółowy raport wydajności strategii
        """
        returns = pd.Series(results['equity_curve']).pct_change()
        
        report = {
            'overall_metrics': {
                'total_return': (results['equity_curve'][-1] / results['equity_curve'][0]) - 1,
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.calculate_sortino_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown(results['equity_curve']),
                'profit_factor': self.calculate_profit_factor(results['trades']),
                'win_rate': sum(1 for t in results['trades'] if t['profit'] > 0) / len(results['trades'])
            },
            'risk_metrics': {
                'var_95': self.calculate_var(returns, 0.95),
                'cvar_95': self.calculate_cvar(returns, 0.95),
                'volatility': returns.std() * np.sqrt(252),
                'downside_deviation': self.calculate_downside_deviation(returns)
            },
            'trade_analysis': self.analyze_trades(results['trades'])
        }
        
        # Dodanie wyników symulacji Monte Carlo
        monte_carlo = self.run_monte_carlo_simulation(returns)
        report['monte_carlo_simulation'] = monte_carlo
        
        return report

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Oblicza współczynnik Sharpe'a
        """
        excess_returns = returns - self.risk_free_rate/252
        if len(excess_returns) < 2:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Oblicza współczynnik Sortino
        """
        excess_returns = returns - self.risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Oblicza maksymalny drawdown
        """
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Oblicza Value at Risk
        """
        return abs(np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Oblicza Conditional Value at Risk
        """
        var = self.calculate_var(returns, confidence_level)
        return abs(returns[returns <= -var].mean())

    def calculate_downside_deviation(self, returns: pd.Series) -> float:
        """
        Oblicza odchylenie standardowe ujemnych zwrotów
        """
        negative_returns = returns[returns < 0]
        return negative_returns.std() * np.sqrt(252)

    def analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Przeprowadza szczegółową analizę transakcji
        """
        if not trades:
            return {}
            
        profits = [t['profit'] for t in trades]
        durations = [(t['exit_time'] - t['entry_time']).total_seconds()/86400 for t in trades]
        
        return {
            'avg_profit': np.mean(profits),
            'median_profit': np.median(profits),
            'profit_std': np.std(profits),
            'avg_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'best_trade': max(profits),
            'worst_trade': min(profits),
            'profit_distribution': {
                'skewness': stats.skew(profits),
                'kurtosis': stats.kurtosis(profits)
            }
        }

    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Generuje wykresy wyników backtestingu
        """
        plt.figure(figsize=(15, 10))
        
        # Wykres equity curve
        plt.subplot(2, 2, 1)
        plt.plot(results['equity_curve'])
        plt.title('Equity Curve')
        plt.grid(True)
        
        # Histogram zwrotów
        plt.subplot(2, 2, 2)
        returns = pd.Series(results['equity_curve']).pct_change()
        plt.hist(returns, bins=50)
        plt.title('Returns Distribution')
        plt.grid(True)
        
        # Wykres drawdownów
        plt.subplot(2, 2, 3)
        equity_series = pd.Series(results['equity_curve'])
        drawdowns = equity_series/equity_series.cummax() - 1
        plt.plot(drawdowns)
        plt.title('Drawdowns')
        plt.grid(True)
        
        # Wykres skumulowanych zysków/strat
        plt.subplot(2, 2, 4)
        profits = [t['profit'] for t in results['trades']]
        plt.plot(np.cumsum(profits))
        plt.title('Cumulative P&L')
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

# Przykład użycia:
if __name__ == "__main__":
    backtester = EnhancedBacktester(
        initial_capital=10000.0,
        commission=0.001,
        spread=0.0005,
        slippage=0.0005
    )