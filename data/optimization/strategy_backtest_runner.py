"""
strategy_backtest_runner.py
--------------------------
Moduł do uruchamiania i porównywania backtestów różnych strategii.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from data.optimization.enhanced_backtesting import EnhancedBacktester
from data.strategies.trend_following import generate_trend_following_signal
from data.strategies.mean_reversion import generate_mean_reversion_signal
from data.strategies.breakout_strategy import breakout_strategy

class StrategyBacktestRunner:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.02,
        position_sizing_method: str = "volatility"
    ):
        self.backtester = EnhancedBacktester(
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            position_sizing_method=position_sizing_method
        )
        self.strategies = {
            'trend_following': self._wrap_trend_following_strategy,
            'mean_reversion': self._wrap_mean_reversion_strategy,
            'breakout': self._wrap_breakout_strategy
        }
        
    def _wrap_trend_following_strategy(
        self,
        data: pd.DataFrame,
        adx_threshold: float = 25,
        macd_threshold: float = 0,
        channel_window: int = 20,
        liquidity_threshold: float = 1000
    ) -> Dict[str, Any]:
        """
        Wrapper dla strategii trend following
        """
        signals = generate_trend_following_signal(
            data,
            adx_threshold=adx_threshold,
            macd_threshold=macd_threshold,
            channel_window=channel_window,
            liquidity_threshold=liquidity_threshold
        )
        return self._execute_signals(data, signals)
        
    def _wrap_mean_reversion_strategy(
        self,
        data: pd.DataFrame,
        window: int = 20,
        zscore_threshold: float = 2.0,
        volume_filter: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wrapper dla strategii mean reversion
        """
        signals = generate_mean_reversion_signal(
            data,
            window=window,
            zscore_threshold=zscore_threshold,
            volume_filter=volume_filter
        )
        return self._execute_signals(data, signals)
        
    def _wrap_breakout_strategy(
        self,
        data: pd.DataFrame,
        window: int = 20,
        volume_threshold: float = 1.5,
        margin_pct: float = 0.02
    ) -> Dict[str, Any]:
        """
        Wrapper dla strategii breakout
        """
        result = breakout_strategy(
            data,
            window=window,
            volume_threshold=volume_threshold,
            margin_pct=margin_pct
        )
        return self._execute_signals(data, result['signal'])
        
    def _execute_signals(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        """
        Wykonuje symulację transakcji na podstawie sygnałów
        """
        equity_curve = [self.backtester.initial_capital]
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(len(data)):
            price = data.iloc[i]['close']
            current_capital = equity_curve[-1]
            
            # Oblicz zmienność dla position sizing
            volatility = data['close'].rolling(window=20).std().iloc[i] / price
            
            if signals[i] == 1 and position == 0:  # Sygnał kupna
                position_size = self.backtester.calculate_position_size(
                    current_capital, price, volatility
                )
                position = 1
                entry_price = price * (1 + self.backtester.spread + self.backtester.slippage)
                entry_time = data.index[i]
                
            elif signals[i] == -1 and position == 1:  # Sygnał sprzedaży
                exit_price = price * (1 - self.backtester.spread - self.backtester.slippage)
                profit = (exit_price - entry_price) * position_size
                profit -= exit_price * position_size * self.backtester.commission
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': data.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'profit': profit
                })
                
                current_capital += profit
                position = 0
                
            equity_curve.append(current_capital)
            
        return {
            'equity_curve': equity_curve,
            'trades': trades
        }
        
    def run_strategy_comparison(
        self,
        data: pd.DataFrame,
        strategies_to_test: List[str],
        params_dict: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Porównuje wyniki różnych strategii na tych samych danych
        """
        if not params_dict:
            params_dict = {strategy: {} for strategy in strategies_to_test}
            
        results = {}
        for strategy_name in strategies_to_test:
            if strategy_name not in self.strategies:
                logging.warning(f"Strategia {strategy_name} nie jest zaimplementowana")
                continue
                
            strategy_func = self.strategies[strategy_name]
            strategy_params = params_dict.get(strategy_name, {})
            
            # Wykonaj backtest
            backtest_results = strategy_func(data, **strategy_params)
            
            # Generuj raport wydajności
            performance_report = self.backtester.generate_performance_report(backtest_results)
            
            # Dodaj wyniki walk-forward analysis
            walk_forward_results = self.backtester.run_walk_forward_analysis(
                data,
                strategy_func,
                optimize_func=None  # Można dodać funkcję optymalizacji parametrów
            )
            
            results[strategy_name] = {
                'performance': performance_report,
                'walk_forward': walk_forward_results,
                'equity_curve': backtest_results['equity_curve'],
                'trades': backtest_results['trades']
            }
            
        return results
        
    def plot_strategy_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Generuje wykresy porównawcze dla wszystkich strategii
        """
        plt.figure(figsize=(15, 10))
        
        # Wykresy equity curves
        plt.subplot(2, 2, 1)
        for strategy_name, strategy_results in results.items():
            plt.plot(
                strategy_results['equity_curve'],
                label=strategy_name
            )
        plt.title('Porównanie Equity Curves')
        plt.legend()
        plt.grid(True)
        
        # Wykresy rozkładów zwrotów
        plt.subplot(2, 2, 2)
        for strategy_name, strategy_results in results.items():
            returns = pd.Series(strategy_results['equity_curve']).pct_change()
            plt.hist(
                returns,
                bins=30,
                alpha=0.5,
                label=strategy_name
            )
        plt.title('Rozkłady zwrotów')
        plt.legend()
        plt.grid(True)
        
        # Tabela metryk
        plt.subplot(2, 2, 3)
        metrics_data = []
        for strategy_name, strategy_results in results.items():
            perf = strategy_results['performance']['overall_metrics']
            metrics_data.append([
                strategy_name,
                f"{perf['total_return']:.2%}",
                f"{perf['sharpe_ratio']:.2f}",
                f"{perf['max_drawdown']:.2%}",
                f"{perf['win_rate']:.2%}"
            ])
        
        plt.table(
            cellText=metrics_data,
            colLabels=['Strategia', 'Total Return', 'Sharpe', 'Max DD', 'Win Rate'],
            loc='center'
        )
        plt.axis('off')
        
        # Wykres skumulowanych zysków
        plt.subplot(2, 2, 4)
        for strategy_name, strategy_results in results.items():
            profits = [t['profit'] for t in strategy_results['trades']]
            plt.plot(
                np.cumsum(profits),
                label=strategy_name
            )
        plt.title('Skumulowane zyski')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

# Przykład użycia:
if __name__ == "__main__":
    runner = StrategyBacktestRunner(
        initial_capital=10000.0,
        position_sizing_method="volatility"
    )