"""
advanced_backtester.py
--------------------
Moduł do zaawansowanego testowania strategii, włączając optymalizację parametrów
i analizę wrażliwości.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import json
import itertools
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.risk_management.risk_assessment import (
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio
)

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Klasa przechowująca wyniki backtestu."""
    strategy_name: str
    parameters: Dict[str, Union[float, int, str]]
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    drawdown_curve: pd.Series
    position_history: pd.DataFrame
    execution_time: float

class AdvancedBacktester:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        commission: float = 0.001,
        risk_free_rate: float = 0.0
    ):
        """
        Inicjalizacja backtestera.
        
        Args:
            data: DataFrame z danymi historycznymi (OHLCV)
            initial_capital: Początkowy kapitał
            commission: Prowizja (jako ułamek)
            risk_free_rate: Stopa wolna od ryzyka (roczna)
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1

    def run_backtest(
        self,
        strategy: Callable,
        parameters: Dict[str, Union[float, int, str]],
        strategy_name: str = "Custom Strategy"
    ) -> BacktestResult:
        """
        Przeprowadza pojedynczy backtest strategii.
        
        Args:
            strategy: Funkcja strategii
            parameters: Parametry strategii
            strategy_name: Nazwa strategii
            
        Returns:
            BacktestResult z wynikami backtestu
        """
        start_time = datetime.now()
        
        # Inicjalizacja zmiennych
        equity = pd.Series(index=self.data.index, dtype=float)
        equity.iloc[0] = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        positions = []
        
        # Główna pętla backtesting
        for i in range(len(self.data)):
            # Przygotuj dane dostępne do tego momentu
            current_data = self.data.iloc[:i+1]
            
            # Pobierz sygnał ze strategii
            signal = strategy(current_data, parameters)
            
            # Aktualizuj pozycję
            if signal != 0 and signal != position:
                # Zamknij istniejącą pozycję
                if position != 0:
                    exit_price = self.data.iloc[i]['close']
                    pnl = position * (exit_price - entry_price)
                    commission = abs(position * exit_price * self.commission)
                    equity.iloc[i] = equity.iloc[i-1] + pnl - commission
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': self.data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'commission': commission
                    })
                
                # Otwórz nową pozycję
                position = signal
                entry_price = self.data.iloc[i]['close']
                entry_time = self.data.index[i]
            
            # Aktualizuj equity dla dni bez transakcji
            if i > 0 and equity.iloc[i] == 0:
                if position != 0:
                    # Aktualizuj niezrealizowany P/L
                    current_price = self.data.iloc[i]['close']
                    unrealized_pnl = position * (current_price - entry_price)
                    equity.iloc[i] = equity.iloc[i-1] + unrealized_pnl
                else:
                    equity.iloc[i] = equity.iloc[i-1]
            
            # Zapisz historię pozycji
            positions.append({
                'timestamp': self.data.index[i],
                'position': position,
                'equity': equity.iloc[i]
            })
        
        # Konwertuj listy na DataFrame'y
        trades_df = pd.DataFrame(trades)
        positions_df = pd.DataFrame(positions)
        
        # Oblicz drawdown
        roll_max = equity.expanding().max()
        drawdown = equity / roll_max - 1
        
        # Oblicz metryki
        returns = equity.pct_change().dropna()
        metrics = {
            'total_return': (equity.iloc[-1] / self.initial_capital - 1),
            'annualized_return': ((equity.iloc[-1] / self.initial_capital) ** (252/len(equity)) - 1),
            'sharpe_ratio': calculate_sharpe_ratio(returns, self.daily_risk_free),
            'sortino_ratio': calculate_sortino_ratio(returns, self.daily_risk_free),
            'max_drawdown': calculate_max_drawdown(equity)[0],
            'calmar_ratio': calculate_calmar_ratio(returns, equity),
            'var_95': calculate_var(returns),
            'cvar_95': calculate_cvar(returns),
            'total_trades': len(trades),
            'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades) if len(trades) > 0 else 0,
            'profit_factor': (
                abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum()) /
                abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                if len(trades_df[trades_df['pnl'] < 0]) > 0
                else float('inf')
            ),
            'avg_trade': trades_df['pnl'].mean() if len(trades) > 0 else 0,
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
            'max_consecutive_wins': self._calculate_consecutive_stats(trades_df, 'wins'),
            'max_consecutive_losses': self._calculate_consecutive_stats(trades_df, 'losses')
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return BacktestResult(
            strategy_name=strategy_name,
            parameters=parameters,
            equity_curve=equity,
            trades=trades_df,
            metrics=metrics,
            drawdown_curve=drawdown,
            position_history=positions_df,
            execution_time=execution_time
        )

    def optimize_parameters(
        self,
        strategy: Callable,
        param_grid: Dict[str, List[Union[float, int, str]]],
        strategy_name: str = "Custom Strategy",
        optimization_metric: str = 'sharpe_ratio',
        max_workers: int = 4
    ) -> List[BacktestResult]:
        """
        Optymalizuje parametry strategii używając grid search.
        
        Args:
            strategy: Funkcja strategii
            param_grid: Siatka parametrów do przetestowania
            strategy_name: Nazwa strategii
            optimization_metric: Metryka do optymalizacji
            max_workers: Maksymalna liczba równoległych procesów
            
        Returns:
            Lista wyników backtestów posortowana według metryki optymalizacji
        """
        # Generuj wszystkie kombinacje parametrów
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        results = []
        
        # Równoległe wykonanie backtestów
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                futures.append(
                    executor.submit(
                        self.run_backtest,
                        strategy=strategy,
                        parameters=param_dict,
                        strategy_name=f"{strategy_name}_{len(futures)}"
                    )
                )
            
            # Zbierz wyniki
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Błąd podczas optymalizacji parametrów: {e}")
        
        # Sortuj wyniki według metryki optymalizacji
        results.sort(key=lambda x: x.metrics[optimization_metric], reverse=True)
        
        return results

    def analyze_sensitivity(
        self,
        strategy: Callable,
        base_parameters: Dict[str, Union[float, int, str]],
        param_ranges: Dict[str, List[Union[float, int, str]]],
        strategy_name: str = "Custom Strategy"
    ) -> Dict[str, pd.DataFrame]:
        """
        Przeprowadza analizę wrażliwości parametrów strategii.
        
        Args:
            strategy: Funkcja strategii
            base_parameters: Bazowe parametry strategii
            param_ranges: Zakresy parametrów do przetestowania
            strategy_name: Nazwa strategii
            
        Returns:
            Dict z wynikami analizy wrażliwości dla każdego parametru
        """
        sensitivity_results = {}
        
        for param_name, param_range in param_ranges.items():
            param_results = []
            
            for value in param_range:
                # Skopiuj bazowe parametry i zmień jeden
                test_params = base_parameters.copy()
                test_params[param_name] = value
                
                # Wykonaj backtest
                result = self.run_backtest(
                    strategy=strategy,
                    parameters=test_params,
                    strategy_name=f"{strategy_name}_{param_name}_{value}"
                )
                
                # Zapisz wyniki
                param_results.append({
                    'param_value': value,
                    'sharpe_ratio': result.metrics['sharpe_ratio'],
                    'total_return': result.metrics['total_return'],
                    'max_drawdown': result.metrics['max_drawdown'],
                    'win_rate': result.metrics['win_rate']
                })
            
            sensitivity_results[param_name] = pd.DataFrame(param_results)
        
        return sensitivity_results

    def compare_strategies(
        self,
        strategies: List[Dict[str, Union[Callable, Dict, str]]],
        plot: bool = True
    ) -> Dict[str, BacktestResult]:
        """
        Porównuje wiele strategii.
        
        Args:
            strategies: Lista słowników z funkcjami strategii i ich parametrami
            plot: Czy generować wykresy
            
        Returns:
            Dict z wynikami dla każdej strategii
        """
        results = {}
        
        for strategy_dict in strategies:
            result = self.run_backtest(
                strategy=strategy_dict['function'],
                parameters=strategy_dict['parameters'],
                strategy_name=strategy_dict['name']
            )
            results[strategy_dict['name']] = result
        
        if plot:
            self._plot_comparison(results)
        
        return results

    def generate_report(
        self,
        result: BacktestResult,
        output_file: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generuje szczegółowy raport z backtestu.
        
        Args:
            result: Wynik backtestu
            output_file: Ścieżka do pliku wyjściowego (opcjonalne)
            
        Returns:
            Dict z raportem
        """
        report = {
            'strategy_name': result.strategy_name,
            'parameters': result.parameters,
            'metrics': result.metrics,
            'execution_time': result.execution_time,
            'trade_statistics': {
                'total_trades': len(result.trades),
                'profitable_trades': len(result.trades[result.trades['pnl'] > 0]),
                'losing_trades': len(result.trades[result.trades['pnl'] < 0]),
                'win_rate': result.metrics['win_rate'],
                'profit_factor': result.metrics['profit_factor'],
                'average_trade': result.metrics['avg_trade'],
                'average_win': result.metrics['avg_win'],
                'average_loss': result.metrics['avg_loss'],
                'largest_win': result.trades['pnl'].max() if len(result.trades) > 0 else 0,
                'largest_loss': result.trades['pnl'].min() if len(result.trades) > 0 else 0,
                'max_consecutive_wins': result.metrics['max_consecutive_wins'],
                'max_consecutive_losses': result.metrics['max_consecutive_losses']
            },
            'risk_metrics': {
                'sharpe_ratio': result.metrics['sharpe_ratio'],
                'sortino_ratio': result.metrics['sortino_ratio'],
                'calmar_ratio': result.metrics['calmar_ratio'],
                'max_drawdown': result.metrics['max_drawdown'],
                'var_95': result.metrics['var_95'],
                'cvar_95': result.metrics['cvar_95']
            },
            'returns': {
                'total_return': result.metrics['total_return'],
                'annualized_return': result.metrics['annualized_return'],
                'daily_returns_mean': result.equity_curve.pct_change().mean(),
                'daily_returns_std': result.equity_curve.pct_change().std(),
                'monthly_returns': self._calculate_period_returns(result.equity_curve, 'M'),
                'yearly_returns': self._calculate_period_returns(result.equity_curve, 'Y')
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report

    def _calculate_consecutive_stats(self, trades_df: pd.DataFrame, stat_type: str) -> int:
        """Oblicza maksymalną liczbę kolejnych wygranych lub przegranych."""
        if len(trades_df) == 0:
            return 0
            
        consecutive = 0
        max_consecutive = 0
        
        for pnl in trades_df['pnl']:
            if (stat_type == 'wins' and pnl > 0) or (stat_type == 'losses' and pnl < 0):
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
                
        return max_consecutive

    def _calculate_period_returns(self, equity: pd.Series, period: str) -> Dict[str, float]:
        """Oblicza zwroty dla danego okresu (M - miesięczne, Y - roczne)."""
        if period == 'M':
            grouped = equity.groupby(pd.Grouper(freq='M'))
        else:
            grouped = equity.groupby(pd.Grouper(freq='Y'))
            
        period_returns = {}
        for name, group in grouped:
            if len(group) > 0:
                period_return = (group.iloc[-1] / group.iloc[0] - 1)
                period_returns[str(name.date())] = period_return
                
        return period_returns

    def _plot_comparison(self, results: Dict[str, BacktestResult]) -> None:
        """Generuje wykresy porównawcze dla wielu strategii."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Krzywe kapitału', 'Drawdown', 'Skumulowany P/L'),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, (name, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            
            # Krzywa kapitału
            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    name=f"{name} Equity",
                    line=dict(color=color)
                ),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=result.drawdown_curve.index,
                    y=result.drawdown_curve.values * 100,
                    name=f"{name} Drawdown",
                    line=dict(color=color)
                ),
                row=2, col=1
            )
            
            # Skumulowany P/L
            cumulative_pnl = result.trades['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=result.trades.index,
                    y=cumulative_pnl.values,
                    name=f"{name} Cum. P/L",
                    line=dict(color=color)
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=1200,
            title_text="Porównanie Strategii",
            showlegend=True,
            template='plotly_dark'
        )
        
        fig.update_yaxes(title_text="Kapitał ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Skumulowany P/L ($)", row=3, col=1)
        
        fig.show()