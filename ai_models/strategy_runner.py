"""
StrategyRunner i StrategyBacktestRunner - moduły do zarządzania i testowania strategii
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from .enhanced_backtester import EnhancedBacktester

class StrategyRunner:
    def __init__(self):
        self.strategies: Dict[str, Callable] = {}
        self.backtester = EnhancedBacktester()
        
    def register_strategy(self, name: str, strategy_fn: Callable) -> None:
        """
        Rejestruje nową strategię w systemie
        """
        self.strategies[name] = strategy_fn
        
    def run_strategy(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Uruchamia wybraną strategię i przeprowadza backtesting
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategia {strategy_name} nie istnieje")
            
        strategy = self.strategies[strategy_name]
        signals = strategy(data, **(params or {}))
        
        # Przeprowadzenie analizy walk-forward
        wf_results = self.backtester.run_walk_forward_analysis(
            data=data,
            strategy_fn=lambda x: strategy(x, **(params or {}))
        )
        
        # Przeprowadzenie symulacji Monte Carlo
        returns = self.backtester._calculate_returns(data, signals)
        mc_results = self.backtester.run_monte_carlo_simulation(returns)
        
        return {
            'strategy_name': strategy_name,
            'signals': signals,
            'walk_forward_results': wf_results,
            'monte_carlo_results': mc_results,
            'performance_metrics': self._calculate_performance_metrics(returns)
        }
        
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """
        Oblicza podstawowe metryki wydajności strategii
        """
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe = self.backtester._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': returns.std() * np.sqrt(252)
        }
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Oblicza maksymalny drawdown dla serii zwrotów
        """
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return drawdowns.min()
        
    def compare_strategies(
        self,
        data: pd.DataFrame,
        strategy_params: Dict[str, Dict] = None
    ) -> Dict:
        """
        Porównuje wyniki wszystkich zarejestrowanych strategii
        """
        results = {}
        strategy_params = strategy_params or {}
        
        for name in self.strategies:
            params = strategy_params.get(name, {})
            results[name] = self.run_strategy(name, data, params)
            
        return results

class StrategyBacktestRunner:
    """
    Klasa do przeprowadzania backtestów i porównywania strategii.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_sizing_method: str = "volatility"
    ):
        self.initial_capital = initial_capital
        self.position_sizing_method = position_sizing_method
        self.strategies = {
            'moving_average_crossover': self._moving_average_crossover,
            'rsi_reversal': self._rsi_reversal
        }
        
    def run_strategy_comparison(
        self,
        data: pd.DataFrame,
        strategies_to_test: List[str],
        params_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Porównuje wyniki różnych strategii na tych samych danych.

        Parameters:
            data (pd.DataFrame): Dane OHLCV
            strategies_to_test (List[str]): Lista nazw strategii do przetestowania
            params_dict (Dict[str, Dict[str, Any]]): Parametry dla każdej strategii

        Returns:
            Dict[str, Dict[str, Any]]: Wyniki dla każdej strategii
        """
        results = {}
        
        for strategy_name in strategies_to_test:
            if strategy_name not in self.strategies:
                logging.warning(f"Strategia {strategy_name} nie została znaleziona")
                continue
                
            # Pobierz funkcję strategii i jej parametry
            strategy_fn = self.strategies[strategy_name]
            params = params_dict.get(strategy_name, {})
            
            # Generuj sygnały
            signals = strategy_fn(data, **params)
            
            # Oblicz zwroty i metryki
            returns = self._calculate_returns(data, signals)
            metrics = self._calculate_metrics(returns)
            equity_curve = self._calculate_equity_curve(returns)
            
            results[strategy_name] = {
                'signals': signals,
                'returns': returns,
                'metrics': metrics,
                'equity_curve': equity_curve
            }
            
        return results
        
    def plot_strategy_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Generuje wykres porównawczy strategii.

        Parameters:
            results: Wyniki z run_strategy_comparison
            save_path: Opcjonalna ścieżka do zapisu wykresu
        """
        plt.figure(figsize=(12, 8))
        
        for strategy_name, strategy_results in results.items():
            equity_curve = strategy_results['equity_curve']
            plt.plot(equity_curve.index, equity_curve.values, 
                    label=f"{strategy_name} ({strategy_results['metrics']['total_return']:.2%})")
            
        plt.title("Porównanie strategii")
        plt.xlabel("Data")
        plt.ylabel("Wartość portfela")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def _moving_average_crossover(
        self,
        data: pd.DataFrame,
        short_window: int = 10,
        long_window: int = 30
    ) -> pd.Series:
        """Strategia przekroczenia średnich kroczących"""
        short_ma = data['close'].rolling(window=short_window).mean()
        long_ma = data['close'].rolling(window=long_window).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        
        return signals
        
    def _rsi_reversal(
        self,
        data: pd.DataFrame,
        period: int = 14,
        overbought: int = 70,
        oversold: int = 30
    ) -> pd.Series:
        """Strategia odwrócenia RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi > overbought] = -1
        signals[rsi < oversold] = 1
        
        return signals
        
    def _calculate_returns(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """Oblicza zwroty dla serii sygnałów"""
        price_returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        return strategy_returns.fillna(0)
        
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Oblicza podstawowe metryki dla strategii"""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Oblicz maksymalny drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown)
        }
        
    def _calculate_equity_curve(self, returns: pd.Series) -> pd.Series:
        """Oblicza krzywą kapitału"""
        return self.initial_capital * (1 + returns).cumprod()
    
    def _initialize_strategies(self) -> Dict[str, Callable]:
        """
        Inicjalizuje dostępne strategie
        """
        return {
            'moving_average_crossover': self._moving_average_crossover_strategy,
            'rsi_reversal': self._rsi_reversal_strategy,
            'bollinger_bands': self._bollinger_bands_strategy,
            'momentum': self._momentum_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'breakout': self._breakout_strategy
        }
    
    def plot_strategy_comparison(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ) -> None:
        """
        Generuje wykresy porównawcze dla strategii
        """
        plt.figure(figsize=(15, 10))
        
        # Wykres krzywych kapitału
        plt.subplot(2, 2, 1)
        for strategy_name, strategy_results in results.items():
            equity_curve = strategy_results['splits'][-1]['test_results']['equity_curve']
            plt.plot(equity_curve, label=strategy_name)
        plt.title('Porównanie krzywych kapitału')
        plt.legend()
        
        # Wykres statystyk
        metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
        values = {metric: [] for metric in metrics}
        strategy_names = []
        
        for strategy_name, strategy_results in results.items():
            strategy_names.append(strategy_name)
            for metric in metrics:
                values[metric].append(
                    strategy_results['aggregated_metrics'][f'{metric}_mean']
                )
        
        # Wykres wskaźników
        plt.subplot(2, 2, 2)
        x = np.arange(len(strategy_names))
        width = 0.15
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, values[metric], width, label=metric)
        plt.xticks(x + width*1.5, strategy_names, rotation=45)
        plt.title('Porównanie wskaźników')
        plt.legend()
        
        # Wykres drawdownów
        plt.subplot(2, 2, 3)
        for strategy_name, strategy_results in results.items():
            drawdowns = []
            equity_curve = pd.Series(
                strategy_results['splits'][-1]['test_results']['equity_curve']
            )
            peak = equity_curve.expanding().max()
            drawdown = (peak - equity_curve) / peak
            plt.plot(drawdown, label=strategy_name)
        plt.title('Porównanie drawdownów')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def _moving_average_crossover_strategy(
        self,
        data: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 50
    ) -> pd.Series:
        """
        Strategia przecięcia średnich kroczących
        """
        signals = pd.Series(0, index=data.index)
        
        ma_short = data['close'].rolling(window=short_window).mean()
        ma_long = data['close'].rolling(window=long_window).mean()
        
        # Sygnały kupna
        signals[ma_short > ma_long] = 1
        # Sygnały sprzedaży
        signals[ma_short < ma_long] = -1
        
        return signals
    
    def _rsi_reversal_strategy(
        self,
        data: pd.DataFrame,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30
    ) -> pd.Series:
        """
        Strategia odwrócenia trendu na podstawie RSI
        """
        signals = pd.Series(0, index=data.index)
        
        # Obliczanie RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Sygnały kupna przy przekroczeniu strefy wykupienia
        signals[rsi < oversold] = 1
        # Sygnały sprzedaży przy przekroczeniu strefy wyprzedania
        signals[rsi > overbought] = -1
        
        return signals
    
    def _bollinger_bands_strategy(
        self,
        data: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """
        Strategia wykorzystująca wstęgi Bollingera
        """
        signals = pd.Series(0, index=data.index)
        
        # Obliczanie wstęg Bollingera
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Sygnały kupna przy przekroczeniu dolnej wstęgi
        signals[data['close'] < lower_band] = 1
        # Sygnały sprzedaży przy przekroczeniu górnej wstęgi
        signals[data['close'] > upper_band] = -1
        
        return signals
    
    def _momentum_strategy(
        self,
        data: pd.DataFrame,
        lookback: int = 20,
        threshold: float = 0.0
    ) -> pd.Series:
        """
        Strategia momentum
        """
        signals = pd.Series(0, index=data.index)
        
        # Obliczanie momentum
        momentum = data['close'].pct_change(lookback)
        
        # Sygnały kupna przy dodatnim momentum
        signals[momentum > threshold] = 1
        # Sygnały sprzedaży przy ujemnym momentum
        signals[momentum < -threshold] = -1
        
        return signals
    
    def _mean_reversion_strategy(
        self,
        data: pd.DataFrame,
        lookback: int = 20,
        std_threshold: float = 2.0
    ) -> pd.Series:
        """
        Strategia powrotu do średniej
        """
        signals = pd.Series(0, index=data.index)
        
        # Obliczanie odchylenia od średniej
        rolling_mean = data['close'].rolling(window=lookback).mean()
        rolling_std = data['close'].rolling(window=lookback).std()
        z_score = (data['close'] - rolling_mean) / rolling_std
        
        # Sygnały kupna przy dużym odchyleniu w dół
        signals[z_score < -std_threshold] = 1
        # Sygnały sprzedaży przy dużym odchyleniu w górę
        signals[z_score > std_threshold] = -1
        
        return signals
    
    def _breakout_strategy(
        self,
        data: pd.DataFrame,
        lookback: int = 20,
        threshold: float = 0.02
    ) -> pd.Series:
        """
        Strategia przebicia poziomów
        """
        signals = pd.Series(0, index=data.index)
        
        # Obliczanie poziomów wsparcia i oporu
        rolling_high = data['high'].rolling(window=lookback).max()
        rolling_low = data['low'].rolling(window=lookback).min()
        
        # Sygnały kupna przy przebiciu oporu
        breakout_up = data['close'] > rolling_high * (1 + threshold)
        # Sygnały sprzedaży przy przebiciu wsparcia
        breakout_down = data['close'] < rolling_low * (1 - threshold)
        
        signals[breakout_up] = 1
        signals[breakout_down] = -1
        
        return signals