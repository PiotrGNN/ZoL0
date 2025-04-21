"""
Enhanced backtester module for advanced trading strategy testing
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple, Callable
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
        position_sizing_method: str = "volatility",
        n_simulations: int = 1000
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.spread = spread
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.position_sizing_method = position_sizing_method
        self.n_simulations = n_simulations
        
    def calculate_position_size(
        self,
        price: float,
        volatility: float,
        risk_per_trade: float,
        available_capital: float
    ) -> float:
        """Oblicza wielkość pozycji na podstawie wybranej metody"""
        if self.position_sizing_method == "volatility":
            position_size = (risk_per_trade * available_capital) / (price * volatility)
        elif self.position_sizing_method == "fixed":
            position_size = (risk_per_trade * available_capital) / price
        else:
            position_size = 0.1 * available_capital / price  # Default 10% kapitału
            
        return position_size
        
    def run_monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 100,
        n_days: int = 30
    ) -> Dict:
        """
        Przeprowadza symulację Monte Carlo dla zwrotów strategii.

        Parameters:
            returns (pd.Series): Seria zwrotów strategii
            n_simulations (int): Liczba symulacji
            n_days (int): Horyzont czasowy symulacji w dniach

        Returns:
            Dict: Wyniki symulacji zawierające paths, final_values_mean i confidence_interval
        """
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generuj ścieżki
        paths = []
        initial_value = 100.0  # Normalizowane do 100
        
        for _ in range(n_simulations):
            # Generuj losowe zwroty z rozkładu normalnego
            daily_returns = np.random.normal(mean_return, std_return, n_days)
            # Oblicz wartości skumulowane
            path_values = initial_value * (1 + daily_returns).cumprod()
            paths.append(path_values)
            
        paths_df = pd.DataFrame(paths)
        final_values = paths_df.iloc[:, -1]
        
        confidence_interval = (
            np.percentile(final_values, 2.5),  # Dolny przedział
            np.percentile(final_values, 97.5)  # Górny przedział
        )
        
        return {
            'paths': paths,
            'final_values_mean': float(final_values.mean()),
            'confidence_interval': list(confidence_interval)
        }
        
    def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        train_ratio: float = 0.7,
        n_splits: int = 3
    ) -> Dict:
        """
        Przeprowadza analizę walk-forward dla strategii.

        Parameters:
            data (pd.DataFrame): Dane cenowe OHLCV
            strategy_fn (Callable): Funkcja strategii generująca sygnały
            train_ratio (float): Proporcja danych treningowych (0-1)
            n_splits (int): Liczba podziałów danych

        Returns:
            Dict: Wyniki analizy zawierające split i metryki zagregowane
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits_results = []
        
        for train_idx, test_idx in tscv.split(data):
            # Podział na zbiór treningowy i testowy
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Generowanie sygnałów na zbiorze testowym
            signals = strategy_fn(test_data)
            returns = self._calculate_returns(test_data, signals)
            metrics = self._calculate_split_metrics(returns)
            
            splits_results.append({
                'train_period': (train_data.index[0].strftime('%Y-%m-%d'), 
                               train_data.index[-1].strftime('%Y-%m-%d')),
                'test_period': (test_data.index[0].strftime('%Y-%m-%d'), 
                              test_data.index[-1].strftime('%Y-%m-%d')),
                'metrics': metrics
            })
        
        # Oblicz zagregowane metryki
        aggregated_metrics = self._aggregate_metrics([split['metrics'] for split in splits_results])
        
        return {
            'splits': splits_results,
            'aggregated_metrics': aggregated_metrics
        }
        
    def _calculate_split_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Oblicza metryki dla pojedynczego splitu.
        """
        if len(returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }
            
        total_return = (1 + returns).prod() - 1
        sharpe = self._calculate_sharpe_ratio(returns)
        max_dd = self._calculate_max_drawdown(returns)
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'volatility': float(volatility)
        }
        
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Agreguje metryki ze wszystkich splitów.
        """
        aggregated = {}
        metrics_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        
        for key in metrics_keys:
            values = [m[key] for m in metrics_list]
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            
        return aggregated
        
    def _calculate_returns(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        Oblicza zwroty na podstawie sygnałów i danych cenowych
        """
        if 'close' not in data.columns:
            raise ValueError("Dane muszą zawierać kolumnę 'close'")
            
        price_returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        return strategy_returns.dropna()
        
    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Oblicza współczynnik Sharpe'a dla serii zwrotów
        """
        excess_returns = returns - risk_free_rate/252
        if len(excess_returns) < 2:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Oblicza maksymalny drawdown dla serii zwrotów
        """
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return drawdowns.min()
        
    def _calculate_confidence_intervals(
        self,
        simulated_paths: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Oblicza przedziały ufności dla symulowanych ścieżek
        """
        return {
            'lower_95': simulated_paths.quantile(0.025, axis=1),
            'upper_95': simulated_paths.quantile(0.975, axis=1),
            'median': simulated_paths.median(axis=1)
        }
        
    def _calculate_drawdown_distribution(
        self,
        simulated_paths: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Oblicza rozkład maksymalnych drawdownów dla symulowanych ścieżek
        """
        drawdowns = []
        
        for col in simulated_paths.columns:
            path = simulated_paths[col]
            rolling_max = path.expanding().max()
            drawdown = path / rolling_max - 1
            drawdowns.append(drawdown.min())
            
        return {
            'mean': np.mean(drawdowns),
            'std': np.std(drawdowns),
            'worst': np.min(drawdowns),
            'var_95': np.percentile(drawdowns, 5)
        }
    
    def _optimize_strategy_params(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        param_grid: Optional[Dict] = None
    ) -> Dict:
        """
        Optymalizuje parametry strategii na danych treningowych
        """
        if param_grid is None:
            # Domyślna siatka parametrów
            param_grid = {
                'lookback': [10, 20, 30],
                'threshold': [1.0, 1.5, 2.0],
                'stop_loss': [0.02, 0.03, 0.04]
            }
            
        best_params = None
        best_sharpe = -np.inf
        
        # Grid search
        for params in self._generate_param_combinations(param_grid):
            results = self._run_backtest_period(
                data,
                lambda x: strategy_fn(x, **params)
            )
            
            if results['metrics']['sharpe_ratio'] > best_sharpe:
                best_sharpe = results['metrics']['sharpe_ratio']
                best_params = params
                
        return best_params
    
    def _run_backtest_period(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable
    ) -> Dict:
        """
        Przeprowadza backtest na danym okresie
        """
        signals = strategy_fn(data)
        equity_curve = self._calculate_equity_curve(data, signals)
        trades = self._extract_trades(data, signals)
        metrics = self._calculate_performance_metrics(equity_curve, trades)
        
        return {
            'equity_curve': equity_curve.tolist(),
            'trades': trades,
            'metrics': metrics
        }
    
    def _calculate_equity_curve(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        Oblicza krzywą kapitału na podstawie sygnałów
        """
        position_sizes = self._calculate_position_sizes(data, signals)
        returns = position_sizes * data['returns']
        
        # Uwzględnienie kosztów transakcyjnych
        trades = signals.diff().fillna(0).abs()
        transaction_costs = trades * (self.commission + self.spread + self.slippage)
        
        net_returns = returns - transaction_costs
        equity_curve = (1 + net_returns).cumprod() * self.initial_capital
        
        return equity_curve
    
    def _calculate_position_sizes(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> pd.Series:
        """
        Oblicza wielkości pozycji dla każdego sygnału
        """
        volatility = data['close'].pct_change().rolling(20).std()
        available_capital = self.initial_capital  # Uproszczone
        risk_per_trade = 0.02  # 2% ryzyko na transakcję
        
        position_sizes = pd.Series(index=signals.index)
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                position_sizes.iloc[i] = self.calculate_position_size(
                    data['close'].iloc[i],
                    volatility.iloc[i],
                    risk_per_trade,
                    available_capital
                ) * signals.iloc[i]
            else:
                position_sizes.iloc[i] = 0
                
        return position_sizes
    
    def _extract_trades(
        self,
        data: pd.DataFrame,
        signals: pd.Series
    ) -> List[Dict]:
        """
        Wyodrębnia poszczególne transakcje z sygnałów
        """
        trades = []
        current_position = None
        
        for i in range(len(signals)):
            if signals.iloc[i] != 0 and current_position is None:
                # Otwarcie pozycji
                current_position = {
                    'entry_time': data.index[i],
                    'entry_price': data['close'].iloc[i],
                    'direction': 'long' if signals.iloc[i] > 0 else 'short',
                    'size': abs(signals.iloc[i])
                }
            elif signals.iloc[i] == 0 and current_position is not None:
                # Zamknięcie pozycji
                exit_price = data['close'].iloc[i]
                pnl = (exit_price - current_position['entry_price']) * current_position['size']
                if current_position['direction'] == 'short':
                    pnl *= -1
                    
                trades.append({
                    'entry_time': current_position['entry_time'],
                    'exit_time': data.index[i],
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'direction': current_position['direction'],
                    'size': current_position['size'],
                    'pnl': pnl,
                    'return': pnl / (current_position['entry_price'] * current_position['size'])
                })
                
                current_position = None
                
        return trades
    
    def _calculate_performance_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict]
    ) -> Dict:
        """
        Oblicza metryki wydajności strategii
        """
        returns = equity_curve.pct_change().dropna()
        
        # Podstawowe metryki
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        cagr = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Metryki ryzyka
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        max_drawdown = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Metryki efektywności
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
        profit_factor = (
            sum(t['pnl'] for t in winning_trades) /
            abs(sum(t['pnl'] for t in losing_trades))
        ) if losing_trades else float('inf')
        
        # Wskaźniki ryzyka-zwrotu
        sharpe_ratio = (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252)
        sortino_ratio = (returns.mean() - self.risk_free_rate/252) / downside_volatility
        
        return {
            'total_return': float(total_return),
            'cagr': float(cagr),
            'volatility': float(volatility),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """
        Generuje wszystkie możliwe kombinacje parametrów
        """
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        return combinations
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """
        Agreguje wyniki z analizy walk-forward
        """
        metrics = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
        aggregated = {}
        
        for metric in metrics:
            values = [r['test_results']['metrics'][metric] for r in results]
            aggregated[f'{metric}_mean'] = float(np.mean(values))
            aggregated[f'{metric}_std'] = float(np.std(values))
            
        return aggregated