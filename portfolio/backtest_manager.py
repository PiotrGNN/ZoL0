"""
Moduł do zarządzania backtestami i porównywaniem strategii.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BacktestResults:
    """Klasa przechowująca wyniki backtestu."""
    strategy_id: str
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    parameters: Dict[str, Any]

class BacktestManager:
    def __init__(self, db_path: str = 'users.db'):
        """Inicjalizuje menedżera backtestów."""
        self.db_path = db_path
        
    def run_backtest(self, strategy_config: Dict[str, Any], 
                    historical_data: pd.DataFrame,
                    initial_capital: float = 10000.0) -> BacktestResults:
        """
        Przeprowadza backtest strategii.
        
        Args:
            strategy_config: Konfiguracja strategii
            historical_data: Dane historyczne
            initial_capital: Kapitał początkowy
            
        Returns:
            BacktestResults: Wyniki backtestu
        """
        try:
            # Implementacja logiki backtestingu
            trades = []
            equity_curve = [initial_capital]
            current_capital = initial_capital
            
            # Symulacja transakcji
            for i in range(1, len(historical_data)):
                signal = self._generate_signal(historical_data.iloc[:i], strategy_config)
                
                if signal != 0:  # 1 dla long, -1 dla short
                    price = historical_data.iloc[i]['close']
                    position_size = self._calculate_position_size(
                        current_capital,
                        price,
                        strategy_config.get('risk_per_trade', 0.02)
                    )
                    
                    # Symuluj transakcję
                    trade_result = self._simulate_trade(
                        signal,
                        position_size,
                        price,
                        historical_data.iloc[i:],
                        strategy_config
                    )
                    
                    trades.append(trade_result)
                    current_capital += trade_result['pnl']
                    
                equity_curve.append(current_capital)
            
            # Oblicz metryki
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] <= 0])
            
            total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Oblicz Sharpe Ratio
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
            
            # Oblicz maksymalny drawdown
            peaks = pd.Series(equity_curve).cummax()
            drawdowns = (pd.Series(equity_curve) - peaks) / peaks
            max_drawdown = abs(drawdowns.min())
            
            results = BacktestResults(
                strategy_id=strategy_config.get('id', 'unknown'),
                initial_capital=initial_capital,
                final_capital=current_capital,
                total_trades=len(trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                trades=trades,
                equity_curve=equity_curve,
                parameters=strategy_config
            )
            
            # Zapisz wyniki
            self._save_backtest_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Błąd podczas backtestingu: {e}")
            raise
            
    def _generate_signal(self, data: pd.DataFrame, config: Dict[str, Any]) -> int:
        """Generuje sygnał na podstawie strategii."""
        try:
            if config['strategy_type'] == 'trend_following':
                return self._trend_following_strategy(data, config)
            elif config['strategy_type'] == 'mean_reversion':
                return self._mean_reversion_strategy(data, config)
            elif config['strategy_type'] == 'breakout':
                return self._breakout_strategy(data, config)
            else:
                return 0
        except Exception as e:
            logger.error(f"Błąd podczas generowania sygnału: {e}")
            return 0
            
    def _trend_following_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> int:
        """Implementacja strategii podążania za trendem."""
        try:
            # Oblicz SMA
            short_period = config.get('short_ma', 20)
            long_period = config.get('long_ma', 50)
            
            if len(data) < long_period:
                return 0
                
            short_ma = data['close'].rolling(window=short_period).mean()
            long_ma = data['close'].rolling(window=long_period).mean()
            
            # Generuj sygnał
            if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                return 1  # Sygnał kupna
            elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                return -1  # Sygnał sprzedaży
                
            return 0
            
        except Exception as e:
            logger.error(f"Błąd w strategii trend following: {e}")
            return 0
            
    def _mean_reversion_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> int:
        """Implementacja strategii powrotu do średniej."""
        try:
            # Oblicz Bollinger Bands
            period = config.get('period', 20)
            std_dev = config.get('std_dev', 2)
            
            if len(data) < period:
                return 0
                
            sma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Generuj sygnał
            if data['close'].iloc[-1] < lower_band.iloc[-1]:
                return 1  # Sygnał kupna
            elif data['close'].iloc[-1] > upper_band.iloc[-1]:
                return -1  # Sygnał sprzedaży
                
            return 0
            
        except Exception as e:
            logger.error(f"Błąd w strategii mean reversion: {e}")
            return 0
            
    def _breakout_strategy(self, data: pd.DataFrame, config: Dict[str, Any]) -> int:
        """Implementacja strategii przełamania."""
        try:
            # Oblicz poziomy wsparcia/oporu
            period = config.get('period', 20)
            
            if len(data) < period:
                return 0
                
            highest_high = data['high'].rolling(window=period).max()
            lowest_low = data['low'].rolling(window=period).min()
            
            # Generuj sygnał
            if data['close'].iloc[-1] > highest_high.iloc[-2]:
                return 1  # Sygnał kupna
            elif data['close'].iloc[-1] < lowest_low.iloc[-2]:
                return -1  # Sygnał sprzedaży
                
            return 0
            
        except Exception as e:
            logger.error(f"Błąd w strategii breakout: {e}")
            return 0
            
    def _calculate_position_size(self, capital: float, price: float, risk_per_trade: float) -> float:
        """Oblicza wielkość pozycji na podstawie kapitału i ryzyka."""
        return (capital * risk_per_trade) / price
        
    def _simulate_trade(self, signal: int, position_size: float, entry_price: float,
                       future_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Symuluje pojedynczą transakcję."""
        try:
            # Parametry zlecenia
            stop_loss = config.get('stop_loss', 0.02)
            take_profit = config.get('take_profit', 0.03)
            max_bars = config.get('max_bars', 20)
            
            exit_price = entry_price
            exit_time = future_data.index[0]
            reason = 'timeout'
            
            for i in range(min(len(future_data), max_bars)):
                bar = future_data.iloc[i]
                
                # Sprawdź Stop Loss
                if signal == 1 and bar['low'] <= entry_price * (1 - stop_loss):
                    exit_price = entry_price * (1 - stop_loss)
                    exit_time = bar.name
                    reason = 'stop_loss'
                    break
                elif signal == -1 and bar['high'] >= entry_price * (1 + stop_loss):
                    exit_price = entry_price * (1 + stop_loss)
                    exit_time = bar.name
                    reason = 'stop_loss'
                    break
                    
                # Sprawdź Take Profit
                if signal == 1 and bar['high'] >= entry_price * (1 + take_profit):
                    exit_price = entry_price * (1 + take_profit)
                    exit_time = bar.name
                    reason = 'take_profit'
                    break
                elif signal == -1 and bar['low'] <= entry_price * (1 - take_profit):
                    exit_price = entry_price * (1 - take_profit)
                    exit_time = bar.name
                    reason = 'take_profit'
                    break
                    
            # Oblicz PnL
            if signal == 1:
                pnl = position_size * (exit_price - entry_price)
            else:
                pnl = position_size * (entry_price - exit_price)
                
            return {
                'entry_time': future_data.index[0],
                'exit_time': exit_time,
                'signal': signal,
                'position_size': position_size,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Błąd podczas symulacji transakcji: {e}")
            return {
                'entry_time': future_data.index[0],
                'exit_time': future_data.index[0],
                'signal': signal,
                'position_size': position_size,
                'entry_price': entry_price,
                'exit_price': entry_price,
                'pnl': 0,
                'reason': 'error'
            }
            
    def _save_backtest_results(self, results: BacktestResults):
        """Zapisuje wyniki backtestu w bazie danych."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT INTO backtest_results (
                    strategy_id, initial_capital, final_capital,
                    total_trades, winning_trades, losing_trades,
                    sharpe_ratio, sortino_ratio, max_drawdown,
                    parameters, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                results.strategy_id,
                results.initial_capital,
                results.final_capital,
                results.total_trades,
                results.winning_trades,
                results.losing_trades,
                results.sharpe_ratio,
                0.0,  # TODO: Dodać obliczanie Sortino Ratio
                results.max_drawdown,
                json.dumps(results.parameters),
                datetime.now().isoformat()
            ))
            
            backtest_id = c.lastrowid
            
            # Zapisz szczegóły transakcji
            for trade in results.trades:
                c.execute("""
                    INSERT INTO backtest_trades (
                        backtest_id, entry_time, exit_time,
                        signal, position_size, entry_price,
                        exit_price, pnl, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    trade['entry_time'].isoformat(),
                    trade['exit_time'].isoformat(),
                    trade['signal'],
                    trade['position_size'],
                    trade['entry_price'],
                    trade['exit_price'],
                    trade['pnl'],
                    trade['reason']
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania wyników backtestu: {e}")
        finally:
            conn.close()
            
    def get_backtest_history(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Pobiera historię backtestów."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            if strategy_id:
                c.execute("""
                    SELECT * FROM backtest_results 
                    WHERE strategy_id = ? 
                    ORDER BY created_at DESC
                """, (strategy_id,))
            else:
                c.execute("""
                    SELECT * FROM backtest_results 
                    ORDER BY created_at DESC
                """)
                
            columns = [description[0] for description in c.description]
            history = []
            
            for row in c.fetchall():
                result = dict(zip(columns, row))
                result['parameters'] = json.loads(result['parameters'])
                
                # Pobierz transakcje dla tego backtestu
                c.execute("""
                    SELECT * FROM backtest_trades 
                    WHERE backtest_id = ?
                    ORDER BY entry_time
                """, (result['id'],))
                
                trade_columns = [description[0] for description in c.description]
                trades = [dict(zip(trade_columns, trade)) for trade in c.fetchall()]
                result['trades'] = trades
                
                history.append(result)
                
            return history
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania historii backtestów: {e}")
            return []
        finally:
            conn.close()
            
    def compare_strategies(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """Porównuje wyniki różnych strategii."""
        try:
            history = []
            for strategy_id in strategy_ids:
                strategy_history = self.get_backtest_history(strategy_id)
                if strategy_history:
                    history.extend(strategy_history)
            
            if not history:
                return {}
            
            # Przygotuj dane do porównania
            comparison = {
                'strategies': {},
                'best_sharpe': None,
                'best_return': None,
                'lowest_drawdown': None
            }
            
            for result in history:
                strategy_id = result['strategy_id']
                if strategy_id not in comparison['strategies']:
                    comparison['strategies'][strategy_id] = {
                        'backtest_count': 0,
                        'avg_sharpe': 0,
                        'avg_return': 0,
                        'avg_drawdown': 0,
                        'best_result': None,
                        'worst_result': None
                    }
                
                stats = comparison['strategies'][strategy_id]
                stats['backtest_count'] += 1
                
                # Aktualizuj średnie
                return_pct = (result['final_capital'] - result['initial_capital']) / result['initial_capital']
                stats['avg_sharpe'] = (stats['avg_sharpe'] * (stats['backtest_count'] - 1) + result['sharpe_ratio']) / stats['backtest_count']
                stats['avg_return'] = (stats['avg_return'] * (stats['backtest_count'] - 1) + return_pct) / stats['backtest_count']
                stats['avg_drawdown'] = (stats['avg_drawdown'] * (stats['backtest_count'] - 1) + result['max_drawdown']) / stats['backtest_count']
                
                # Aktualizuj najlepszy/najgorszy wynik
                if stats['best_result'] is None or result['sharpe_ratio'] > stats['best_result']['sharpe_ratio']:
                    stats['best_result'] = result
                if stats['worst_result'] is None or result['sharpe_ratio'] < stats['worst_result']['sharpe_ratio']:
                    stats['worst_result'] = result
                
                # Aktualizuj globalne najlepsze wyniki
                if comparison['best_sharpe'] is None or result['sharpe_ratio'] > comparison['best_sharpe']['sharpe_ratio']:
                    comparison['best_sharpe'] = result
                if comparison['best_return'] is None or return_pct > comparison['best_return']['return_pct']:
                    comparison['best_return'] = {**result, 'return_pct': return_pct}
                if comparison['lowest_drawdown'] is None or result['max_drawdown'] < comparison['lowest_drawdown']['max_drawdown']:
                    comparison['lowest_drawdown'] = result
            
            return comparison
            
        except Exception as e:
            logger.error(f"Błąd podczas porównywania strategii: {e}")
            return {}