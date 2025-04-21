#!/usr/bin/env python3
"""
technical_analysis.py - Moduł analizy technicznej dla systemu ZoL0

Ten moduł zawiera klasę TechnicalAnalyzer odpowiedzialną za:
- Obliczanie wskaźników technicznych
- Przeprowadzanie backtestów strategii
- Tworzenie wizualizacji wyników
- Optymalizację strategii tradingowych
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sqlite3
import random
from scipy.optimize import brute, minimize

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/technical_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Klasa do przeprowadzania analizy technicznej i backtestowania strategii.
    
    Umożliwia obliczanie wskaźników technicznych, przeprowadzanie backtestów,
    tworzenie wizualizacji wyników oraz optymalizację parametrów strategii.
    """
    
    def __init__(self, db_path='users.db'):
        """
        Inicjalizuje analizator techniczny.
        
        Args:
            db_path: Ścieżka do bazy danych SQLite
        """
        self.db_path = db_path
        
        # Upewnij się, że potrzebne katalogi istnieją
        for directory in ["logs", "static/img"]:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Zainicjalizowano TechnicalAnalyzer")
    
    def run_backtest(self, strategy_config, historical_data, initial_balance=10000.0):
        """
        Przeprowadza backtest strategii na podstawie dostarczonych danych i konfiguracji.
        
        Args:
            strategy_config: Słownik zawierający konfigurację strategii
            historical_data: DataFrame z danymi historycznymi
            initial_balance: Początkowy kapitał
            
        Returns:
            Słownik zawierający wyniki backtestingu
        """
        try:
            # Konwersja danych historycznych do odpowiedniego formatu
            if isinstance(historical_data, list):
                historical_data = pd.DataFrame(historical_data)
            
            # Przetwórz dane historyczne (upewnij się, że mamy poprawne dane)
            if 'timestamp' in historical_data.columns:
                # Konwersja timestamp z ISO do obiektu datetime
                if isinstance(historical_data['timestamp'].iloc[0], str):
                    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                
                # Ustaw timestamp jako indeks
                historical_data = historical_data.set_index('timestamp')
            
            # Jeśli dane są już typu OHLCV (mają kolumny open, high, low, close, volume)
            if all(col in historical_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                data = historical_data
            else:
                # Próba konwersji danych do formatu OHLCV
                if all(col in historical_data.columns for col in ['open', 'high', 'low', 'close']):
                    if 'volume' not in historical_data.columns:
                        # Dodaj sztuczne wartości wolumenu
                        historical_data['volume'] = 100
                    data = historical_data
                else:
                    # Jeśli dane nie są w odpowiednim formacie, zwróć błąd
                    logger.error("Dane historyczne nie są w odpowiednim formacie OHLCV")
                    return None
            
            # Przygotuj dane do analizy
            # Upewnij się, że nazwy kolumn są małymi literami
            data.columns = [col.lower() for col in data.columns]
            
            # Wybór strategii na podstawie konfiguracji
            strategy_type = strategy_config.get('type', 'simple_ma_crossover')
            
            if strategy_type == 'simple_ma_crossover':
                return self._run_ma_crossover(
                    data,
                    initial_balance,
                    fast_period=strategy_config.get('fast_period', 20),
                    slow_period=strategy_config.get('slow_period', 50)
                )
            elif strategy_type == 'rsi':
                return self._run_rsi_strategy(
                    data,
                    initial_balance,
                    rsi_period=strategy_config.get('rsi_period', 14),
                    oversold=strategy_config.get('oversold', 30),
                    overbought=strategy_config.get('overbought', 70)
                )
            elif strategy_type == 'bollinger_bands':
                return self._run_bollinger_strategy(
                    data,
                    initial_balance,
                    window=strategy_config.get('window', 20),
                    num_std=strategy_config.get('num_std', 2.0)
                )
            else:
                logger.error(f"Nieznany typ strategii: {strategy_type}")
                return None
                
        except Exception as e:
            logger.error(f"Błąd podczas przeprowadzania backtestingu: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _run_ma_crossover(self, data, initial_balance, fast_period=20, slow_period=50):
        """
        Przeprowadza backtest strategii przecięcia średnich ruchomych.
        
        Args:
            data: DataFrame z danymi historycznymi
            initial_balance: Początkowy kapitał
            fast_period: Okres krótszej średniej
            slow_period: Okres dłuższej średniej
            
        Returns:
            Słownik zawierający wyniki backtestingu
        """
        # Oblicz średnie ruchome
        data['ma_fast'] = data['close'].rolling(window=fast_period).mean()
        data['ma_slow'] = data['close'].rolling(window=slow_period).mean()
        
        # Usuń rekordy z NaN (początkowe okresy)
        data = data.dropna()
        
        # Inicjalizacja zmiennych backtestingu
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        
        # Przeprowadzenie backtestingu
        for i in range(1, len(data)):
            # Sprawdź sygnały
            # Golden cross (przecięcie MA_fast powyżej MA_slow)
            if data['ma_fast'].iloc[i-1] < data['ma_slow'].iloc[i-1] and \
               data['ma_fast'].iloc[i] > data['ma_slow'].iloc[i]:
                # Kupuj
                if position <= 0:
                    # Jeśli byliśmy w pozycji krótkiej, zamknij ją
                    if position < 0:
                        profit_loss = position * (entry_price - data['close'].iloc[i])
                        balance += profit_loss
                        trades.append({
                            'timestamp': data.index[i],
                            'type': 'exit_short',
                            'price': data['close'].iloc[i],
                            'profit_loss': profit_loss,
                            'balance': balance
                        })
                    
                    # Otwórz pozycję długą
                    position = balance / data['close'].iloc[i]
                    entry_price = data['close'].iloc[i]
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'enter_long',
                        'price': entry_price,
                        'position': position,
                        'balance': balance
                    })
            
            # Death cross (przecięcie MA_fast poniżej MA_slow)
            elif data['ma_fast'].iloc[i-1] > data['ma_slow'].iloc[i-1] and \
                 data['ma_fast'].iloc[i] < data['ma_slow'].iloc[i]:
                # Sprzedaj
                if position >= 0:
                    # Jeśli byliśmy w pozycji długiej, zamknij ją
                    if position > 0:
                        profit_loss = position * (data['close'].iloc[i] - entry_price)
                        balance += profit_loss
                        trades.append({
                            'timestamp': data.index[i],
                            'type': 'exit_long',
                            'price': data['close'].iloc[i],
                            'profit_loss': profit_loss,
                            'balance': balance
                        })
                    
                    # Otwórz pozycję krótką
                    position = -balance / data['close'].iloc[i]
                    entry_price = data['close'].iloc[i]
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'enter_short',
                        'price': entry_price,
                        'position': position,
                        'balance': balance
                    })
            
            # Aktualizacja equity curve
            if position > 0:
                # Długa pozycja
                equity = balance + position * (data['close'].iloc[i] - entry_price)
            elif position < 0:
                # Krótka pozycja
                equity = balance + position * (entry_price - data['close'].iloc[i])
            else:
                # Brak pozycji
                equity = balance
                
            equity_curve.append(equity)
        
        # Zamknij ostatnią pozycję
        if position != 0:
            last_price = data['close'].iloc[-1]
            if position > 0:
                profit_loss = position * (last_price - entry_price)
            else:
                profit_loss = position * (entry_price - last_price)
                
            balance += profit_loss
            trades.append({
                'timestamp': data.index[-1],
                'type': 'exit_position',
                'price': last_price,
                'profit_loss': profit_loss,
                'balance': balance
            })
        
        # Oblicz statystyki
        returns = np.diff(equity_curve) / np.array(equity_curve)[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Oblicz statystyki transakcji
        winning_trades = [t['profit_loss'] for t in trades if 'profit_loss' in t and t['profit_loss'] > 0]
        losing_trades = [t['profit_loss'] for t in trades if 'profit_loss' in t and t['profit_loss'] <= 0]
        
        # Oblicz dodatkowe metryki
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if (len(winning_trades) + len(losing_trades)) > 0 else 0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
        
        # Przygotuj wyniki
        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'profit_loss': balance - initial_balance,
            'profit_loss_percent': (balance / initial_balance - 1) * 100,
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': {
                'total_trades': len(winning_trades) + len(losing_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'parameters': {
                'fast_period': fast_period,
                'slow_period': slow_period
            }
        }
        
        return results
    
    def _run_rsi_strategy(self, data, initial_balance, rsi_period=14, oversold=30, overbought=70):
        """
        Przeprowadza backtest strategii RSI.
        
        Args:
            data: DataFrame z danymi historycznymi
            initial_balance: Początkowy kapitał
            rsi_period: Okres RSI
            oversold: Poziom wyprzedania
            overbought: Poziom wykupienia
            
        Returns:
            Słownik zawierający wyniki backtestingu
        """
        # Oblicz RSI
        delta = data['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        ma_up = up.rolling(window=rsi_period).mean()
        ma_down = down.rolling(window=rsi_period).mean()
        
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        data['rsi'] = rsi
        
        # Usuń rekordy z NaN (początkowe okresy)
        data = data.dropna()
        
        # Inicjalizacja zmiennych backtestingu
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        
        # Przeprowadzenie backtestingu
        for i in range(1, len(data)):
            # Sprawdź sygnały
            # Sygnał kupna (RSI przebija poziom wyprzedania od dołu)
            if data['rsi'].iloc[i-1] < oversold and data['rsi'].iloc[i] > oversold:
                # Kupuj
                if position <= 0:
                    # Jeśli byliśmy w pozycji krótkiej, zamknij ją
                    if position < 0:
                        profit_loss = position * (entry_price - data['close'].iloc[i])
                        balance += profit_loss
                        trades.append({
                            'timestamp': data.index[i],
                            'type': 'exit_short',
                            'price': data['close'].iloc[i],
                            'profit_loss': profit_loss,
                            'balance': balance
                        })
                    
                    # Otwórz pozycję długą
                    position = balance / data['close'].iloc[i]
                    entry_price = data['close'].iloc[i]
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'enter_long',
                        'price': entry_price,
                        'position': position,
                        'balance': balance
                    })
            
            # Sygnał sprzedaży (RSI przebija poziom wykupienia od góry)
            elif data['rsi'].iloc[i-1] > overbought and data['rsi'].iloc[i] < overbought:
                # Sprzedaj
                if position >= 0:
                    # Jeśli byliśmy w pozycji długiej, zamknij ją
                    if position > 0:
                        profit_loss = position * (data['close'].iloc[i] - entry_price)
                        balance += profit_loss
                        trades.append({
                            'timestamp': data.index[i],
                            'type': 'exit_long',
                            'price': data['close'].iloc[i],
                            'profit_loss': profit_loss,
                            'balance': balance
                        })
                    
                    # Otwórz pozycję krótką
                    position = -balance / data['close'].iloc[i]
                    entry_price = data['close'].iloc[i]
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'enter_short',
                        'price': entry_price,
                        'position': position,
                        'balance': balance
                    })
            
            # Aktualizacja equity curve
            if position > 0:
                # Długa pozycja
                equity = balance + position * (data['close'].iloc[i] - entry_price)
            elif position < 0:
                # Krótka pozycja
                equity = balance + position * (entry_price - data['close'].iloc[i])
            else:
                # Brak pozycji
                equity = balance
                
            equity_curve.append(equity)
        
        # Zamknij ostatnią pozycję
        if position != 0:
            last_price = data['close'].iloc[-1]
            if position > 0:
                profit_loss = position * (last_price - entry_price)
            else:
                profit_loss = position * (entry_price - last_price)
                
            balance += profit_loss
            trades.append({
                'timestamp': data.index[-1],
                'type': 'exit_position',
                'price': last_price,
                'profit_loss': profit_loss,
                'balance': balance
            })
        
        # Oblicz statystyki jak w _run_ma_crossover
        returns = np.diff(equity_curve) / np.array(equity_curve)[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Oblicz statystyki transakcji
        winning_trades = [t['profit_loss'] for t in trades if 'profit_loss' in t and t['profit_loss'] > 0]
        losing_trades = [t['profit_loss'] for t in trades if 'profit_loss' in t and t['profit_loss'] <= 0]
        
        # Oblicz dodatkowe metryki
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if (len(winning_trades) + len(losing_trades)) > 0 else 0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
        
        # Przygotuj wyniki
        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'profit_loss': balance - initial_balance,
            'profit_loss_percent': (balance / initial_balance - 1) * 100,
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': {
                'total_trades': len(winning_trades) + len(losing_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'parameters': {
                'rsi_period': rsi_period,
                'oversold': oversold,
                'overbought': overbought
            }
        }
        
        return results
    
    def _run_bollinger_strategy(self, data, initial_balance, window=20, num_std=2.0):
        """
        Przeprowadza backtest strategii wstęg Bollingera.
        
        Args:
            data: DataFrame z danymi historycznymi
            initial_balance: Początkowy kapitał
            window: Okno dla średniej ruchomej
            num_std: Liczba odchyleń standardowych
            
        Returns:
            Słownik zawierający wyniki backtestingu
        """
        # Oblicz wstęgi Bollingera
        data['ma'] = data['close'].rolling(window=window).mean()
        data['std'] = data['close'].rolling(window=window).std()
        data['upper'] = data['ma'] + num_std * data['std']
        data['lower'] = data['ma'] - num_std * data['std']
        
        # Usuń rekordy z NaN (początkowe okresy)
        data = data.dropna()
        
        # Inicjalizacja zmiennych backtestingu
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        
        # Przeprowadzenie backtestingu
        for i in range(1, len(data)):
            # Sprawdź sygnały
            # Sygnał kupna (cena przebija dolną wstęgę Bollingera od dołu)
            if data['close'].iloc[i-1] < data['lower'].iloc[i-1] and \
               data['close'].iloc[i] > data['lower'].iloc[i]:
                # Kupuj
                if position <= 0:
                    # Jeśli byliśmy w pozycji krótkiej, zamknij ją
                    if position < 0:
                        profit_loss = position * (entry_price - data['close'].iloc[i])
                        balance += profit_loss
                        trades.append({
                            'timestamp': data.index[i],
                            'type': 'exit_short',
                            'price': data['close'].iloc[i],
                            'profit_loss': profit_loss,
                            'balance': balance
                        })
                    
                    # Otwórz pozycję długą
                    position = balance / data['close'].iloc[i]
                    entry_price = data['close'].iloc[i]
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'enter_long',
                        'price': entry_price,
                        'position': position,
                        'balance': balance
                    })
            
            # Sygnał sprzedaży (cena przebija górną wstęgę Bollingera od góry)
            elif data['close'].iloc[i-1] > data['upper'].iloc[i-1] and \
                 data['close'].iloc[i] < data['upper'].iloc[i]:
                # Sprzedaj
                if position >= 0:
                    # Jeśli byliśmy w pozycji długiej, zamknij ją
                    if position > 0:
                        profit_loss = position * (data['close'].iloc[i] - entry_price)
                        balance += profit_loss
                        trades.append({
                            'timestamp': data.index[i],
                            'type': 'exit_long',
                            'price': data['close'].iloc[i],
                            'profit_loss': profit_loss,
                            'balance': balance
                        })
                    
                    # Otwórz pozycję krótką
                    position = -balance / data['close'].iloc[i]
                    entry_price = data['close'].iloc[i]
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'enter_short',
                        'price': entry_price,
                        'position': position,
                        'balance': balance
                    })
            
            # Aktualizacja equity curve
            if position > 0:
                # Długa pozycja
                equity = balance + position * (data['close'].iloc[i] - entry_price)
            elif position < 0:
                # Krótka pozycja
                equity = balance + position * (entry_price - data['close'].iloc[i])
            else:
                # Brak pozycji
                equity = balance
                
            equity_curve.append(equity)
        
        # Zamknij ostatnią pozycję
        if position != 0:
            last_price = data['close'].iloc[-1]
            if position > 0:
                profit_loss = position * (last_price - entry_price)
            else:
                profit_loss = position * (entry_price - last_price)
                
            balance += profit_loss
            trades.append({
                'timestamp': data.index[-1],
                'type': 'exit_position',
                'price': last_price,
                'profit_loss': profit_loss,
                'balance': balance
            })
        
        # Oblicz statystyki jak w poprzednich strategiach
        returns = np.diff(equity_curve) / np.array(equity_curve)[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Oblicz statystyki transakcji
        winning_trades = [t['profit_loss'] for t in trades if 'profit_loss' in t and t['profit_loss'] > 0]
        losing_trades = [t['profit_loss'] for t in trades if 'profit_loss' in t and t['profit_loss'] <= 0]
        
        # Oblicz dodatkowe metryki
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if (len(winning_trades) + len(losing_trades)) > 0 else 0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) != 0 else float('inf')
        
        # Przygotuj wyniki
        results = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'profit_loss': balance - initial_balance,
            'profit_loss_percent': (balance / initial_balance - 1) * 100,
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': {
                'total_trades': len(winning_trades) + len(losing_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'parameters': {
                'window': window,
                'num_std': num_std
            }
        }
        
        return results
    
    def plot_backtest_results(self, results, save_path=None):
        """
        Tworzy wizualizację wyników backtestingu.
        
        Args:
            results: Słownik zawierający wyniki backtestingu
            save_path: Ścieżka do zapisania wykresu (opcjonalne)
            
        Returns:
            None
        """
        try:
            # Utwórz wykres
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Wykres 1: Krzywa equity
            equity_curve = results['equity_curve']
            axes[0].plot(equity_curve, label='Equity Curve')
            axes[0].set_title('Backtest Results')
            axes[0].set_ylabel('Equity')
            axes[0].legend()
            axes[0].grid(True)
            
            # Wykres 2: Drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (np.array(equity_curve) - peak) / peak
            axes[1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            axes[1].set_ylabel('Drawdown')
            axes[1].grid(True)
            
            # Wykres 3: Zyski/straty z transakcji
            trades_pnl = [t['profit_loss'] for t in results['trades'] if 'profit_loss' in t]
            if trades_pnl:
                axes[2].bar(range(len(trades_pnl)), trades_pnl, color=['green' if pnl > 0 else 'red' for pnl in trades_pnl])
                axes[2].set_xlabel('Trade #')
                axes[2].set_ylabel('Profit/Loss')
                axes[2].grid(True)
            
            # Dodaj informacje statystyczne
            metrics = results['metrics']
            stats_text = (
                f"Initial Balance: ${results['initial_balance']:.2f}\n"
                f"Final Balance: ${results['final_balance']:.2f}\n"
                f"Profit/Loss: ${results['profit_loss']:.2f} ({results['profit_loss_percent']:.2f}%)\n"
                f"Total Trades: {metrics['total_trades']}\n"
                f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%\n"
                f"Profit Factor: {metrics['profit_factor']:.2f}"
            )
            
            axes[0].annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                            va='top', fontsize=10)
            
            plt.tight_layout()
            
            # Zapisz wykres, jeśli podano ścieżkę
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Zapisano wykres wyników backtestingu do {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Błąd podczas tworzenia wykresu wyników backtestingu: {e}")
    
    def optimize_strategy(self, strategy_config, historical_data, param_grid):
        """
        Optymalizuje parametry strategii.
        
        Args:
            strategy_config: Słownik zawierający podstawową konfigurację strategii
            historical_data: DataFrame z danymi historycznymi
            param_grid: Słownik zawierający zakresy parametrów do optymalizacji
            
        Returns:
            Słownik zawierający wyniki optymalizacji
        """
        try:
            strategy_type = strategy_config.get('type', 'simple_ma_crossover')
            
            # Przygotuj dane historyczne
            if isinstance(historical_data, list):
                historical_data = pd.DataFrame(historical_data)
            
            # Konwersja danych do odpowiedniego formatu
            if 'timestamp' in historical_data.columns:
                if isinstance(historical_data['timestamp'].iloc[0], str):
                    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
                historical_data = historical_data.set_index('timestamp')
            
            # Sprawdź, czy dane są w formacie OHLCV
            if not all(col in historical_data.columns for col in ['open', 'high', 'low', 'close']):
                logger.error("Dane historyczne nie zawierają kolumn OHLCV")
                return None
            
            # Upewnij się, że nazwy kolumn są małymi literami
            historical_data.columns = [col.lower() for col in historical_data.columns]
            if 'volume' not in historical_data.columns:
                historical_data['volume'] = 100
            
            # Podziel dane na zestawy trenujący i testowy (80/20)
            train_size = int(len(historical_data) * 0.8)
            train_data = historical_data[:train_size]
            test_data = historical_data[train_size:]
            
            # Funkcja optymalizacji dla danego typu strategii
            if strategy_type == 'simple_ma_crossover':
                return self._optimize_ma_crossover(train_data, test_data, param_grid)
            elif strategy_type == 'rsi':
                return self._optimize_rsi(train_data, test_data, param_grid)
            elif strategy_type == 'bollinger_bands':
                return self._optimize_bollinger(train_data, test_data, param_grid)
            else:
                logger.error(f"Nieznany typ strategii: {strategy_type}")
                return None
                
        except Exception as e:
            logger.error(f"Błąd podczas optymalizacji strategii: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _optimize_ma_crossover(self, train_data, test_data, param_grid):
        """
        Optymalizuje parametry strategii przecięcia średnich ruchomych.
        
        Args:
            train_data: DataFrame z danymi treningowymi
            test_data: DataFrame z danymi testowymi
            param_grid: Słownik zawierający zakresy parametrów do optymalizacji
            
        Returns:
            Słownik zawierający wyniki optymalizacji
        """
        # Pobierz zakresy parametrów
        fast_period_range = param_grid.get('fast_period', [5, 10, 15, 20, 25])
        slow_period_range = param_grid.get('slow_period', [30, 40, 50, 60, 70])
        
        # Lista przechowująca wszystkie wyniki
        all_results = []
        
        # Optymalizacja na danych treningowych
        for fast_period in fast_period_range:
            for slow_period in slow_period_range:
                # Pomiń nieprawidłowe kombinacje (szybka średnia musi być krótsza niż wolna)
                if fast_period >= slow_period:
                    continue
                    
                # Przeprowadź backtest
                results = self._run_ma_crossover(
                    train_data,
                    10000.0,  # Stały kapitał początkowy
                    fast_period=fast_period,
                    slow_period=slow_period
                )
                
                # Jeśli backtest zakończony powodzeniem, zapisz wyniki
                if results:
                    # Dodaj parametry do wyników
                    result_entry = {
                        'parameters': {
                            'fast_period': fast_period,
                            'slow_period': slow_period
                        },
                        'metrics': results['metrics'],
                        'profit_loss_percent': results['profit_loss_percent']
                    }
                    
                    all_results.append(result_entry)
        
        # Jeśli nie ma wyników, zwróć None
        if not all_results:
            logger.warning("Brak wyników optymalizacji dla strategii MA Crossover")
            return None
        
        # Posortuj wyniki według zwrotu procentowego (malejąco)
        all_results.sort(key=lambda x: x['profit_loss_percent'], reverse=True)
        
        # Pobierz najlepszy zestaw parametrów
        best_params = all_results[0]['parameters']
        
        # Przeprowadź backtest na danych testowych z najlepszymi parametrami
        test_results = self._run_ma_crossover(
            test_data,
            10000.0,  # Stały kapitał początkowy
            fast_period=best_params['fast_period'],
            slow_period=best_params['slow_period']
        )
        
        # Przygotuj wyniki
        if test_results:
            return {
                'best_result': {
                    'parameters': best_params,
                    'train_metrics': all_results[0]['metrics'],
                    'train_profit_percent': all_results[0]['profit_loss_percent'],
                    'test_metrics': test_results['metrics'],
                    'test_profit_percent': test_results['profit_loss_percent']
                },
                'all_results': all_results[:10]  # Zwróć tylko 10 najlepszych wyników
            }
        else:
            return {
                'best_result': {
                    'parameters': best_params,
                    'train_metrics': all_results[0]['metrics'],
                    'train_profit_percent': all_results[0]['profit_loss_percent'],
                    'test_metrics': {},
                    'test_profit_percent': 0.0
                },
                'all_results': all_results[:10]  # Zwróć tylko 10 najlepszych wyników
            }
    
    def _optimize_rsi(self, train_data, test_data, param_grid):
        """
        Optymalizuje parametry strategii RSI.
        
        Args:
            train_data: DataFrame z danymi treningowymi
            test_data: DataFrame z danymi testowymi
            param_grid: Słownik zawierający zakresy parametrów do optymalizacji
            
        Returns:
            Słownik zawierający wyniki optymalizacji
        """
        # Pobierz zakresy parametrów
        rsi_period_range = param_grid.get('rsi_period', [7, 10, 14, 21])
        oversold_range = param_grid.get('oversold', [20, 25, 30, 35])
        overbought_range = param_grid.get('overbought', [65, 70, 75, 80])
        
        # Lista przechowująca wszystkie wyniki
        all_results = []
        
        # Optymalizacja na danych treningowych
        for rsi_period in rsi_period_range:
            for oversold in oversold_range:
                for overbought in overbought_range:
                    # Pomiń nieprawidłowe kombinacje
                    if oversold >= overbought:
                        continue
                        
                    # Przeprowadź backtest
                    results = self._run_rsi_strategy(
                        train_data,
                        10000.0,  # Stały kapitał początkowy
                        rsi_period=rsi_period,
                        oversold=oversold,
                        overbought=overbought
                    )
                    
                    # Jeśli backtest zakończony powodzeniem, zapisz wyniki
                    if results:
                        # Dodaj parametry do wyników
                        result_entry = {
                            'parameters': {
                                'rsi_period': rsi_period,
                                'oversold': oversold,
                                'overbought': overbought
                            },
                            'metrics': results['metrics'],
                            'profit_loss_percent': results['profit_loss_percent']
                        }
                        
                        all_results.append(result_entry)
        
        # Jeśli nie ma wyników, zwróć None
        if not all_results:
            logger.warning("Brak wyników optymalizacji dla strategii RSI")
            return None
        
        # Posortuj wyniki według zwrotu procentowego (malejąco)
        all_results.sort(key=lambda x: x['profit_loss_percent'], reverse=True)
        
        # Pobierz najlepszy zestaw parametrów
        best_params = all_results[0]['parameters']
        
        # Przeprowadź backtest na danych testowych z najlepszymi parametrami
        test_results = self._run_rsi_strategy(
            test_data,
            10000.0,  # Stały kapitał początkowy
            rsi_period=best_params['rsi_period'],
            oversold=best_params['oversold'],
            overbought=best_params['overbought']
        )
        
        # Przygotuj wyniki
        if test_results:
            return {
                'best_result': {
                    'parameters': best_params,
                    'train_metrics': all_results[0]['metrics'],
                    'train_profit_percent': all_results[0]['profit_loss_percent'],
                    'test_metrics': test_results['metrics'],
                    'test_profit_percent': test_results['profit_loss_percent']
                },
                'all_results': all_results[:10]  # Zwróć tylko 10 najlepszych wyników
            }
        else:
            return {
                'best_result': {
                    'parameters': best_params,
                    'train_metrics': all_results[0]['metrics'],
                    'train_profit_percent': all_results[0]['profit_loss_percent'],
                    'test_metrics': {},
                    'test_profit_percent': 0.0
                },
                'all_results': all_results[:10]  # Zwróć tylko 10 najlepszych wyników
            }
    
    def _optimize_bollinger(self, train_data, test_data, param_grid):
        """
        Optymalizuje parametry strategii wstęg Bollingera.
        
        Args:
            train_data: DataFrame z danymi treningowymi
            test_data: DataFrame z danymi testowymi
            param_grid: Słownik zawierający zakresy parametrów do optymalizacji
            
        Returns:
            Słownik zawierający wyniki optymalizacji
        """
        # Pobierz zakresy parametrów
        window_range = param_grid.get('window', [10, 15, 20, 25, 30])
        num_std_range = param_grid.get('num_std', [1.5, 2.0, 2.5, 3.0])
        
        # Lista przechowująca wszystkie wyniki
        all_results = []
        
        # Optymalizacja na danych treningowych
        for window in window_range:
            for num_std in num_std_range:
                # Przeprowadź backtest
                results = self._run_bollinger_strategy(
                    train_data,
                    10000.0,  # Stały kapitał początkowy
                    window=window,
                    num_std=num_std
                )
                
                # Jeśli backtest zakończony powodzeniem, zapisz wyniki
                if results:
                    # Dodaj parametry do wyników
                    result_entry = {
                        'parameters': {
                            'window': window,
                            'num_std': num_std
                        },
                        'metrics': results['metrics'],
                        'profit_loss_percent': results['profit_loss_percent']
                    }
                    
                    all_results.append(result_entry)
        
        # Jeśli nie ma wyników, zwróć None
        if not all_results:
            logger.warning("Brak wyników optymalizacji dla strategii Bollinger Bands")
            return None
        
        # Posortuj wyniki według zwrotu procentowego (malejąco)
        all_results.sort(key=lambda x: x['profit_loss_percent'], reverse=True)
        
        # Pobierz najlepszy zestaw parametrów
        best_params = all_results[0]['parameters']
        
        # Przeprowadź backtest na danych testowych z najlepszymi parametrami
        test_results = self._run_bollinger_strategy(
            test_data,
            10000.0,  # Stały kapitał początkowy
            window=best_params['window'],
            num_std=best_params['num_std']
        )
        
        # Przygotuj wyniki
        if test_results:
            return {
                'best_result': {
                    'parameters': best_params,
                    'train_metrics': all_results[0]['metrics'],
                    'train_profit_percent': all_results[0]['profit_loss_percent'],
                    'test_metrics': test_results['metrics'],
                    'test_profit_percent': test_results['profit_loss_percent']
                },
                'all_results': all_results[:10]  # Zwróć tylko 10 najlepszych wyników
            }
        else:
            return {
                'best_result': {
                    'parameters': best_params,
                    'train_metrics': all_results[0]['metrics'],
                    'train_profit_percent': all_results[0]['profit_loss_percent'],
                    'test_metrics': {},
                    'test_profit_percent': 0.0
                },
                'all_results': all_results[:10]  # Zwróć tylko 10 najlepszych wyników
            }
    
    def get_custom_indicators(self, user_id):
        """
        Pobiera niestandardowe wskaźniki dla danego użytkownika.
        
        Args:
            user_id: ID użytkownika
            
        Returns:
            Lista słowników z niestandardowymi wskaźnikami
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Sprawdź, czy istnieje tabela z niestandardowymi wskaźnikami
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='custom_indicators'")
            if not c.fetchone():
                # Jeśli nie istnieje, utwórz ją
                c.execute('''
                    CREATE TABLE custom_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        formula TEXT NOT NULL,
                        parameters TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
            
            # Pobierz wskaźniki dla danego użytkownika
            c.execute('''
                SELECT id, name, formula, parameters, description, created_at
                FROM custom_indicators
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            rows = c.fetchall()
            conn.close()
            
            # Przygotuj wyniki
            columns = ['id', 'name', 'formula', 'parameters', 'description', 'created_at']
            indicators = []
            
            for row in rows:
                indicator = dict(zip(columns, row))
                
                # Rozpakuj parametry z formatu JSON
                if indicator['parameters']:
                    try:
                        indicator['parameters'] = json.loads(indicator['parameters'])
                    except:
                        indicator['parameters'] = {}
                else:
                    indicator['parameters'] = {}
                    
                indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Błąd podczas pobierania niestandardowych wskaźników: {e}")
            return []
    
    def add_custom_indicator(self, user_id, indicator_data):
        """
        Dodaje niestandardowy wskaźnik dla danego użytkownika.
        
        Args:
            user_id: ID użytkownika
            indicator_data: Słownik zawierający dane wskaźnika
            
        Returns:
            bool: True jeśli dodano wskaźnik, False w przeciwnym razie
        """
        try:
            # Sprawdź, czy podano wymagane pola
            if not all(key in indicator_data for key in ['name', 'formula']):
                logger.error("Brak wymaganych pól wskaźnika (name, formula)")
                return False
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Sprawdź, czy istnieje tabela z niestandardowymi wskaźnikami
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='custom_indicators'")
            if not c.fetchone():
                # Jeśli nie istnieje, utwórz ją
                c.execute('''
                    CREATE TABLE custom_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        formula TEXT NOT NULL,
                        parameters TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
            
            # Sprawdź, czy wskaźnik o tej nazwie już istnieje
            c.execute('''
                SELECT id FROM custom_indicators
                WHERE user_id = ? AND name = ?
            ''', (user_id, indicator_data['name']))
            
            if c.fetchone():
                logger.warning(f"Wskaźnik o nazwie '{indicator_data['name']}' już istnieje")
                conn.close()
                return False
            
            # Przygotuj parametry w formacie JSON
            parameters = indicator_data.get('parameters', {})
            parameters_json = json.dumps(parameters) if parameters else None
            
            # Dodaj wskaźnik
            c.execute('''
                INSERT INTO custom_indicators (user_id, name, formula, parameters, description, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                user_id,
                indicator_data['name'],
                indicator_data['formula'],
                parameters_json,
                indicator_data.get('description')
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Dodano niestandardowy wskaźnik '{indicator_data['name']}' dla użytkownika {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas dodawania niestandardowego wskaźnika: {e}")
            return False