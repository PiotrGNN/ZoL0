"""
Moduł do monitorowania wydajności systemu tradingowego.
Śledzi statystyki wydajności, takie jak zysk/strata, współczynnik Sharpe'a, drawdown itp.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Klasa do monitorowania wydajności handlowej systemu.
    Śledzi metryki wydajności, takie jak:
    - Zysk/strata
    - Wskaźnik Sharpe'a
    - Maksymalny drawdown
    - Win rate (% wygranych transakcji)
    - Wartość portfela w czasie
    """
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.portfolio_history = []
        self.start_time = datetime.now()
        
        # Zapisz początkową wartość portfela
        self._record_portfolio_value(initial_balance)
        
        logger.info(f"Zainicjalizowano PerformanceMonitor z początkowym saldem {initial_balance}")

    def _record_portfolio_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Zapisuje wartość portfela w określonym czasie."""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.portfolio_history.append({
            "timestamp": timestamp,
            "value": value
        })
    
    def add_trade(self, trade_data: Dict) -> None:
        """
        Dodaje zakończoną transakcję do historii handlowej.
        
        Args:
            trade_data: Słownik zawierający dane o transakcji:
                - symbol: Symbol handlowy (np. "BTCUSDT")
                - entry_price: Cena wejścia
                - exit_price: Cena wyjścia
                - quantity: Ilość
                - side: "BUY" lub "SELL"
                - entry_time: Czas otwarcia pozycji
                - exit_time: Czas zamknięcia pozycji
                - profit_loss: Zysk/strata (opcjonalne, może być obliczone)
                - commission: Prowizja (opcjonalne)
        """
        if "profit_loss" not in trade_data:
            # Oblicz zysk/stratę jeśli nie podano
            if trade_data["side"] == "BUY":
                trade_data["profit_loss"] = (trade_data["exit_price"] - trade_data["entry_price"]) * trade_data["quantity"]
            else:  # SELL
                trade_data["profit_loss"] = (trade_data["entry_price"] - trade_data["exit_price"]) * trade_data["quantity"]
                
        # Oblicz prowizję jeśli nie podano
        if "commission" not in trade_data:
            # Domyślna prowizja 0.1% od wartości transakcji
            commission_rate = 0.001
            entry_value = trade_data["entry_price"] * trade_data["quantity"]
            exit_value = trade_data["exit_price"] * trade_data["quantity"]
            trade_data["commission"] = (entry_value + exit_value) * commission_rate
        
        # Dodaj czas trwania transakcji
        if "entry_time" in trade_data and "exit_time" in trade_data:
            entry_time = trade_data["entry_time"]
            exit_time = trade_data["exit_time"]
            
            # Konwersja do datetime jeśli podano jako string
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                
            trade_data["duration"] = (exit_time - entry_time).total_seconds() / 3600  # w godzinach
        
        # Aktualizuj saldo
        self.current_balance += trade_data["profit_loss"] - trade_data["commission"]
        
        # Dodaj transakcję do historii
        self.trades.append(trade_data)
        
        # Zapisz nową wartość portfela
        self._record_portfolio_value(self.current_balance, 
                                   trade_data.get("exit_time", datetime.now()))
                                   
        logger.info(f"Dodano transakcję: {trade_data['symbol']} {trade_data['side']} z P/L: {trade_data['profit_loss']:.2f}")

    def update_balance(self, new_balance: float) -> None:
        """
        Aktualizuje bieżące saldo bez dodawania transakcji.
        Przydatne do aktualizacji wartości portfela w czasie rzeczywistym.
        
        Args:
            new_balance: Nowe saldo
        """
        self.current_balance = new_balance
        self._record_portfolio_value(new_balance)
        
    def get_win_rate(self) -> float:
        """
        Oblicza współczynnik wygranych transakcji.
        
        Returns:
            Procent wygranych transakcji (0-100)
        """
        if not self.trades:
            return 0.0
            
        winning_trades = sum(1 for trade in self.trades if trade["profit_loss"] > 0)
        return (winning_trades / len(self.trades)) * 100
        
    def get_profit_loss(self) -> float:
        """
        Oblicza całkowity zysk/stratę.
        
        Returns:
            Wartość zysku/straty
        """
        return self.current_balance - self.initial_balance
        
    def get_profit_loss_percentage(self) -> float:
        """
        Oblicza procentowy zysk/stratę.
        
        Returns:
            Procentowa wartość zysku/straty
        """
        return (self.get_profit_loss() / self.initial_balance) * 100
        
    def get_max_drawdown(self) -> float:
        """
        Oblicza maksymalny drawdown w procentach.
        
        Returns:
            Wartość maksymalnego drawdownu (0-100)
        """
        if len(self.portfolio_history) < 2:
            return 0.0
            
        values = [entry["value"] for entry in self.portfolio_history]
        max_drawdown = 0
        peak_value = values[0]
        
        for value in values:
            if value > peak_value:
                peak_value = value
            else:
                drawdown = (peak_value - value) / peak_value * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    
        return max_drawdown
        
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """
        Oblicza wskaźnik Sharpe'a.
        
        Args:
            risk_free_rate: Stopa wolna od ryzyka (domyślnie 2%)
            trading_days: Liczba dni handlowych w roku (domyślnie 252)
            
        Returns:
            Wartość wskaźnika Sharpe'a
        """
        if len(self.portfolio_history) < 2:
            return 0.0
            
        # Oblicz dzienne zwroty
        values = [entry["value"] for entry in self.portfolio_history]
        returns = np.diff(values) / values[:-1]
        
        # Oblicz średni zwrot i odchylenie standardowe
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        # Annualizuj zwroty
        annualized_return = (1 + mean_return) ** trading_days - 1
        annualized_std = std_return * np.sqrt(trading_days)
        
        # Oblicz wskaźnik Sharpe'a
        daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days) - 1
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
        
        return sharpe_ratio
        
    def get_sortino_ratio(self, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """
        Oblicza wskaźnik Sortino (podobny do Sharpe'a, ale uwzględnia tylko negatywne odchylenia).
        
        Args:
            risk_free_rate: Stopa wolna od ryzyka (domyślnie 2%)
            trading_days: Liczba dni handlowych w roku (domyślnie 252)
            
        Returns:
            Wartość wskaźnika Sortino
        """
        if len(self.portfolio_history) < 2:
            return 0.0
            
        # Oblicz dzienne zwroty
        values = [entry["value"] for entry in self.portfolio_history]
        returns = np.diff(values) / values[:-1]
        
        # Oblicz średni zwrot
        mean_return = np.mean(returns)
        
        # Znajdź negatywne zwroty
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf')  # Brak negatywnych zwrotów
            
        # Oblicz odchylenie standardowe negatywnych zwrotów
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
            
        # Annualizuj zwroty
        annualized_return = (1 + mean_return) ** trading_days - 1
        annualized_downside_std = downside_std * np.sqrt(trading_days)
        
        # Oblicz wskaźnik Sortino
        sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_std
        
        return sortino_ratio
        
    def get_calmar_ratio(self, trading_days: int = 252) -> float:
        """
        Oblicza wskaźnik Calmara (stosunek annualizowanego zwrotu do maksymalnego drawdownu).
        
        Args:
            trading_days: Liczba dni handlowych w roku (domyślnie 252)
            
        Returns:
            Wartość wskaźnika Calmara
        """
        if len(self.portfolio_history) < 2:
            return 0.0
            
        # Oblicz annualizowany zwrot
        values = [entry["value"] for entry in self.portfolio_history]
        total_return = values[-1] / values[0] - 1
        
        # Oblicz czas trwania w latach
        start_date = self.portfolio_history[0]["timestamp"]
        end_date = self.portfolio_history[-1]["timestamp"]
        days_diff = (end_date - start_date).days
        years = days_diff / 365.0
        
        if years == 0:
            years = 1/365.0  # co najmniej jeden dzień
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Oblicz maksymalny drawdown
        max_dd = self.get_max_drawdown() / 100  # konwersja z % na ułamek
        
        if max_dd == 0:
            return float('inf')  # Brak drawdownu
            
        calmar_ratio = annualized_return / max_dd
        
        return calmar_ratio
        
    def get_performance_stats(self) -> Dict:
        """
        Generuje pełne statystyki wydajności.
        
        Returns:
            Słownik zawierający metryki wydajności
        """
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade["profit_loss"] > 0)
        losing_trades = total_trades - winning_trades
        
        # Oblicz średni czas trwania transakcji
        durations = [trade.get("duration", 0) for trade in self.trades if "duration" in trade]
        avg_duration = np.mean(durations) if durations else 0
        
        # Oblicz średni zysk/stratę
        avg_profit = np.mean([trade["profit_loss"] for trade in self.trades if trade["profit_loss"] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([trade["profit_loss"] for trade in self.trades if trade["profit_loss"] < 0]) if losing_trades > 0 else 0
        
        # Oblicz wskaźnik zysku (stosunek średniego zysku do średniej straty)
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        
        stats = {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "profit_loss": self.get_profit_loss(),
            "profit_loss_percentage": self.get_profit_loss_percentage(),
            "max_drawdown": self.get_max_drawdown(),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "sortino_ratio": self.get_sortino_ratio(),
            "calmar_ratio": self.get_calmar_ratio(),
            "win_rate": self.get_win_rate(),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "avg_trade_duration_hours": avg_duration,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "trading_duration_days": (datetime.now() - self.start_time).days
        }
        
        return stats
        
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """
        Zwraca historię wartości portfela jako DataFrame.
        
        Returns:
            DataFrame z historią wartości portfela
        """
        df = pd.DataFrame(self.portfolio_history)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        return df
        
    def reset(self) -> None:
        """
        Resetuje monitor wydajności do początkowego stanu.
        """
        self.current_balance = self.initial_balance
        self.trades = []
        self.portfolio_history = []
        self.start_time = datetime.now()
        
        # Zapisz początkową wartość portfela
        self._record_portfolio_value(self.initial_balance)
        
        logger.info(f"Zresetowano PerformanceMonitor do początkowego salda {self.initial_balance}")
        
    def export_to_csv(self, trades_file: str = "trades.csv", portfolio_file: str = "portfolio.csv") -> Tuple[str, str]:
        """
        Eksportuje dane do plików CSV.
        
        Args:
            trades_file: Nazwa pliku dla historii transakcji
            portfolio_file: Nazwa pliku dla historii portfela
            
        Returns:
            Tuple z nazwami zapisanych plików
        """
        import os
        
        # Utwórz katalog dla raportów, jeśli nie istnieje
        os.makedirs("reports", exist_ok=True)
        
        # Pełne ścieżki plików
        trades_path = os.path.join("reports", trades_file)
        portfolio_path = os.path.join("reports", portfolio_file)
        
        # Zapisz historię transakcji
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(trades_path, index=False)
            
        # Zapisz historię portfela
        if self.portfolio_history:
            portfolio_df = self.get_portfolio_history_df()
            portfolio_df.to_csv(portfolio_path)
            
        logger.info(f"Wyeksportowano dane wydajności do plików: {trades_path}, {portfolio_path}")
        
        return trades_path, portfolio_path
        
# Globalna instancja monitora wydajności
performance_monitor = PerformanceMonitor()

# Funkcje pomocnicze do interakcji z globalną instancją
def add_trade(trade_data: Dict) -> None:
    """Dodaje transakcję do globalnego monitora wydajności."""
    performance_monitor.add_trade(trade_data)
    
def update_balance(new_balance: float) -> None:
    """Aktualizuje saldo w globalnym monitorze wydajności."""
    performance_monitor.update_balance(new_balance)
    
def get_performance_stats() -> Dict:
    """Pobiera statystyki wydajności z globalnego monitora."""
    return performance_monitor.get_performance_stats()
    
def reset_performance_monitor() -> None:
    """Resetuje globalny monitor wydajności."""
    performance_monitor.reset()