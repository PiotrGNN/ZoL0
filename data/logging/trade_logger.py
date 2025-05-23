"""
trade_logger.py
--------------
Moduł do logowania transakcji i zdarzeń tradingowych.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Konfiguracja logowania
logger = logging.getLogger("trade_logger")
if not logger.handlers:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "trade_logger.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


class TradeLogger:
    """
    Klasa do logowania transakcji i zdarzeń tradingowych.
    """

    def __init__(self, log_file: str = None):
        """
        Inicjalizuje logger transakcji.

        Parameters:
            log_file (str, optional): Ścieżka do pliku logu.
        """
        self.trades = []
        self.log_file = log_file or os.path.join("logs", "trades.log")
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Upewnia się, że katalog logów istnieje."""
        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Loguje transakcję.

        Parameters:
            trade_data (Dict[str, Any]): Dane transakcji.
        """
        timestamp = trade_data.get("timestamp", int(time.time() * 1000))
        trade_with_time = {
            **trade_data,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        }

        # Logujemy do pliku
        with open(self.log_file, "a") as f:
            f.write(json.dumps(trade_with_time) + "\n")

        # Dodajemy do pamięci
        self.trades.append(trade_with_time)
        logger.info(f"Zalogowano transakcję: {trade_with_time}")

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Pobiera ostatnie transakcje.

        Parameters:
            limit (int): Maksymalna liczba transakcji do pobrania.

        Returns:
            List[Dict[str, Any]]: Lista ostatnich transakcji.
        """
        return self.trades[-limit:]

    def get_trades_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Pobiera transakcje dla danego symbolu.

        Parameters:
            symbol (str): Symbol pary handlowej.

        Returns:
            List[Dict[str, Any]]: Lista transakcji dla danego symbolu.
        """
        return [trade for trade in self.trades if trade.get("symbol") == symbol]

    def get_trades_statistics(self) -> Dict[str, Any]:
        """
        Pobiera statystyki transakcji.

        Returns:
            Dict[str, Any]: Statystyki transakcji.
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_loss": 0,
                "avg_profit": 0,
                "max_profit": 0,
                "max_loss": 0
            }

        wins = sum(1 for trade in self.trades if trade.get("profit", 0) > 0)
        losses = sum(1 for trade in self.trades if trade.get("profit", 0) < 0)
        total_trades = len(self.trades)

        profits = [trade.get("profit", 0) for trade in self.trades]
        total_profit = sum(profits)

        return {
            "total_trades": total_trades,
            "win_rate": (wins / total_trades) * 100 if total_trades > 0 else 0,
            "profit_loss": total_profit,
            "avg_profit": total_profit / total_trades if total_trades > 0 else 0,
            "max_profit": max(profits) if profits else 0,
            "max_loss": min(profits) if profits else 0
        }

    def has_more_trades(self, current_index: int) -> bool:
        """
        Sprawdza czy są kolejne transakcje do przetworzenia.

        Parameters:
            current_index (int): Aktualny indeks w liście transakcji.

        Returns:
            bool: True jeśli są kolejne transakcje, False w przeciwnym razie.
        """
        return current_index < len(self.trades) - 1
    
    def get_next_trade(self, current_index: int) -> tuple[Dict[str, Any], int]:
        """
        Pobiera następną transakcję z listy.

        Parameters:
            current_index (int): Aktualny indeks w liście transakcji.

        Returns:
            tuple[Dict[str, Any], int]: Krotka zawierająca następną transakcję i nowy indeks.
            Jeśli nie ma więcej transakcji, zwraca None i ten sam indeks.
        """
        if self.has_more_trades(current_index):
            next_index = current_index + 1
            return self.trades[next_index], next_index
        return None, current_index


# Singleton instancja dla łatwego dostępu z różnych modułów
trade_logger = TradeLogger()


def log_trade(trade_data: Dict[str, Any]) -> None:
    """
    Funkcja pomocnicza do logowania transakcji.

    Parameters:
        trade_data (Dict[str, Any]): Dane transakcji.
    """
    trade_logger.log_trade(trade_data)