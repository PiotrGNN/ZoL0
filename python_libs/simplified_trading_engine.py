"""
simplified_trading_engine.py
----------------------------
Uproszczony silnik handlowy kompatybilny zarówno z lokalnym środowiskiem, jak i Replit.
"""

import logging
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimplifiedTradingEngine:
    """
    Uproszczony silnik handlowy dla systemu tradingowego.
    """

    def __init__(self, risk_manager=None, strategy_manager=None, exchange_connector=None):
        """
        Inicjalizacja silnika handlowego.

        Args:
            risk_manager: Manager ryzyka
            strategy_manager: Manager strategii
            exchange_connector: Konektor do giełdy
        """
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.exchange_connector = exchange_connector
        self.is_running = False
        self.active_symbols = []
        self.last_error = None
        self.positions = {}
        self.orders = {}
        self.simulated_mode = self.exchange_connector is None
        if self.simulated_mode:
            logger.warning("Konektor giełdy nie jest ustawiony - działanie w trybie symulacji!")
        else:
            logger.info(f"Zainicjalizowano SimplifiedTradingEngine z konektorem {type(self.exchange_connector).__name__}")

    def start_trading(self, symbols: List[str]) -> bool:
        """
        Uruchamia handel dla podanych symboli.

        Args:
            symbols: Lista symboli do handlu

        Returns:
            bool: True jeśli uruchomienie się powiodło, False w przeciwnym razie
        """
        try:
            if not self.exchange_connector:
                self.last_error = "Brak konektora giełdy"
                logger.error(self.last_error)
                return False

            if not self.strategy_manager:
                self.last_error = "Brak managera strategii"
                logger.error(self.last_error)
                return False

            if not self.risk_manager:
                self.last_error = "Brak managera ryzyka"
                logger.error(self.last_error)
                return False

            self.active_symbols = symbols
            self.is_running = True
            logger.info(f"Uruchomiono handel dla symboli: {symbols}")

            # Symulacja pobierania kapitału
            if hasattr(self.exchange_connector, 'get_account_balance'):
                try:
                    balance = self.exchange_connector.get_account_balance()
                    if 'balances' in balance and 'USDT' in balance['balances']:
                        capital = balance['balances']['USDT'].get('equity', 1000.0)
                        if self.risk_manager:
                            self.risk_manager.set_capital(capital)
                            logger.info(f"Ustawiono kapitał: ${capital:.2f}")
                except Exception as e:
                    logger.warning(f"Nie udało się pobrać kapitału: {e}")

            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Błąd podczas uruchamiania handlu: {e}")
            return False

    def stop(self) -> Dict[str, Any]:
        """
        Zatrzymuje handel.

        Returns:
            Dict: Status zatrzymania
        """
        if not self.is_running:
            return {"success": True, "message": "Handel już jest zatrzymany"}

        self.is_running = False
        logger.info("Zatrzymano handel")
        return {"success": True, "message": "Handel zatrzymany pomyślnie"}

    def start(self) -> Dict[str, Any]:
        """
        Uruchamia handel dla wcześniej ustawionych symboli.

        Returns:
            Dict: Status uruchomienia
        """
        if self.is_running:
            return {"success": True, "message": "Handel już jest uruchomiony"}

        if not self.active_symbols:
            return {"success": False, "error": "Brak aktywnych symboli"}

        success = self.start_trading(self.active_symbols)
        if success:
            return {"success": True, "message": f"Handel uruchomiony dla symboli: {self.active_symbols}"}
        else:
            return {"success": False, "error": self.last_error or "Nieznany błąd"}

    def reset(self) -> Dict[str, Any]:
        """
        Resetuje silnik handlowy.

        Returns:
            Dict: Status resetu
        """
        was_running = self.is_running
        if was_running:
            self.stop()

        self.last_error = None
        self.positions = {}
        self.orders = {}

        logger.info("Zresetowano silnik handlowy")

        if was_running:
            success = self.start_trading(self.active_symbols)
            if success:
                return {"success": True, "message": "Silnik handlowy zresetowany i uruchomiony ponownie"}
            else:
                return {"success": False, "error": self.last_error or "Nie udało się uruchomić handlu po resecie"}
        else:
            return {"success": True, "message": "Silnik handlowy zresetowany"}

    def get_status(self) -> Dict[str, Any]:
        """
        Zwraca status silnika handlowego.

        Returns:
            Dict: Status silnika
        """
        return {
            "status": "running" if self.is_running else "stopped",
            "active_symbols": self.active_symbols,
            "active_strategies": self.strategy_manager.get_active_strategies() if self.strategy_manager else [],
            "positions_count": len(self.positions),
            "orders_count": len(self.orders),
            "last_error": self.last_error
        }

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Pobiera dane rynkowe dla podanego symbolu.

        Args:
            symbol: Symbol do pobrania danych

        Returns:
            Dict: Dane rynkowe
        """
        if self.exchange_connector is None:
            logger.warning("[SIMULATION] Pobrano dane " + symbol + " z lokalnego źródła")
            # Symulowane dane
            import random
            price = 50000 + random.uniform(-1000, 1000)
            return {
                "symbol": symbol,
                "price": price,
                "volume": random.uniform(10, 100),
                "timestamp": datetime.now().isoformat(),
                "simulated": True
            }

        try:
            # Próba pobrania danych z konektora giełdy
            if hasattr(self.exchange_connector, 'get_ticker'):
                ticker = self.exchange_connector.get_ticker(symbol)
                return ticker
            elif hasattr(self.exchange_connector, 'get_klines'):
                klines = self.exchange_connector.get_klines(symbol=symbol, limit=1)
                if klines and len(klines) > 0:
                    latest = klines[0]
                    return {
                        "symbol": symbol,
                        "price": latest.get("close", 0),
                        "volume": latest.get("volume", 0),
                        "timestamp": latest.get("datetime", datetime.now().isoformat()),
                        "simulated": False
                    }
            
            # Fallback do symulowanych danych
            logger.warning(f"Konektor giełdy nie ma odpowiedniej metody do pobrania danych dla {symbol}, używam symulacji")
            import random
            price = 50000 + random.uniform(-1000, 1000)
            return {
                "symbol": symbol,
                "price": price,
                "volume": random.uniform(10, 100),
                "timestamp": datetime.now().isoformat(),
                "simulated": True
            }
        except Exception as e:
            logger.error(f"Błąd podczas pobierania danych rynkowych: {e}")
            return {"error": str(e), "simulated": True}

    def calculate_positions_risk(self) -> Dict[str, float]:
        """
        Oblicza ryzyko dla aktywnych pozycji.

        Returns:
            Dict: Poziomy ryzyka dla aktywnych pozycji
        """
        risk_levels = {}

        if not self.positions:
            return risk_levels

        for symbol, position in self.positions.items():
            try:
                # Przykładowa logika obliczania ryzyka
                entry_price = position.get('entry_price', 0)
                size = position.get('size', 0)
                side = position.get('side', 'NONE')

                market_data = self.get_market_data(symbol)
                current_price = market_data.get('price', entry_price)

                if entry_price <= 0 or size <= 0:
                    risk_levels[symbol] = 0
                    continue

                # Obliczenie P&L
                if side == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Obliczenie ryzyka (przykładowo, w praktyce byłaby bardziej złożona logika)
                if pnl_pct < -0.05:  # Strata > 5%
                    risk_levels[symbol] = 0.8  # Wysokie ryzyko
                elif pnl_pct < -0.02:  # Strata > 2%
                    risk_levels[symbol] = 0.5  # Średnie ryzyko
                elif pnl_pct < 0:  # Jakakolwiek strata
                    risk_levels[symbol] = 0.3  # Niskie ryzyko
                else:
                    risk_levels[symbol] = 0.1  # Bardzo niskie ryzyko
            except Exception as e:
                logger.error(f"Błąd podczas obliczania ryzyka dla {symbol}: {e}")
                risk_levels[symbol] = 0

        return risk_levels

# Przykład użycia
def __bool__(self):
        """Gwarantuje, że instancja klasy zawsze zwraca True w kontekście logicznym."""
        return True


if __name__ == "__main__":
    from .simplified_risk_manager import SimplifiedRiskManager
    from .simplified_strategy import StrategyManager

    # Przykładowa inicjalizacja komponentów
    risk_manager = SimplifiedRiskManager(max_risk=0.02, max_position_size=0.2, max_drawdown=0.1)

    strategies = {
        "trend_following": {"name": "Trend Following", "enabled": True},
        "mean_reversion": {"name": "Mean Reversion", "enabled": False},
        "breakout": {"name": "Breakout", "enabled": True}
    }

    exposure_limits = {
        "trend_following": 0.5,
        "mean_reversion": 0.3,
        "breakout": 0.4
    }

    strategy_manager = StrategyManager(strategies, exposure_limits)
    strategy_manager.activate_strategy("trend_following")

    # Inicjalizacja silnika handlowego
    trading_engine = SimplifiedTradingEngine(
        risk_manager=risk_manager,
        strategy_manager=strategy_manager,
        exchange_connector=None  # W rzeczywistym użyciu byłby tutaj konektor do giełdy
    )

    # Uruchomienie handlu
    trading_engine.start_trading(["BTCUSDT", "ETHUSDT"])

    # Sprawdzenie statusu
    status = trading_engine.get_status()
    print(f"Status silnika: {status}")

    # Symulacja danych rynkowych
    market_data = trading_engine.get_market_data("BTCUSDT")
    print(f"Dane rynkowe: {market_data}")

    # Zatrzymanie handlu
    stop_result = trading_engine.stop()
    print(f"Wynik zatrzymania: {stop_result}")