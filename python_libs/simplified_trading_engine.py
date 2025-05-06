"""
simplified_trading_engine.py
---------------------------
Uproszczony silnik handlowy dla platformy tradingowej.
"""

import logging
import time
import random
import os
import json
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class SimplifiedTradingEngine:
    """Uproszczony silnik handlowy do wykonywania transakcji."""

    def __init__(self, risk_manager=None, strategy_manager=None, exchange_connector=None):
        """
        Inicjalizuje silnik handlowy.

        Parameters:
            risk_manager: Menedżer ryzyka
            strategy_manager: Menedżer strategii
            exchange_connector: Konektor giełdy
        """
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.exchange_connector = exchange_connector

        self.status = {
            "running": False,
            "active_symbols": [],
            "last_trade_time": None,
            "last_error": None,
            "trade_count": 0
        }

        self.settings = {
            "trade_interval": 60,  # Interwał handlu w sekundach
            "max_orders_per_symbol": 5,
            "enable_auto_trading": False
        }

        self.orders = []
        self.positions = []

        logger.info("Zainicjalizowano uproszczony silnik handlowy (SimplifiedTradingEngine)")

    def start_trading(self, symbols: List[str]) -> bool:
        """
        Uruchamia handel na określonych symbolach.

        Parameters:
            symbols (List[str]): Lista symboli do handlu

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            if not symbols:
                self.status["last_error"] = "Brak określonych symboli do handlu"
                logger.error(self.status["last_error"])
                return False

            self.status["active_symbols"] = symbols
            self.status["running"] = True
            self.status["last_error"] = None

            logger.info(f"Uruchomiono handel na symbolach: {symbols}")
            return True
        except Exception as e:
            self.status["last_error"] = f"Błąd podczas uruchamiania handlu: {str(e)}"
            logger.error(self.status["last_error"])
            return False

    def stop_trading(self) -> bool:
        """
        Zatrzymuje handel.

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            self.status["running"] = False
            logger.info("Zatrzymano handel")
            return True
        except Exception as e:
            self.status["last_error"] = f"Błąd podczas zatrzymywania handlu: {str(e)}"
            logger.error(self.status["last_error"])
            return False

    def create_order(self, symbol: str, order_type: str, side: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Tworzy zlecenie handlowe.

        Parameters:
            symbol (str): Symbol instrumentu
            order_type (str): Typ zlecenia ('market', 'limit')
            side (str): Strona zlecenia ('buy', 'sell')
            quantity (float): Ilość instrumentu
            price (Optional[float]): Cena dla zleceń limit

        Returns:
            Dict[str, Any]: Informacje o zleceniu
        """
        try:
            # Sprawdź, czy handel jest uruchomiony
            if not self.status["running"]:
                return {"success": False, "error": "Handel nie jest uruchomiony"}

            # Sprawdź, czy symbol jest w aktywnych symbolach
            if symbol not in self.status["active_symbols"]:
                return {"success": False, "error": f"Symbol {symbol} nie jest aktywny"}

            # Sprawdź limity zleceń
            symbol_orders = [o for o in self.orders if o["symbol"] == symbol and o["status"] == "open"]
            if len(symbol_orders) >= self.settings["max_orders_per_symbol"]:
                return {"success": False, "error": f"Osiągnięto maksymalną liczbę zleceń dla symbolu {symbol}"}

            # Jeśli jest menedżer ryzyka, sprawdź limity ryzyka
            if self.risk_manager:
                risk_check = self.risk_manager.check_trade_risk(symbol, side, quantity, price)
                if not risk_check["success"]:
                    return {"success": False, "error": risk_check["error"]}

            # Przygotuj zlecenie
            order_id = f"order_{int(time.time())}_{random.randint(1000, 9999)}"

            order = {
                "id": order_id,
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "quantity": quantity,
                "price": price if order_type == "limit" else None,
                "status": "open",
                "filled": 0.0,
                "timestamp": time.time()
            }

            # Dodaj zlecenie do listy
            self.orders.append(order)

            # Jeśli jest konektor giełdy, wyślij zlecenie
            if self.exchange_connector:
                exchange_order = self.exchange_connector.create_order(
                    symbol=symbol,
                    order_type=order_type,
                    side=side,
                    quantity=quantity,
                    price=price
                )

                if exchange_order.get("success"):
                    # Aktualizuj zlecenie z informacjami z giełdy
                    order["exchange_id"] = exchange_order.get("order_id")
                    order["exchange_status"] = exchange_order.get("status")

                    logger.info(f"Utworzono zlecenie na giełdzie: {order_id}")
                else:
                    # Jeśli zlecenie nie zostało utworzone na giełdzie, oznacz je jako anulowane
                    order["status"] = "cancelled"
                    order["error"] = exchange_order.get("error")

                    logger.error(f"Błąd podczas tworzenia zlecenia na giełdzie: {exchange_order.get('error')}")
                    return {"success": False, "error": exchange_order.get("error")}
            else:
                # Symulacja wykonania zlecenia
                if order_type == "market":
                    # Symuluj natychmiastowe wykonanie zlecenia rynkowego
                    order["status"] = "filled"
                    order["filled"] = quantity

                    # Dodaj pozycję
                    position = {
                        "id": f"position_{int(time.time())}_{random.randint(1000, 9999)}",
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": price or random.uniform(30000, 40000),  # Symulowana cena
                        "current_price": price or random.uniform(30000, 40000),
                        "timestamp": time.time()
                    }

                    self.positions.append(position)

                    logger.info(f"Zasymulowano wykonanie zlecenia rynkowego: {order_id}")
                else:
                    # Symuluj częściowe wykonanie zlecenia limit
                    order["filled"] = random.uniform(0, quantity)

                    logger.info(f"Zasymulowano częściowe wykonanie zlecenia limit: {order_id}")

            # Aktualizuj statystyki
            self.status["last_trade_time"] = time.time()
            self.status["trade_count"] += 1

            return {"success": True, "order": order}
        except Exception as e:
            error_msg = f"Błąd podczas tworzenia zlecenia: {str(e)}"
            self.status["last_error"] = error_msg
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Anuluje zlecenie.

        Parameters:
            order_id (str): ID zlecenia

        Returns:
            Dict[str, Any]: Wynik operacji
        """
        try:
            # Znajdź zlecenie
            order = next((o for o in self.orders if o["id"] == order_id), None)

            if not order:
                return {"success": False, "error": f"Nie znaleziono zlecenia o ID {order_id}"}

            if order["status"] != "open":
                return {"success": False, "error": f"Zlecenie o ID {order_id} nie jest otwarte"}

            # Jeśli jest konektor giełdy i zlecenie ma ID na giełdzie, anuluj je
            if self.exchange_connector and "exchange_id" in order:
                exchange_result = self.exchange_connector.cancel_order(
                    symbol=order["symbol"],
                    order_id=order["exchange_id"]
                )

                if exchange_result.get("success"):
                    order["status"] = "cancelled"
                    logger.info(f"Anulowano zlecenie na giełdzie: {order_id}")
                else:
                    logger.error(f"Błąd podczas anulowania zlecenia na giełdzie: {exchange_result.get('error')}")
                    return {"success": False, "error": exchange_result.get("error")}
            else:
                # Symuluj anulowanie zlecenia
                order["status"] = "cancelled"
                logger.info(f"Zasymulowano anulowanie zlecenia: {order_id}")

            return {"success": True, "order": order}
        except Exception as e:
            error_msg = f"Błąd podczas anulowania zlecenia: {str(e)}"
            self.status["last_error"] = error_msg
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Zwraca listę zleceń.

        Parameters:
            symbol (Optional[str]): Filtrowanie po symbolu
            status (Optional[str]): Filtrowanie po statusie

        Returns:
            List[Dict[str, Any]]: Lista zleceń
        """
        filtered_orders = self.orders

        if symbol:
            filtered_orders = [o for o in filtered_orders if o["symbol"] == symbol]

        if status:
            filtered_orders = [o for o in filtered_orders if o["status"] == status]

        return filtered_orders

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Zwraca listę pozycji.

        Parameters:
            symbol (Optional[str]): Filtrowanie po symbolu

        Returns:
            List[Dict[str, Any]]: Lista pozycji
        """
        if symbol:
            return [p for p in self.positions if p["symbol"] == symbol]
        return self.positions

    def get_status(self) -> Dict[str, Any]:
        """
        Zwraca status silnika handlowego.

        Returns:
            Dict[str, Any]: Status silnika handlowego
        """
        return self.status

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Aktualizuje ustawienia silnika handlowego.

        Parameters:
            settings (Dict[str, Any]): Nowe ustawienia

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            for key, value in settings.items():
                if key in self.settings:
                    self.settings[key] = value

            logger.info(f"Zaktualizowano ustawienia silnika handlowego: {settings}")
            return True
        except Exception as e:
            self.status["last_error"] = f"Błąd podczas aktualizacji ustawień: {str(e)}"
            logger.error(self.status["last_error"])
            return False

    def start(self) -> Dict[str, Any]:
        """
        Uruchamia silnik handlowy (alias dla start_trading).

        Returns:
            Dict[str, Any]: Wynik operacji
        """
        symbols = self.status["active_symbols"] or ["BTCUSDT"]
        logger.info(f"Uruchamianie silnika handlowego dla symboli: {symbols}")
        success = self.start_trading(symbols)
        
        if success:
            logger.info("Silnik handlowy uruchomiony pomyślnie")
            self.settings["enable_auto_trading"] = True
        else:
            logger.error(f"Nie udało się uruchomić silnika handlowego: {self.status.get('last_error', 'Nieznany błąd')}")
            
        return {"success": success, "status": self.get_status()}

    def stop(self) -> Dict[str, Any]:
        """
        Zatrzymuje silnik handlowy (alias dla stop_trading).

        Returns:
            Dict[str, Any]: Wynik operacji
        """
        logger.info("Zatrzymywanie silnika handlowego...")
        success = self.stop_trading()
        
        if success:
            logger.info("Silnik handlowy zatrzymany pomyślnie")
            self.settings["enable_auto_trading"] = False
        else:
            logger.error(f"Nie udało się zatrzymać silnika handlowego: {self.status.get('last_error', 'Nieznany błąd')}")
            
        return {"success": success, "status": self.get_status()}

    def reset(self) -> Dict[str, Any]:
        """
        Resetuje silnik handlowy.

        Returns:
            Dict[str, Any]: Wynik operacji
        """
        try:
            self.stop_trading()
            self.orders = []
            self.positions = []
            self.status["trade_count"] = 0
            self.status["last_error"] = None

            logger.info("Zresetowano silnik handlowy")
            return {"success": True, "status": self.get_status()}
        except Exception as e:
            self.status["last_error"] = f"Błąd podczas resetowania silnika handlowego: {str(e)}"
            logger.error(self.status["last_error"])
            return {"success": False, "error": self.status["last_error"]}

    def _get_real_market_data(self):
        """Pobiera rzeczywiste dane rynkowe z API."""
        try:
            from data.data.market_data_fetcher import MarketDataFetcher

            # Pobierz klucz API z .env lub konfiguracji
            api_key = os.getenv("BYBIT_API_KEY", "")
            api_secret = os.getenv("BYBIT_API_SECRET", "")

            if not api_key or not api_secret:
                logging.warning("Brak kluczy API do pobierania rzeczywistych danych. Używam zapisanych danych.")
                return self._get_cached_data()

            # Inicjalizacja fetchera danych
            fetcher = MarketDataFetcher(api_key=api_key)

            # Pobierz dane dla pary BTC/USDT w interwale 15m
            df = fetcher.fetch_data(symbol="BTCUSDT", interval="15m", limit=100)

            # Konwersja na listę cen zamknięcia
            if df is not None and not df.empty and 'close' in df.columns:
                price_data = df['close'].tolist()
                logging.info(f"Pobrano {len(price_data)} punktów danych rzeczywistych")

                # Zapisz dane do cache
                self._cache_data(price_data)

                return price_data
            else:
                logging.warning("Pobrane dane są puste lub nieprawidłowe")
                return self._get_cached_data()

        except Exception as e:
            logging.error(f"Błąd podczas pobierania rzeczywistych danych: {e}")
            return self._get_cached_data()

    def _get_cached_data(self):
        """Pobiera dane z cache lub generuje nowe jeśli cache jest pusty."""
        cache_file = "data/cache/market_data_cache.json"

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logging.info(f"Używam danych z cache ({len(data)} punktów)")
                return data
            except Exception as e:
                logging.error(f"Błąd odczytu cache: {e}")

        # Fallback do generowania danych
        return self._generate_mock_data()

    def _cache_data(self, data):
        """Zapisuje dane do cache."""
        cache_file = "data/cache/market_data_cache.json"
        cache_dir = os.path.dirname(cache_file)

        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logging.info(f"Zapisano {len(data)} punktów danych do cache")
        except Exception as e:
            logging.error(f"Błąd zapisu cache: {e}")

    def _generate_mock_data(self):
        """Generuje symulowane dane rynkowe jako ostateczny fallback."""
        logging.warning("Używam symulowanych danych jako ostateczność!")
        start_price = 100.0
        price_data = []

        # Generowanie losowych cen z trendem
        current_price = start_price
        for _ in range(100):
            change = random.uniform(-2, 2)
            # Dodajemy trend
            if _ < 50:
                change += 0.1  # trend wzrostowy w pierwszej połowie
            else:
                change -= 0.1  # trend spadkowy w drugiej połowie

            current_price += change
            current_price = max(current_price, 50)  # Zapobieganie ujemnym cenom

            price_data.append(current_price)

        return price_data
"""
simplified_trading_engine.py
------------------------
Simplified trading engine implementation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class SimplifiedTradingEngine:
    def __init__(self, risk_manager, strategy_manager, exchange_connector):
        """Initialize trading engine."""
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.exchange = exchange_connector
        self.active = False
        self.trading_pairs = []
        self.status = {
            "active": False,
            "error": None,
            "last_update": None,
            "current_trades": 0,
            "total_trades": 0
        }
        logging.info("Initialized SimplifiedTradingEngine")

    def start_trading(self, symbols: List[str]) -> bool:
        """Start trading engine."""
        try:
            if self.active:
                logging.warning("Trading engine is already active")
                return True

            self.trading_pairs = symbols
            self.active = True
            self.status.update({
                "active": True,
                "error": None,
                "last_update": datetime.now().isoformat()
            })
            logging.info(f"Trading engine started with symbols: {symbols}")
            return True
        except Exception as e:
            logging.error(f"Error starting trading engine: {e}")
            self.status.update({
                "active": False,
                "error": str(e),
                "last_update": datetime.now().isoformat()
            })
            return False

    def stop(self) -> Dict[str, Any]:
        """Stop trading engine."""
        try:
            if not self.active:
                return {"success": True, "message": "Trading engine already stopped"}

            # Close all open positions
            for symbol in self.trading_pairs:
                position = self.exchange.get_position(symbol)
                if position.get("success") and position["position"]["size"] != 0:
                    self.close_position(symbol)

            self.active = False
            self.status.update({
                "active": False,
                "error": None,
                "last_update": datetime.now().isoformat()
            })
            logging.info("Trading engine stopped")
            return {"success": True, "message": "Trading engine stopped"}
        except Exception as e:
            logging.error(f"Error stopping trading engine: {e}")
            return {"success": False, "error": str(e)}

    def reset(self) -> bool:
        """Reset trading engine state."""
        try:
            self.stop()
            self.trading_pairs = []
            self.status = {
                "active": False,
                "error": None,
                "last_update": datetime.now().isoformat(),
                "current_trades": 0,
                "total_trades": 0
            }
            logging.info("Trading engine reset")
            return True
        except Exception as e:
            logging.error(f"Error resetting trading engine: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current trading engine status."""
        try:
            active_strategies = self.strategy_manager.get_active_strategies()
            risk_limits = self.risk_manager.get_risk_limits()
            
            status = {
                **self.status,
                "active_strategies": active_strategies,
                "risk_limits": risk_limits,
                "trading_pairs": self.trading_pairs,
                "last_update": datetime.now().isoformat()
            }
            return status
        except Exception as e:
            logging.error(f"Error getting trading engine status: {e}")
            return {
                "error": str(e),
                "active": False,
                "last_update": datetime.now().isoformat()
            }

    def create_order(self, symbol: str, side: str, order_type: str, qty: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Create new order with risk management."""
        try:
            if not self.active:
                return {"success": False, "error": "Trading engine is not active"}

            # Check risk parameters
            current_price = price or float(self.exchange.get_klines(symbol, limit=1)[0]["close"])
            risk_check = self.risk_manager.check_risk_level(qty, current_price * 0.95, current_price)

            if not risk_check["allowed"]:
                return {"success": False, "error": risk_check["reason"]}

            # Create the order
            order = self.exchange.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                qty=qty,
                price=price
            )

            if order["success"]:
                self.status["current_trades"] += 1
                self.status["total_trades"] += 1
                self.status["last_update"] = datetime.now().isoformat()

            return order
        except Exception as e:
            logging.error(f"Error creating order: {e}")
            return {"success": False, "error": str(e)}

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close position for a symbol."""
        try:
            position = self.exchange.get_position(symbol)
            if not position["success"]:
                return position

            if position["position"]["size"] == 0:
                return {"success": True, "message": "No position to close"}

            side = "SELL" if position["position"]["size"] > 0 else "BUY"
            order = self.create_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                qty=abs(position["position"]["size"])
            )

            if order["success"]:
                logging.info(f"Closed position for {symbol}")
            return order
        except Exception as e:
            logging.error(f"Error closing position: {e}")
            return {"success": False, "error": str(e)}
