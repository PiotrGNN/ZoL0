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
--------------------------
Uproszczony silnik handlowy dla platformy tradingowej.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class SimplifiedTradingEngine:
    """
    Uproszczony silnik handlowy, który zarządza wykonywaniem strategii tradingowych.
    """
    
    def __init__(self, risk_manager=None, strategy_manager=None, exchange_connector=None):
        """
        Inicjalizacja silnika handlowego.
        
        Args:
            risk_manager: Manager ryzyka
            strategy_manager: Manager strategii
            exchange_connector: Konektor giełdowy
        """
        self.logger = logging.getLogger('trading_engine')
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.exchange_connector = exchange_connector
        
        self.running = False
        self.trading_thread = None
        self.stop_event = threading.Event()
        
        self.symbols = []
        self.last_status = {
            'running': False,
            'timestamp': datetime.now().isoformat(),
            'last_error': None,
            'active_symbols': [],
            'active_strategies': []
        }
        
        self.logger.info("Zainicjalizowano uproszczony silnik handlowy")
    
    def start_trading(self, symbols: List[str]) -> bool:
        """
        Uruchamia silnik handlowy dla podanych symboli.
        
        Args:
            symbols: Lista symboli do handlu
            
        Returns:
            bool: Czy uruchomienie się powiodło
        """
        if self.running:
            self.logger.warning("Silnik handlowy już działa")
            return False
        
        try:
            self.symbols = symbols
            self.stop_event.clear()
            self.running = True
            
            # Aktualizacja statusu
            self.last_status = {
                'running': True,
                'timestamp': datetime.now().isoformat(),
                'last_error': None,
                'active_symbols': symbols,
                'active_strategies': self._get_active_strategies()
            }
            
            # Uruchomienie wątku handlowego
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True  # Wątek zostanie zamknięty przy zamknięciu głównego programu
            self.trading_thread.start()
            
            self.logger.info(f"Silnik handlowy uruchomiony dla symboli: {symbols}")
            return True
        except Exception as e:
            self.running = False
            self.last_status['last_error'] = str(e)
            self.last_status['running'] = False
            self.logger.error(f"Błąd podczas uruchamiania silnika handlowego: {e}")
            return False
    
    def stop(self) -> Dict[str, Any]:
        """
        Zatrzymuje silnik handlowy.
        
        Returns:
            Dict[str, Any]: Status zatrzymania
        """
        if not self.running:
            return {'success': True, 'message': 'Silnik handlowy nie jest uruchomiony'}
        
        try:
            self.logger.info("Zatrzymywanie silnika handlowego...")
            self.stop_event.set()
            
            # Czekanie na zakończenie wątku
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5.0)
            
            self.running = False
            
            # Aktualizacja statusu
            self.last_status = {
                'running': False,
                'timestamp': datetime.now().isoformat(),
                'last_error': None,
                'active_symbols': [],
                'active_strategies': []
            }
            
            self.logger.info("Silnik handlowy zatrzymany")
            return {'success': True, 'message': 'Silnik handlowy zatrzymany'}
        except Exception as e:
            self.last_status['last_error'] = str(e)
            self.logger.error(f"Błąd podczas zatrzymywania silnika handlowego: {e}")
            return {'success': False, 'error': str(e)}
    
    def reset(self) -> Dict[str, Any]:
        """
        Resetuje silnik handlowy.
        
        Returns:
            Dict[str, Any]: Status resetowania
        """
        try:
            # Zatrzymaj silnik jeśli działa
            if self.running:
                self.stop()
            
            # Resetowanie stanu
            self.symbols = []
            self.last_status = {
                'running': False,
                'timestamp': datetime.now().isoformat(),
                'last_error': None,
                'active_symbols': [],
                'active_strategies': []
            }
            
            self.logger.info("Silnik handlowy zresetowany")
            return {'success': True, 'message': 'Silnik handlowy zresetowany'}
        except Exception as e:
            self.last_status['last_error'] = str(e)
            self.logger.error(f"Błąd podczas resetowania silnika handlowego: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Zwraca aktualny status silnika handlowego.
        
        Returns:
            Dict[str, Any]: Status silnika
        """
        self.last_status['timestamp'] = datetime.now().isoformat()
        return self.last_status
    
    def _trading_loop(self):
        """Główna pętla handlowa."""
        self.logger.info("Rozpoczęcie pętli handlowej")
        
        while not self.stop_event.is_set():
            try:
                # Wykonaj cykl handlowy
                for symbol in self.symbols:
                    if self.stop_event.is_set():
                        break
                    
                    # Pobierz dane rynkowe
                    market_data = self._get_market_data(symbol)
                    
                    # Zastosuj strategie
                    signals = self._apply_strategies(symbol, market_data)
                    
                    # Zastosuj zarządzanie ryzykiem
                    filtered_signals = self._apply_risk_management(signals, symbol, market_data)
                    
                    # Wykonaj sygnały
                    self._execute_signals(filtered_signals, symbol, market_data)
                
                # Poczekaj przed następnym cyklem
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Błąd w pętli handlowej: {e}")
                self.last_status['last_error'] = str(e)
                time.sleep(10)  # Dłuższe czekanie po błędzie
        
        self.logger.info("Zakończenie pętli handlowej")
    
    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Pobiera dane rynkowe dla symbolu.
        
        Args:
            symbol: Symbol do pobrania danych
            
        Returns:
            Dict[str, Any]: Dane rynkowe
        """
        if not self.exchange_connector:
            return {'symbol': symbol, 'timestamp': datetime.now().isoformat()}
        
        try:
            # Pobierz aktualny kurs
            ticker = self.exchange_connector.get_ticker(symbol)
            
            # Pobierz świece (opcjonalnie)
            candles = []
            try:
                candles = self.exchange_connector.get_klines(symbol=symbol, interval="15m", limit=20)
            except:
                pass
            
            # Dodatkowe dane (opcjonalnie)
            orderbook = {}
            try:
                orderbook = self.exchange_connector.get_order_book(symbol=symbol, limit=5)
            except:
                pass
            
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'candles': candles,
                'orderbook': orderbook
            }
            
            return market_data
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania danych rynkowych dla {symbol}: {e}")
            return {'symbol': symbol, 'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def _apply_strategies(self, symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Stosuje strategie do danych rynkowych.
        
        Args:
            symbol: Symbol
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Lista sygnałów
        """
        signals = []
        
        if not self.strategy_manager:
            return signals
        
        try:
            # Pobierz aktywne strategie
            active_strategies = self.strategy_manager.get_active_strategies()
            
            # Zastosuj każdą strategię
            for strategy_name, strategy in active_strategies.items():
                try:
                    strategy_signals = strategy.generate_signals(market_data)
                    
                    # Dodaj informację o strategii
                    for signal in strategy_signals:
                        signal['strategy'] = strategy_name
                        signals.append(signal)
                except Exception as strategy_error:
                    self.logger.error(f"Błąd w strategii {strategy_name}: {strategy_error}")
        except Exception as e:
            self.logger.error(f"Błąd podczas stosowania strategii: {e}")
        
        return signals
    
    def _apply_risk_management(self, signals: List[Dict[str, Any]], symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Stosuje zarządzanie ryzykiem do sygnałów.
        
        Args:
            signals: Lista sygnałów
            symbol: Symbol
            market_data: Dane rynkowe
            
        Returns:
            List[Dict[str, Any]]: Przefiltrowane sygnały
        """
        if not self.risk_manager or not signals:
            return signals
        
        try:
            return self.risk_manager.filter_signals(signals, symbol, market_data)
        except Exception as e:
            self.logger.error(f"Błąd podczas zarządzania ryzykiem: {e}")
            return []
    
    def _execute_signals(self, signals: List[Dict[str, Any]], symbol: str, market_data: Dict[str, Any]):
        """
        Wykonuje sygnały handlowe.
        
        Args:
            signals: Lista sygnałów
            symbol: Symbol
            market_data: Dane rynkowe
        """
        if not signals or not self.exchange_connector:
            return
        
        for signal in signals:
            try:
                # Wykonaj zlecenie
                order_type = signal.get('type', 'LIMIT')
                side = signal.get('side', 'BUY')
                quantity = signal.get('quantity', 0.0)
                price = signal.get('price')
                
                if quantity <= 0:
                    continue
                
                # Logowanie zamiast rzeczywistego wykonania (symulacja)
                self.logger.info(f"Symulacja zlecenia: {side} {quantity} {symbol} @ {price} ({order_type})")
                
                # W rzeczywistej implementacji:
                # result = self.exchange_connector.place_order(
                #     symbol=symbol,
                #     side=side,
                #     order_type=order_type,
                #     quantity=quantity,
                #     price=price
                # )
            except Exception as e:
                self.logger.error(f"Błąd podczas wykonywania sygnału: {e}")
    
    def _get_active_strategies(self) -> List[str]:
        """
        Pobiera listę aktywnych strategii.
        
        Returns:
            List[str]: Nazwy aktywnych strategii
        """
        if not self.strategy_manager:
            return []
        
        try:
            strategies = self.strategy_manager.get_active_strategies()
            return list(strategies.keys())
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania aktywnych strategii: {e}")
            return []
    
    def start(self) -> Dict[str, Any]:
        """
        Rozpoczyna trading dla zapisanych symboli.
        
        Returns:
            Dict[str, Any]: Status uruchomienia
        """
        if not self.symbols:
            self.symbols = ["BTCUSDT"]  # Domyślny symbol
        
        success = self.start_trading(self.symbols)
        
        if success:
            return {'success': True, 'message': f'Trading uruchomiony dla {self.symbols}'}
        else:
            return {'success': False, 'error': self.last_status.get('last_error', 'Nieznany błąd')}
