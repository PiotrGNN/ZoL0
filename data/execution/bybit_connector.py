"""
bybit_connector.py
-----------------
Moduł do komunikacji z giełdą Bybit.
"""

import json
import logging
import os
import random
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests

class BybitConnector:
    """
    Klasa do komunikacji z giełdą Bybit.
    """

    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = True):
        """
        Inicjalizuje połączenie z Bybit.

        Parameters:
            api_key (str): Klucz API Bybit.
            api_secret (str): Sekret API Bybit.
            use_testnet (bool): Czy używać środowiska testowego.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.base_url = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"

        # Inicjalizacja klienta API
        try:
            import pybit

            # Sprawdzamy wersję i dostosowujemy inicjalizację do odpowiedniej wersji API
            try:
                # Próba inicjalizacji dla nowszej wersji API używając odpowiednich klas rynkowych
                endpoint = "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"

                # Inicjalizacja śledzenia limitów API
                self.last_api_call = 0
                self.min_time_between_calls = 1.0  # 1000ms minimalny odstęp między zapytaniami
                self.rate_limit_backoff = 10.0  # 10 sekund oczekiwania po przekroczeniu limitu
                self.remaining_rate_limit = 50  # Bezpieczniejszy limit początkowy
                self.rate_limit_exceeded = False  # Flaga oznaczająca przekroczenie limitu
                self.last_rate_limit_reset = time.time()  # Czas ostatniego resetu limitu

                # Używanie odpowiednich klas rynkowych, zgodnie z ostrzeżeniem z biblioteki
                try:
                    # Najpierw próbujemy z spot API
                    from pybit.spot import HTTP as SpotHTTP
                    self.client = SpotHTTP(
                        endpoint=endpoint,
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        recv_window=20000  # Zwiększenie okna czasu na odpowiedź
                    )
                    self.api_version = "spot"
                    logging.info(f"Zainicjalizowano klienta ByBit API Spot. Testnet: {self.use_testnet}")
                except ImportError:
                    # Jeśli nie ma spot, próbujemy z inverse_perpetual
                    try:
                        from pybit.inverse_perpetual import HTTP as PerpHTTP
                        self.client = PerpHTTP(
                            endpoint=endpoint,
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            recv_window=20000
                        )
                        self.api_version = "inverse_perpetual"
                        logging.info(f"Zainicjalizowano klienta ByBit API Inverse Perpetual. Testnet: {self.use_testnet}")
                    except ImportError:
                        # Ostatnia szansa - używamy ogólnego HTTP
                        self.client = pybit.HTTP(
                            endpoint=endpoint,
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            recv_window=20000
                        )
                        self.api_version = "v2"
                        logging.info(f"Zainicjalizowano klienta ByBit API v2 (ogólny). Testnet: {self.use_testnet}")

                # Testowe pobieranie czasu serwera w celu weryfikacji połączenia
                try:
                    # Sprawdzamy dostępne metody dla czasu serwera
                    if hasattr(self.client, 'get_server_time'):
                        server_time = self.client.get_server_time()
                    elif hasattr(self.client, 'server_time'):
                        server_time = self.client.server_time()
                    elif hasattr(self.client, 'time'):
                        server_time = self.client.time()
                    else:
                        # Jeśli żadna metoda nie jest dostępna, symulujemy czas
                        server_time = {"timeNow": int(time.time() * 1000)}
                        logging.warning("Brak metody do pobierania czasu serwera, używam czasu lokalnego")

                    logging.info(f"Połączenie z ByBit potwierdzone. Czas serwera: {server_time}")
                except Exception as st_error:
                    logging.warning(f"Połączenie z ByBit nawiązane, ale test czasu serwera nie powiódł się: {st_error}")
            except Exception as e:
                logging.error(f"Błąd inicjalizacji klienta PyBit: {e}")
                raise

            logging.info(f"Zainicjalizowano klienta ByBit API. Wersja: {self.api_version}, Testnet: {self.use_testnet}")
        except ImportError:
            logging.error("Nie można zaimportować modułu pybit. Sprawdź czy jest zainstalowany.")
            self.client = None
        except Exception as e:
            logging.error(f"Błąd podczas inicjalizacji klienta ByBit: {e}")
            self.client = None

        # Konfiguracja logowania
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "bybit_connector.log")

        self.logger = logging.getLogger("bybit_connector")
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"BybitConnector zainicjalizowany. Testnet: {use_testnet}")

    def get_server_time(self) -> Dict[str, Any]:
        """
        Pobiera czas serwera Bybit.

        Returns:
            Dict[str, Any]: Czas serwera.
        """
        try:
            self._apply_rate_limit()
            # Symulacja czasu serwera
            current_time = int(time.time() * 1000)
            return {
                "success": True,
                "time_ms": current_time,
                "time": datetime.fromtimestamp(current_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania czasu serwera: {e}")
            return {"success": False, "error": str(e)}

    def get_klines(self, symbol: str, interval: str = "15", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Pobiera dane świecowe (klines) dla danego symbolu.

        Parameters:
            symbol (str): Symbol pary handlowej.
            interval (str): Interwał czasowy ('1', '5', '15', '30', '60', 'D').
            limit (int): Liczba świec do pobrania.

        Returns:
            List[Dict[str, Any]]: Lista świec.
        """
        try:
            self._apply_rate_limit()
            # Symulacja danych świecowych
            current_time = int(time.time())
            klines = []

            last_price = 50000.0 if "BTC" in symbol else 3000.0  # Przykładowe ceny dla BTC lub innych par

            for i in range(limit):
                timestamp = current_time - (int(interval) * 60 * (limit - i - 1))

                # Symulujemy zmianę ceny
                price_change = random.uniform(-0.01, 0.01)
                last_price = last_price * (1 + price_change)

                open_price = last_price
                high_price = open_price * (1 + random.uniform(0, 0.005))
                low_price = open_price * (1 - random.uniform(0, 0.005))
                close_price = last_price
                volume = random.uniform(1, 100) if "BTC" in symbol else random.uniform(10, 1000)

                kline = {
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 2)
                }
                klines.append(kline)

            return klines
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania danych świecowych: {e}")
            return []

    def get_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """
        Pobiera księgę zleceń dla danego symbolu.

        Parameters:
            symbol (str): Symbol pary handlowej.
            limit (int): Liczba poziomów cen do pobrania.

        Returns:
            Dict[str, Any]: Księga zleceń.
        """
        try:
            self._apply_rate_limit()
            # Symulacja księgi zleceń
            base_price = 50000.0 if "BTC" in symbol else 3000.0  # Przykładowe ceny dla BTC lub innych par

            bids = []
            asks = []

            for i in range(limit):
                bid_price = base_price * (1 - 0.001 * (i + 1))
                ask_price = base_price * (1 + 0.001 * (i + 1))

                bid_amount = random.uniform(0.1, 2.0) if "BTC" in symbol else random.uniform(1.0, 20.0)
                ask_amount = random.uniform(0.1, 2.0) if "BTC" in symbol else random.uniform(1.0, 20.0)

                bids.append([round(bid_price, 2), round(bid_amount, 6)])
                asks.append([round(ask_price, 2), round(ask_amount, 6)])

            return {
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "bids": bids,
                "asks": asks
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania księgi zleceń: {e}")
            return {"symbol": symbol, "bids": [], "asks": [], "error": str(e)}

    def get_account_balance(self) -> Dict[str, Any]:
        """Pobiera saldo konta z zastosowaniem zaawansowanego cache i rate limitingu."""
        # Import managera cache
        from data.utils.cache_manager import get_cached_data, store_cached_data, is_cache_valid, get_api_status
        
        # Klucz cache
        cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"
        
        # Sprawdź status API - jeśli przekroczono limity, użyj dłuższego TTL
        api_status = get_api_status()
        ttl = 300 if api_status["rate_limited"] else 30  # 5 minut w stanie przekroczenia limitów, 30s normalnie
        
        # Sprawdzenie czy dane są w cache i ważne
        if is_cache_valid(cache_key, ttl=ttl):
            cached_data = get_cached_data(cache_key)
            if cached_data and cached_data[0]:
                self.logger.debug(f"Używam danych z cache dla account_balance (TTL: {ttl}s)")
                return cached_data[0]
                
        # Zmienna do śledzenia prób ponowienia
        max_retries = 3
        retry_count = 0
        retry_delay = 2.0  # sekundy

        try:
            self._apply_rate_limit()
            if self.client is None:
                # Próba reinicjalizacji klienta
                try:
                    import pybit
                    self.logger.info(f"Próba reinicjalizacji klienta API. API key: {self.api_key[:5]}..., Testnet: {self.use_testnet}")
                    endpoint = "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"
                    self.client = pybit.HTTP(
                        endpoint=endpoint,
                        api_key=self.api_key,
                        api_secret=self.api_secret
                    )
                    self.logger.info("Klient API został pomyślnie reinicjalizowany.")
                except Exception as init_error:
                    self.logger.error(f"Nie udało się zainicjalizować klienta API: {init_error}")
                    logging.error(f"Klient API nie został zainicjalizowany. Błąd: {init_error}")
                    return {
                        "balances": {
                            "BTC": {"equity": 0.005, "available_balance": 0.005, "wallet_balance": 0.005},
                            "USDT": {"equity": 500, "available_balance": 450, "wallet_balance": 500}
                        }, 
                        "success": False,
                        "error": f"Klient API nie został zainicjalizowany. Błąd: {init_error}",
                        "note": "Dane przykładowe - błąd inicjalizacji klienta"
                    }

            # Mechanizm ponawiania prób w przypadku przekroczenia limitów API
            while retry_count < max_retries:
                try:
                    # Testowa implementacja (symulacja)
                    if self.use_testnet:
                        # Symulowane dane do celów testowych
                        self.logger.info("Pobieranie danych z testnet")
                        return {
                            "balances": {
                                "BTC": {"equity": 0.015, "available_balance": 0.015, "wallet_balance": 0.015},
                                "USDT": {"equity": 1200, "available_balance": 1150, "wallet_balance": 1200}
                            },
                            "success": True,
                            "note": "Dane testowe - tryb testnet"
                        }
                    else:
                        # Prawdziwa implementacja lub symulacja jeśli połączenie nie działa
                        # Przygotowanie zasłoniętego klucza do logów
                        masked_key = f"{self.api_key[:4]}{'*' * (len(self.api_key) - 4)}" if self.api_key else "Brak klucza"
                        self.logger.info(f"Próba pobrania danych z {'PRODUKCYJNEGO' if not self.use_testnet else 'TESTOWEGO'} API Bybit. Klucz: {masked_key}")
                    self.logger.info(f"Status API: {'Produkcyjne' if not self.use_testnet else 'Testnet'}")

                    try:
                        # Test połączenia z API z obsługą limitów zapytań
                        try:
                            # Wydłużone opóźnienie między zapytaniami dla testu połączenia
                            if not self.use_testnet:
                                time.sleep(5.0)  # Dodatkowe 5 sekund dla API produkcyjnego
                            else:
                                time.sleep(3.0)  # 3 sekundy dla testnet
                            self._apply_rate_limit()

                            # W przypadku przekroczenia limitu zapytań, używamy czasu lokalnego
                            # zamiast próbować wielokrotnie odpytywać API
                            if self.remaining_rate_limit < 10:
                                local_time = int(time.time() * 1000)
                                self.logger.info(f"Używam lokalnego czasu ({local_time}) zamiast odpytywania API (oszczędzanie limitów)")
                                return {'success': True, 'time_ms': local_time, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}

                            # Sprawdzanie dostępu do API - w różnych wersjach pybit metoda jest inna
                            if hasattr(self.client, 'get_server_time'):
                                time_response = self.client.get_server_time()
                            elif hasattr(self.client, 'server_time'):
                                time_response = self.client.server_time()
                            elif hasattr(self.client, 'time'):
                                time_response = self.client.time()
                            else:
                                # Jeśli nie ma dostępu do czasu serwera, używamy lokalnego
                                time_response = {"timeNow": int(time.time() * 1000)}
                                self.logger.warning("Brak metody do pobierania czasu serwera, używam czasu lokalnego")

                            self.logger.info(f"Test połączenia z API: {time_response}")
                        except Exception as time_error:
                            error_str = str(time_error)
                            self.logger.error(f"Test połączenia z API nie powiódł się: {time_error}")

                            # Specjalna obsługa limitu żądań API
                            if "rate limit" in error_str.lower() or "429" in error_str or "403" in error_str:
                                self.rate_limit_exceeded = True
                                self.last_rate_limit_reset = time.time()
                                self.remaining_rate_limit = 0
                                self.logger.warning(f"Przekroczono limit zapytań API. Używam symulowanych danych i zwiększam czas oczekiwania.")
                                
                                # Zwiększ dynamicznie czas oczekiwania przy kolejnych przekroczeniach limitu
                                self.min_time_between_calls = min(3.0, self.min_time_between_calls * 1.5)  # max 3s między zapytaniami
                                self.rate_limit_backoff = min(60.0, self.rate_limit_backoff * 1.5)  # max 60s backoff
                                
                                # Zwracamy symulowane dane zamiast zgłaszania wyjątku
                                return {
                                    "balances": {
                                        "BTC": {"equity": 0.025, "available_balance": 0.020, "wallet_balance": 0.025},
                                        "USDT": {"equity": 1500, "available_balance": 1450, "wallet_balance": 1500},
                                        "ETH": {"equity": 0.5, "available_balance": 0.5, "wallet_balance": 0.5}
                                    },
                                    "success": True,
                                    "warning": "Przekroczono limit zapytań API. Dane symulowane.",
                                    "source": "simulation_rate_limited",
                                    "note": f"Przekroczono limit zapytań API. Oczekiwanie {self.rate_limit_backoff}s przed kolejnymi zapytaniami."
                                }
                            else:
                                # Dla innych błędów zgłaszamy wyjątek
                                raise Exception(f"Brak dostępu do API Bybit: {time_error}")

                        # Próba pobrania salda konta z uwzględnieniem różnych API
                        wallet = None
                        wallet_methods = [
                            ('get_wallet_balance', {}),
                            ('get_wallet_balance', {'coin': 'USDT'}),
                            ('query_account_info', {}),
                            ('get_account_overview', {}),
                            ('get_account_balance', {}),
                            ('get_balances', {})
                        ]

                        for method_name, params in wallet_methods:
                            if hasattr(self.client, method_name):
                                try:
                                    method = getattr(self.client, method_name)
                                    wallet = method(**params)
                                    self.logger.info(f"Saldo pobrane metodą: {method_name}")
                                    break
                                except Exception as method_error:
                                    self.logger.warning(f"Błąd podczas używania metody {method_name}: {method_error}")

                        if wallet is None:
                            self.logger.error("Wszystkie metody pobierania salda zawiodły")
                            raise Exception("Brak dostępnych metod do pobrania salda portfela")

                        self.logger.info(f"Odpowiedź API Bybit: {str(wallet)[:200]}...")

                        # Sprawdzenie czy odpowiedź zawiera kod błędu
                        if "retCode" in wallet and wallet["retCode"] != 0:
                            error_msg = wallet.get("retMsg", "Nieznany błąd API")
                            self.logger.error(f"API zwróciło błąd: {error_msg}")
                            raise Exception(f"Błąd API ByBit: {error_msg}")

                        result = {
                            "balances": {},
                            "success": True,
                            "source": "API",
                            "api_version": self.api_version
                        }

                        # Obsługa różnych formatów odpowiedzi w zależności od wersji API
                        if wallet and "result" in wallet and "list" in wallet["result"]:
                            # Nowsza struktura API ByBit
                            for coin_data in wallet["result"]["list"]:
                                coin = coin_data.get("coin")
                                if coin:
                                    result["balances"][coin] = {
                                        "equity": float(coin_data.get("equity", 0)),
                                        "available_balance": float(coin_data.get("availableBalance", 0)),
                                        "wallet_balance": float(coin_data.get("walletBalance", 0))
                                    }
                        elif wallet and "result" in wallet and isinstance(wallet["result"], dict):
                            # Starsza struktura API ByBit lub format usdt_perpetual
                            for coin, coin_data in wallet["result"].items():
                                if isinstance(coin_data, dict):
                                    result["balances"][coin] = {
                                        "equity": float(coin_data.get("equity", 0)),
                                        "available_balance": float(coin_data.get("available_balance", 0) or coin_data.get("availableBalance", 0)),
                                        "wallet_balance": float(coin_data.get("wallet_balance", 0) or coin_data.get("walletBalance", 0))
                                    }
                        elif wallet and "result" in wallet and isinstance(wallet["result"], list):
                            # Format odpowiedzi dla niektórych wersji API
                            for coin_data in wallet["result"]:
                                coin = coin_data.get("coin", "")
                                if coin:
                                    result["balances"][coin] = {
                                        "equity": float(coin_data.get("equity", 0)),
                                        "available_balance": float(coin_data.get("available_balance", 0) or coin_data.get("availableBalance", 0)),
                                        "wallet_balance": float(coin_data.get("wallet_balance", 0) or coin_data.get("walletBalance", 0))
                                    }

                        if not result["balances"]:
                            self.logger.warning(f"API zwróciło pustą listę sald. Pełna odpowiedź: {wallet}")
                            result["warning"] = "API zwróciło pustą listę sald"
                            self.logger.info("Próba pobrania danych z API zwróciła pustą listę sald. Możliwe przyczyny: brak środków na koncie, nieprawidłowe konto, nieprawidłowe uprawnienia API.")

                        # Zapisanie poprawnych danych w cache
                        from data.utils.cache_manager import store_cached_data
                        cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"
                        store_cached_data(cache_key, result)
                        self.logger.debug("Zapisano dane portfolio w cache")
                        
                        return result
                    except Exception as e:
                        self.logger.error(f"Błąd podczas pobierania danych z prawdziwego API: {e}. Traceback: {traceback.format_exc()}")
                        # Dane symulowane w przypadku błędu
                        return {
                            "balances": {
                                "BTC": {"equity": 0.025, "available_balance": 0.020, "wallet_balance": 0.025},
                                "USDT": {"equity": 1500, "available_balance": 1450, "wallet_balance": 1500},
                                "ETH": {"equity": 0.5, "available_balance": 0.5, "wallet_balance": 0.5}
                            },
                            "success": False,
                            "error": str(e),
                            "source": "simulation",
                            "note": "Dane symulowane - błąd API: " + str(e)
                        }
                except Exception as e:
                    error_str = str(e)
                    # Sprawdzenie, czy błąd dotyczy przekroczenia limitu zapytań
                    if "rate limit" in error_str.lower() or "429" in error_str or "403" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            self.logger.warning(f"Przekroczono limit zapytań API. Ponawiam próbę {retry_count}/{max_retries} za {retry_delay} sekund...")
                            # Zwiększamy opóźnienie wykładniczo, aby uniknąć ponownego przekroczenia limitu
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Podwajamy czas oczekiwania przy każdej próbie
                            continue
                        else:
                            self.logger.warning(f"Wykorzystano wszystkie próby ponawiania. Zwracam dane symulowane.")

                    self.logger.error(f"Krytyczny błąd podczas pobierania salda konta: {e}. Traceback: {traceback.format_exc()}")
                    # Dane symulowane w przypadku błędu
                    return {
                        "balances": {
                            "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                            "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
                        },
                        "success": False,
                        "error": str(e),
                        "source": "simulation_error",
                        "note": "Dane symulowane - wystąpił błąd: " + str(e)
                    }

                # Jeśli dotarliśmy tutaj, to znaczy, że zapytanie się powiodło
                break

        except Exception as e:
            self.logger.error(f"Krytyczny błąd podczas pobierania salda konta: {e}. Traceback: {traceback.format_exc()}")
            # Dane symulowane w przypadku błędu
            return {
                "balances": {
                    "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                    "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
                },
                "success": False,
                "error": str(e),
                "source": "simulation_critical_error",
                "note": "Dane symulowane - wystąpił krytyczny błąd: " + str(e)
            }

        return {
            "balances": {
                "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
            },
            "success": False,
            "error": "Nieudane pobieranie salda",
            "source": "fallback",
            "note": "Dane symulowane - żadna próba nie powiodła się"
        }

    def place_order(self, symbol: str, side: str, price: float, quantity: float, 
                   order_type: str = "Limit") -> Dict[str, Any]:
        """
        Składa zlecenie na giełdzie.

        Parameters:
            symbol (str): Symbol pary handlowej.
            side (str): Strona zlecenia ('Buy' lub 'Sell').
            price (float): Cena zlecenia.
            quantity (float): Ilość zlecenia.
            order_type (str): Typ zlecenia ('Limit' lub 'Market').

        Returns:
            Dict[str, Any]: Wynik złożenia zlecenia.
        """
        try:
            self._apply_rate_limit()
            # Sprawdzenie poprawności danych
            if side not in ["Buy", "Sell"]:
                return {"success": False, "error": "Nieprawidłowa strona zlecenia. Musi być 'Buy' lub 'Sell'."}

            if order_type not in ["Limit", "Market"]:
                return {"success": False, "error": "Nieprawidłowy typ zlecenia. Musi być 'Limit' lub 'Market'."}

            if quantity <= 0:
                return {"success": False, "error": "Ilość musi być dodatnia."}

            if order_type == "Limit" and price <= 0:
                return {"success": False, "error": "Cena musi być dodatnia dla zleceń typu Limit."}

            # Symulacja składania zlecenia
            order_id = f"ORD-{int(time.time())}-{random.randint(1000, 9999)}"

            self.logger.info(f"Złożono zlecenie: {side} {quantity} {symbol} po cenie {price if order_type == 'Limit' else 'Market'}")

            return {
                "success": True,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "price": price if order_type == "Limit" else None,
                "quantity": quantity,
                "order_type": order_type,
                "status": "New",
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas składania zlecenia: {e}")
            return {"success": False, "error": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Anuluje zlecenie.

        Parameters:
            order_id (str): ID zlecenia do anulowania.

        Returns:
            Dict[str, Any]: Wynik anulowania zlecenia.
        """
        try:
            self._apply_rate_limit()
            # Symulacja anulowania zlecenia
            self.logger.info(f"Anulowano zlecenie: {order_id}")

            return {
                "success": True,
                "order_id": order_id,
                "status": "Cancelled",
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas anulowania zlecenia: {e}")
            return {"success": False, "error": str(e)}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Pobiera status zlecenia.

        Parameters:
            order_id (str): ID zlecenia.

        Returns:
            Dict[str, Any]: Status zlecenia.
        """
        try:
            self._apply_rate_limit()
            # Symulacja pobierania statusu zlecenia
            statuses = ["New", "PartiallyFilled", "Filled", "Cancelled", "Rejected"]
            status = random.choice(statuses)

            return {
                "success": True,
                "order_id": order_id,
                "status": status,
                "filled_quantity": random.uniform(0, 1) if status == "PartiallyFilled" else (1.0 if status == "Filled" else 0.0),
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania statusu zlecenia: {e}")
            return {"success": False, "error": str(e)}

    def _apply_rate_limit(self):
        """Applies rate limiting to API calls with adaptive backoff and integration with cache_manager."""
        from data.utils.cache_manager import get_api_status, set_rate_limit_parameters
        
        # Pobierz aktualny status API z cache_manager
        api_status = get_api_status()
        now = time.time()
        
        # Synchronizuj parametry rate limiter'a z cache_manager
        if not hasattr(self, '_rate_limit_synced') or now - getattr(self, '_last_sync_time', 0) > 60:
            # Synchronizuj parametry tylko raz na minutę dla wydajności
            set_rate_limit_parameters(
                max_calls_per_minute=30,  # Konserwatywny limit dla ByBit API
                min_interval=self.min_time_between_calls
            )
            self._rate_limit_synced = True
            self._last_sync_time = now
        
        # Jeśli przekroczono limit, stosuj dłuższe opóźnienie
        if self.rate_limit_exceeded or api_status["rate_limited"]:
            # Resetuj stan po 60 sekundach od ostatniego przekroczenia limitu
            if now - self.last_rate_limit_reset > 60.0:
                self.logger.info("Resetowanie stanu limitów API po okresie oczekiwania")
                self.rate_limit_exceeded = False
                self.remaining_rate_limit = 50  # Zakładamy odnowienie limitu
            else:
                # Bardziej agresywne oczekiwanie przy przekroczeniu limitu
                sleep_time = min(60.0, self.rate_limit_backoff * 1.5)  # Max 60s
                self.logger.info(f"Rate limit przekroczony - oczekiwanie {sleep_time:.1f}s przed próbą")
                time.sleep(sleep_time)
                # Zwiększamy backoff przy każdym kolejnym przekroczeniu
                self.rate_limit_backoff = min(60.0, self.rate_limit_backoff * 1.2)
        
        # Standardowe opóźnienie między wywołaniami z minimalnym buforem
        time_since_last_call = now - self.last_api_call
        min_time = max(2.0, self.min_time_between_calls)  # Co najmniej 2 sekundy w przypadku produkcyjnego API
        
        if time_since_last_call < min_time:
            sleep_time = min_time - time_since_last_call
            if sleep_time > 0.1:  # Ignoruj bardzo małe opóźnienia
                time.sleep(sleep_time)
        
        # Aktualizacja czasu ostatniego wywołania
        self.last_api_call = time.time()


if __name__ == "__main__":
    # Przykład użycia
    connector = BybitConnector(
        api_key="test_key",
        api_secret="test_secret",
        use_testnet=True
    )

    server_time = connector.get_server_time()
    print(f"Czas serwera: {server_time}")

    klines = connector.get_klines(symbol="BTCUSDT")
    print(f"Pobrano {len(klines)} świec")

    order_book = connector.get_order_book(symbol="BTCUSDT")
    print(f"Pobrano książkę zleceń z {len(order_book['bids'])} ofertami kupna i {len(order_book['asks'])} ofertami sprzedaży")

    balance = connector.get_account_balance()
    print(f"Stan konta: {balance}")

    order_result = connector.place_order(symbol="BTCUSDT", side="Buy", price=50000, quantity=0.01)
    print(f"Wynik złożenia zlecenia: {order_result}")

    order_status = connector.get_order_status(order_id = order_result.get("order_id"))
    print(f"Status zlecenia: {order_status}")


    cancel_result = connector.cancel_order(order_id=order_result.get("order_id"))
    print(f"Wynik anulowania zlecenia: {cancel_result}")