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
import hmac
import hashlib
import requests
import pybit # Dodano brakujący import
from datetime import datetime
from typing import Dict, List, Any, Optional

class RateLimitExceeded(Exception):
    """Wyjątek rzucany gdy przekroczono limit zapytań API."""
    pass

class TimeoutError(Exception):
    """Wyjątek rzucany gdy zapytanie przekroczyło limit czasu."""
    pass

def is_env_flag_true(env_var_name: str) -> bool:
    """
    Sprawdza, czy zmienna środowiskowa jest ustawiona na wartość prawdziwą.

    Args:
        env_var_name: Nazwa zmiennej środowiskowej do sprawdzenia

    Returns:
        bool: True jeśli wartość zmiennej to "1", "true" lub "yes" (bez uwzględnienia wielkości liter)
    """
    return os.getenv(env_var_name, "").strip().lower() in ["1", "true", "yes"]


class ApiMetricsCollector:
    """Kolektor metryk wydajności API."""
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "response_times": [],
            "rate_limit_hits": 0,
            "errors_by_type": {},
            "last_reset_time": time.time()
        }
        
    def record_request(self, success: bool, response_time: float, error_type: str = None):
        """Zapisuje metryki pojedynczego zapytania."""
        self.metrics["requests_total"] += 1
        if success:
            self.metrics["requests_success"] += 1
        else:
            self.metrics["requests_failed"] += 1
            if error_type:
                self.metrics["errors_by_type"][error_type] = self.metrics["errors_by_type"].get(error_type, 0) + 1
        
        self.metrics["response_times"].append(response_time)
        
    def record_rate_limit(self):
        """Zapisuje wystąpienie przekroczenia limitu zapytań."""
        self.metrics["rate_limit_hits"] += 1
        
    def get_metrics_summary(self) -> dict:
        """Zwraca podsumowanie zebranych metryk."""
        if not self.metrics["response_times"]:
            return {"error": "Brak zebranych metryk"}
            
        avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        success_rate = (self.metrics["requests_success"] / self.metrics["requests_total"] * 100) if self.metrics["requests_total"] > 0 else 0
        
        return {
            "total_requests": self.metrics["requests_total"],
            "success_rate": f"{success_rate:.1f}%",
            "average_response_time": f"{avg_response_time:.3f}s",
            "rate_limit_hits": self.metrics["rate_limit_hits"],
            "errors_by_type": dict(self.metrics["errors_by_type"]),
            "collection_period": f"{time.time() - self.metrics['last_reset_time']:.0f}s"
        }
        
    def reset_metrics(self):
        """Resetuje wszystkie zebrane metryki."""
        self.__init__()


class BybitConnector:
    """
    Klasa do komunikacji z giełdą Bybit.
    """
    def __init__(self,
                 api_key: str = None,
                 api_secret: str = None,
                 use_testnet: bool = None,
                 lazy_connect: bool = True,
                 proxies: Dict[str, str] = None,
                 market_type: str = "spot",
                 account_type: str = "UNIFIED"):
        """
        Inicjalizuje połączenie z Bybit.

        Parameters:
            api_key (str): Klucz API Bybit.
            api_secret (str): Sekret API Bybit.
            use_testnet (bool): Czy używać środowiska testowego (domyślnie odczytywane ze zmiennych środowiskowych).
            lazy_connect (bool): Czy opóźnić połączenie z API do pierwszego użycia.
            proxies (Dict[str, str]): Nie używane w wersji lokalnej (zachowane dla kompatybilności)
            market_type (str): Typ rynku - "spot" lub "futures"
        """
        # Priorytetyzacja parametrów:
        # 1. Przekazany parametr use_testnet
        # 2. BYBIT_TESTNET
        # 3. Domyślnie False (produkcja)
        if use_testnet is None:
            if is_env_flag_true("BYBIT_TESTNET"):
                self.use_testnet = True
            elif os.getenv("BYBIT_TESTNET", "false").lower() == "true":
                self.use_testnet = True
            else:
                self.use_testnet = False  # Domyślnie używamy produkcyjnego API
        else:
            self.use_testnet = use_testnet

        self.api_key = api_key
        self.api_secret = api_secret
        self.lazy_connect = lazy_connect  # Dodany brakujący atrybut
        self.proxies = None  # Nie używamy proxy w wersji lokalnej
        self.market_type = market_type.lower()
        self.account_type = account_type.upper()
        self.max_requests_per_minute = 50 if self.use_testnet else 15
        self.max_order_value_usd = 1000.0 if self.use_testnet else 100000.0

        # Inicjalizacja licznika błędów
        self.consecutive_failures = 0

        # Upewnij się, że używasz odpowiedniego URL API
        self.base_url = "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"
        self._client = None  # Inicjalizacja atrybutu _client (poprawione)
        self.client = None
        self.api_version = None
        self._connection_initialized = False
        self._connection_test_time = 0
        self._connection_test_result = None

        # Inicjalizacja śledzenia limitów API - bardziej restrykcyjne dla produkcji
        self.last_api_call = 0
        self.min_time_between_calls = 10.0 if not self.use_testnet else 3.0  # Ostrzejsze limity dla produkcji
        self.rate_limit_backoff = 30.0 if not self.use_testnet else 10.0  # Znacznie dłuższy backoff dla produkcji
        self.remaining_rate_limit = 20 if not self.use_testnet else 50  # Bezpieczniejszy limit początkowy dla produkcji
        self.rate_limit_exceeded = False  # Flaga oznaczająca przekroczenie limitu
        self.last_rate_limit_reset = time.time(
        )  # Czas ostatniego resetu limitu

        # Dodatkowe limity API dla środowiska produkcyjnego
        if not self.use_testnet:
            self.min_time_between_calls = 15.0  # 15 sekund między zapytaniami
            self.rate_limit_reset_time = 300.0  # 5 minut na reset limitów
            self.max_retry_attempts = 3
        else:
            self.min_time_between_calls = 3.0
            self.rate_limit_reset_time = 60.0
            self.max_retry_attempts = 5

        # Konfiguracja logowania
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "bybit_connector.log")

        self.logger = logging.getLogger("bybit_connector")
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        # Sprawdź czy używamy produkcyjnego API i pokaż ostrzeżenie
        if self.is_production_api():
            # Ostrzeżenie jest pokazywane w is_production_api()
            # Dodatkowe sprawdzenia kluczy produkcyjnych
            if self.api_key is None or len(
                    self.api_key) < 10 or self.api_secret is None or len(
                        self.api_secret) < 10:
                self.logger.critical(
                    "BŁĄD: Nieprawidłowe klucze API dla środowiska produkcyjnego!"
                )
                raise ValueError(
                    "Nieprawidłowe klucze API dla środowiska produkcyjnego. Sprawdź konfigurację."
                )

            # Mechanizm podwójnego potwierdzenia trybu produkcyjnego
            production_confirmed = os.getenv("BYBIT_PRODUCTION_CONFIRMED", "false").lower() == "true"
            production_extra_check = os.getenv("BYBIT_PRODUCTION_ENABLED", "false").lower() == "true"

            if not (production_confirmed and production_extra_check):
                self.logger.critical("UWAGA: Używanie produkcyjnego API wymaga jawnego potwierdzenia!")
                self.logger.critical("Musisz ustawić OBYDWIE zmienne środowiskowe:")
                self.logger.critical("1. BYBIT_PRODUCTION_CONFIRMED=true")
                self.logger.critical("2. BYBIT_PRODUCTION_ENABLED=true")
                
                print("\n" + "!"*80)
                print("!!! UWAGA !!! Wykryto próbę użycia PRODUKCYJNEGO API Bybit bez pełnego potwierdzenia!")
                print("!!! To może prowadzić do REALNYCH TRANSAKCJI z prawdziwymi środkami !!!")
                print("!!! Aby potwierdzić, że chcesz użyć produkcyjnego API, ustaw OBYDWIE zmienne środowiskowe:")
                print("!!! 1. BYBIT_PRODUCTION_CONFIRMED=true")
                print("!!! 2. BYBIT_PRODUCTION_ENABLED=true")
                print("!"*80 + "\n")
                
                # Automatycznie przełącz na testnet w przypadku braku potwierdzenia
                self.use_testnet = True
                self.base_url = "https://api-testnet.bybit.com"
                self.logger.warning("Przełączono na testnet ze względu na brak pełnego potwierdzenia dla API produkcyjnego")
                
                # Nie rzucamy wyjątku - zamiast tego przełączamy na testnet
                # Zapobiega to przerwaniu aplikacji
        else:
            self.logger.info("Używasz testnet API (środowisko testowe).")

        self.logger.info(
            f"BybitConnector zainicjalizowany. Testnet: {self.use_testnet}, Lazy connect: {lazy_connect}, Market type: {self.market_type}"
        )

        # Sprawdź czy już mamy informację o przekroczeniu limitów w cache
        try:
            from data.utils.cache_manager import get_cached_data
            rate_limited_data, found = get_cached_data("api_rate_limited")

            # Inicjalizujemy flagę jako False
            self.rate_limit_exceeded = False

            if found and rate_limited_data is not None:
                # Bezpieczny dostęp - wartość może być słownikiem lub typem prostym
                if isinstance(rate_limited_data, dict):
                    # Jeśli to słownik, sprawdź klucz "value"
                    if rate_limited_data.get("value", False):
                        self.rate_limit_exceeded = True
                elif isinstance(rate_limited_data, bool):
                    # Jeśli to bezpośrednio boolean
                    self.rate_limit_exceeded = rate_limited_data
                else:
                    # Próba konwersji na boolean
                    try:
                        self.rate_limit_exceeded = bool(rate_limited_data)
                    except (ValueError, TypeError):
                        self.logger.warning(
                            f"Nie można skonwertować wartości cache na boolean: {rate_limited_data}"
                        )

                if self.rate_limit_exceeded:
                    self.logger.warning(
                        "Wykryto zapisaną flagę przekroczenia limitów API. Ustawiam tryb oszczędzania limitów."
                    )
        except Exception as e:
            self.logger.warning(f"Nie można sprawdzić stanu limitów API: {e}")
            # Bezpiecznie zakładamy False w przypadku błędu
            self.rate_limit_exceeded = False

        # Kontynuuj tylko inicjalizację komponentów, ale nie testuj API jeśli lazy_connect
        if not lazy_connect:
            self._initialize_client()
        else:
            self.logger.info(
                f"BybitConnector w trybie lazy initialization. Testnet: {self.use_testnet}, Market type: {self.market_type}"
            )

    def _initialize_client(self):
        """Inicjalizuje klienta API Bybit."""
        if self._client:
            return  # Już zainicjowany

        self.logger.info(f"Inicjalizacja klienta Bybit (Testnet: {self.use_testnet}, Market: {self.market_type}, Account: {self.account_type})")

        api_key = self.api_key or os.getenv("BYBIT_API_KEY")
        api_secret = self.api_secret or os.getenv("BYBIT_API_SECRET")

        if not api_key or not api_secret:
            self.logger.warning("Brak kluczy API Bybit. Niektóre funkcje mogą być niedostępne.")
            # Inicjalizacja klienta bez uwierzytelniania dla publicznych endpointów
            try:
                if self.market_type == "spot":
                    # Użycie pybit.HTTP dla Spot v3 (publiczne)
                    self._client = pybit.HTTP(testnet=self.use_testnet)
                    self.logger.info("Zainicjowano klienta Bybit Spot (publiczny).")
                elif self.market_type in ["futures", "linear", "inverse"]:
                    # Użycie pybit.HTTP dla Unified/Contract v5 (publiczne)
                    self._client = pybit.HTTP(testnet=self.use_testnet)
                    self.logger.info(f"Zainicjowano klienta Bybit {self.market_type.capitalize()} (publiczny).")
                else:
                    self.logger.error(f"Nieznany typ rynku: {self.market_type}")
                    self._client = None # Ustawienie na None w przypadku błędu
                    return # Zakończ inicjalizację jeśli typ rynku jest nieznany
            except Exception as e:
                self.logger.error(f"Błąd inicjalizacji publicznego klienta Bybit: {e}", exc_info=True)
                self._client = None # Ustawienie na None w przypadku błędu
            return # Zakończ, bo nie ma kluczy do pełnej inicjalizacji

        # Inicjalizacja z uwierzytelnianiem
        try:
            # Używamy zunifikowanego klienta HTTP dla v5 API
            self._client = pybit.HTTP(
                api_key=api_key,
                api_secret=api_secret,
                testnet=self.use_testnet,
                # Można dodać inne opcje konfiguracyjne, np. log_requests=True
            )
            self.logger.info(f"Pomyślnie zainicjowano klienta Bybit v5 (Testnet: {self.use_testnet}, Market: {self.market_type}, Account: {self.account_type}).")

            # Sprawdzenie połączenia przez pobranie czasu serwera
            server_time = self.get_server_time()
            if server_time and server_time.get('success'):
                 self.logger.info(f"Połączenie z Bybit API v5 pomyślne. Czas serwera: {server_time.get('time_ms')}")
            else:
                 self.logger.warning("Nie udało się zweryfikować połączenia z Bybit API v5.")

        except Exception as e:
            self.logger.error(f"Krytyczny błąd inicjalizacji klienta Bybit: {e}", exc_info=True)
            self._client = None # Ustawienie na None w przypadku błędu

    def get_server_time(self) -> Optional[Dict]:
        """Pobiera aktualny czas serwera Bybit."""
        response = self._make_request('GET', '/v5/market/time')
        if response and response.get('retCode') == 0 and response.get('result'):
            # Dostosowanie do formatu v5
            time_nano = int(response['result'].get('timeNano', '0')) # Bezpieczne pobranie i konwersja
            time_sec = int(response['result'].get('timeSecond', '0'))
            return {
                'success': True,
                'time_ms': time_nano // 1000000,
                'time_s': time_sec,
                'raw_response': response # Dodajemy surową odpowiedź dla debugowania
            }
        else:
            self.logger.error(f"Nie udało się pobrać czasu serwera Bybit. Odpowiedź: {response}")
            return {'success': False, 'error': response.get('retMsg', 'Unknown error'), 'raw_response': response}

    def get_klines(self, symbol: str, interval: str, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: int = 200) -> List[Dict]:
        """
        Pobiera dane K-line (świecowe) dla danego symbolu i interwału.

        Args:
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT').
            interval (str): Interwał świec (np. '1', '5', '15', '60', 'D', 'W', 'M').
                            Mapowanie interwałów Bybit: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
            start_time (Optional[int]): Początkowy timestamp w milisekundach.
            end_time (Optional[int]): Końcowy timestamp w milisekundach.
            limit (int): Liczba świec do pobrania (max 1000).

        Returns:
            List[Dict]: Lista słowników reprezentujących świece, lub pusta lista w przypadku błędu.
                        Format świecy: {'timestamp': ms, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'turnover': float}
        """
        params = {
            'category': 'spot' if self.market_type == 'spot' else ('linear' if self.market_type == 'futures' else self.market_type), # Dostosuj kategorię
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000) # Limit API to 1000
        }
        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time

        response = self._make_request('GET', '/v5/market/kline', params=params)

        if response and response.get('retCode') == 0 and response.get('result') and 'list' in response['result']:
            klines_raw = response['result']['list']
            # Formatowanie danych do oczekiwanej struktury
            klines_formatted = []
            # Indeksy w odpowiedzi Bybit v5: [timestamp, open, high, low, close, volume, turnover]
            for k in klines_raw:
                 try:
                     klines_formatted.append({
                         'timestamp': int(k[0]),
                         'open': float(k[1]),
                         'high': float(k[2]),
                         'low': float(k[3]),
                         'close': float(k[4]),
                         'volume': float(k[5]),
                         'turnover': float(k[6])
                     })
                 except (IndexError, ValueError, TypeError) as e:
                      self.logger.warning(f"Błąd przetwarzania świecy: {k}. Błąd: {e}")
                      continue # Pomiń błędną świecę
            # Odwrócenie listy, bo Bybit zwraca od najnowszych do najstarszych
            return klines_formatted[::-1]
        else:
            self.logger.error(f"Nie udało się pobrać danych K-line dla {symbol}. Odpowiedź: {response}")
            return []

    def get_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, List[List[float]]]:
        """
        Pobiera księgę zleceń dla danego symbolu.

        Args:
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT').
            limit (Optional[int]): Głębokość księgi zleceń.
                                   Spot: 1-500. Domyślnie 1.
                                   Futures (Linear/Inverse): 1, 25, 50, 100, 200, 500. Domyślnie 25.

        Returns:
            Dict[str, List[List[float]]]: Słownik zawierający 'bids' i 'asks',
                                         lub pusty słownik w przypadku błędu.
                                         Format: {'bids': [[price, size], ...], 'asks': [[price, size], ...]}
        """
        params = {
            'category': 'spot' if self.market_type == 'spot' else ('linear' if self.market_type == 'futures' else self.market_type),
            'symbol': symbol.upper()
        }
        # Ustawienie limitu zgodnie z dokumentacją API v5
        if self.market_type == 'spot':
            params['limit'] = limit if limit and 1 <= limit <= 500 else 1 # Domyślnie 1 dla spot
        else: # Futures (linear/inverse)
            valid_limits = [1, 25, 50, 100, 200, 500]
            params['limit'] = limit if limit in valid_limits else 25 # Domyślnie 25 dla futures

        response = self._make_request('GET', '/v5/market/orderbook', params=params)

        if response and response.get('retCode') == 0 and response.get('result'):
            result = response['result']
            # Formatowanie danych do oczekiwanej struktury [price, size]
            # Odpowiedź v5: result['b'] = bids, result['a'] = asks
            # Każdy element to [price_str, size_str]
            try:
                bids = [[float(p), float(s)] for p, s in result.get('b', [])]
                asks = [[float(p), float(s)] for p, s in result.get('a', [])]
                return {'bids': bids, 'asks': asks}
            except (ValueError, TypeError) as e:
                 self.logger.error(f"Błąd przetwarzania danych orderbook dla {symbol}: {e}. Odpowiedź: {response}")
                 return {'bids': [], 'asks': []}
        else:
            self.logger.error(f"Nie udało się pobrać księgi zleceń dla {symbol}. Odpowiedź: {response}")
            return {'bids': [], 'asks': []}

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: Optional[float] = None, time_in_force: str = 'GTC', reduce_only: bool = False, close_on_trigger: bool = False, position_idx: Optional[int] = None, client_order_id: Optional[str] = None) -> Optional[Dict]:
        """
        Składa nowe zlecenie.

        Args:
            symbol (str): Symbol (np. 'BTCUSDT').
            side (str): 'Buy' lub 'Sell'.
            order_type (str): 'Limit' lub 'Market'.
            qty (float): Ilość. Dla zleceń Market typu 'Buy' na rynku spot, może to być wartość w walucie kwotowanej (np. USDT). Sprawdź dokumentację Bybit.
            price (Optional[float]): Cena dla zleceń Limit.
            time_in_force (str): 'GTC', 'IOC', 'FOK', 'PostOnly'.
            reduce_only (bool): Tylko dla zleceń zamykających pozycję (Futures).
            close_on_trigger (bool): Tylko dla zleceń warunkowych (Futures).
            position_idx (Optional[int]): Indeks pozycji (0 dla trybu jednostronnego, 1 dla Buy side, 2 dla Sell side w trybie Hedge) (Futures).
            client_order_id (Optional[str]): Własny identyfikator zlecenia.

        Returns:
            Optional[Dict]: Słownik z informacjami o złożonym zleceniu lub None w przypadku błędu.
        """
        if not self._client:
            self.logger.error("Klient API nie jest zainicjowany.")
            return None

        category = 'spot' if self.market_type == 'spot' else ('linear' if self.market_type == 'futures' else self.market_type)

        params = {
            'category': category,
            'symbol': symbol.upper(),
            'side': side.capitalize(), # 'Buy' lub 'Sell'
            'orderType': order_type.capitalize(), # 'Limit' lub 'Market'
            'qty': str(qty), # Ilość musi być stringiem
            'timeInForce': time_in_force,
        }

        if order_type.capitalize() == 'Limit':
            if price is None:
                self.logger.error("Cena jest wymagana dla zlecenia typu Limit.")
                return None
            params['price'] = str(price) # Cena musi być stringiem

        if client_order_id:
            # Sprawdzenie długości i znaków client_order_id zgodnie z wymaganiami Bybit v5
            if 1 <= len(client_order_id) <= 36: # Maksymalna długość to 36 znaków
                 # Można dodać walidację znaków, jeśli jest wymagana
                 params['orderLinkId'] = client_order_id
            else:
                 self.logger.warning(f"Nieprawidłowa długość client_order_id: {len(client_order_id)}. Ignoruję.")

        # Parametry specyficzne dla Futures
        if category != 'spot':
            params['reduceOnly'] = reduce_only
            params['closeOnTrigger'] = close_on_trigger
            if position_idx is not None and position_idx in [0, 1, 2]:
                params['positionIdx'] = position_idx
            # Dla zleceń Market typu 'Buy' na futures, 'qty' to ilość kontraktów
            # Dla zleceń Market typu 'Sell' na futures, 'qty' to ilość kontraktów

        # Specyficzna obsługa zleceń Market 'Buy' na rynku Spot (jeśli 'qty' ma być wartością)
        # Dokumentacja v5 mówi, że dla Spot Market Buy, 'qty' to ilość bazowej waluty.
        # Jeśli chcemy kupić za określoną kwotę USDT, musimy użyć innego parametru lub endpointu (jeśli dostępny)
        # lub obliczyć ilość BTC na podstawie aktualnej ceny rynkowej (co jest ryzykowne).
        # Na razie zakładamy, że 'qty' to ilość waluty bazowej (np. BTC).

        response = self._make_request('POST', '/v5/order/create', data=params)

        if response and response.get('retCode') == 0 and response.get('result'):
            self.logger.info(f"Zlecenie złożone pomyślnie: {response['result']}")
            return response['result'] # Zwraca {'orderId': '...', 'orderLinkId': '...'}
        else:
            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            ret_code = response.get('retCode', -1) if response else -1
            self.logger.error(f"Nie udało się złożyć zlecenia ({symbol}, {side}, {qty}). Kod: {ret_code}, Błąd: {error_msg}. Parametry: {params}. Odpowiedź: {response}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None, order_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Pobiera listę otwartych zleceń.

        Args:
            symbol (Optional[str]): Filtruj wg symbolu.
            order_id (Optional[str]): Filtruj wg ID zlecenia.
            limit (int): Maksymalna liczba zleceń do pobrania (max 50).

        Returns:
            List[Dict]: Lista otwartych zleceń lub pusta lista w przypadku błędu.
        """
        category = 'spot' if self.market_type == 'spot' else ('linear' if self.market_type == 'futures' else self.market_type)
        params = {
            'category': category,
            'limit': min(limit, 50)
        }
        if symbol:
            params['symbol'] = symbol.upper()
        if order_id:
            params['orderId'] = order_id
            # Można też dodać filtrowanie po 'orderLinkId'

        response = self._make_request('GET', '/v5/order/realtime', params=params)

        if response and response.get('retCode') == 0 and response.get('result') and 'list' in response['result']:
            return response['result']['list']
        else:
            self.logger.error(f"Nie udało się pobrać otwartych zleceń. Odpowiedź: {response}")
            return []

    def cancel_order(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Optional[Dict]:
        """
        Anuluje istniejące zlecenie.

        Args:
            symbol (str): Symbol zlecenia.
            order_id (Optional[str]): ID zlecenia Bybit.
            client_order_id (Optional[str]): Własny ID zlecenia (orderLinkId).
                                            Należy podać order_id LUB client_order_id.

        Returns:
            Optional[Dict]: Słownik z wynikiem operacji lub None w przypadku błędu.
        """
        if not order_id and not client_order_id:
            self.logger.error("Należy podać order_id lub client_order_id do anulowania zlecenia.")
            return None

        category = 'spot' if self.market_type == 'spot' else ('linear' if self.market_type == 'futures' else self.market_type)
        params = {
            'category': category,
            'symbol': symbol.upper(),
        }
        if order_id:
            params['orderId'] = order_id
        if client_order_id:
            params['orderLinkId'] = client_order_id

        response = self._make_request('POST', '/v5/order/cancel', data=params)

        if response and response.get('retCode') == 0 and response.get('result'):
            self.logger.info(f"Zlecenie anulowane pomyślnie: {response['result']}")
            # Odpowiedź v5: {'orderId': '...', 'orderLinkId': '...'}
            return response['result']
        else:
            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            ret_code = response.get('retCode', -1) if response else -1
            self.logger.error(f"Nie udało się anulować zlecenia ({symbol}, ID: {order_id or client_order_id}). Kod: {ret_code}, Błąd: {error_msg}. Odpowiedź: {response}")
            return None

    def get_balance(self, account_type: Optional[str] = None, coin: Optional[str] = None) -> Optional[Dict]:
        """
        Pobiera saldo konta.

        Args:
            account_type (Optional[str]): Typ konta ('UNIFIED', 'CONTRACT', 'SPOT'). Domyślnie używa self.account_type.
            coin (Optional[str]): Filtruj wg konkretnej waluty (np. 'USDT', 'BTC').

        Returns:
            Optional[Dict]: Słownik z saldem konta lub None w przypadku błędu.
                           Struktura odpowiedzi zależy od account_type.
        """
        acc_type = account_type or self.account_type
        params = {
            'accountType': acc_type.upper()
        }
        if coin:
            params['coin'] = coin.upper()

        response = self._make_request('GET', '/v5/account/wallet-balance', params=params)

        if response and response.get('retCode') == 0 and response.get('result') and 'list' in response['result']:
             # Odpowiedź zawiera listę, zazwyczaj z jednym elementem dla danego typu konta
             if response['result']['list']:
                  # Zwracamy pierwszy element listy, który zawiera szczegóły salda
                  return response['result']['list'][0]
             else:
                  self.logger.warning(f"Otrzymano pustą listę sald dla konta {acc_type}.")
                  return {} # Zwróć pusty słownik, jeśli lista jest pusta
        else:
            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            ret_code = response.get('retCode', -1) if response else -1
            self.logger.error(f"Nie udało się pobrać salda dla konta {acc_type}. Kod: {ret_code}, Błąd: {error_msg}. Odpowiedź: {response}")
            return None

    def get_positions(self, symbol: Optional[str] = None, base_coin: Optional[str] = None) -> List[Dict]:
        """
        Pobiera otwarte pozycje (tylko dla Futures).

        Args:
            symbol (Optional[str]): Filtruj wg symbolu (np. 'BTCUSDT').
            base_coin (Optional[str]): Filtruj wg waluty bazowej (np. 'BTC') - dla Inverse.

        Returns:
            List[Dict]: Lista otwartych pozycji lub pusta lista w przypadku błędu.
        """
        if self.market_type == 'spot':
            self.logger.warning("Pobieranie pozycji nie jest dostępne dla rynku Spot.")
            return []

        # Dla v5, kategoria 'linear' lub 'inverse'
        category = 'linear' if self.market_type == 'futures' else self.market_type # Zakładając, że 'futures' to linear
        params = {
            'category': category,
            'settleCoin': 'USDT' if category == 'linear' else None # Settle coin wymagany dla linear/inverse
        }
        if symbol:
            params['symbol'] = symbol.upper()
        if base_coin and category == 'inverse': # baseCoin ma sens tylko dla inverse
             params['baseCoin'] = base_coin.upper()
        elif category == 'linear' and base_coin:
             self.logger.warning("Parametr 'base_coin' jest ignorowany dla rynku Linear (Użyj 'symbol').")

        # Można dodać 'limit' i 'cursor' do paginacji, jeśli spodziewamy się wielu pozycji
        # params['limit'] = 50

        response = self._make_request('GET', '/v5/position/list', params=params)

        if response and response.get('retCode') == 0 and response.get('result') and 'list' in response['result']:
            return response['result']['list']
        else:
            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            ret_code = response.get('retCode', -1) if response else -1
            self.logger.error(f"Nie udało się pobrać pozycji dla {category}. Kod: {ret_code}, Błąd: {error_msg}. Odpowiedź: {response}")
            return []

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Pobiera informacje o tickerze dla danego symbolu.

        Args:
            symbol (str): Symbol pary handlowej (np. 'BTCUSDT').

        Returns:
            Optional[Dict]: Słownik z informacjami o tickerze lub None w przypadku błędu.
        """
        category = 'spot' if self.market_type == 'spot' else ('linear' if self.market_type == 'futures' else self.market_type)
        params = {
            'category': category,
            'symbol': symbol.upper()
        }

        response = self._make_request('GET', '/v5/market/tickers', params=params)

        if response and response.get('retCode') == 0 and response.get('result') and 'list' in response['result']:
            tickers = response['result']['list']
            if tickers:
                # Zwracamy pierwszy ticker z listy (powinien być tylko jeden dla danego symbolu)
                return tickers[0]
            else:
                self.logger.warning(f"Otrzymano pustą listę tickerów dla {symbol}.")
                return None
        else:
            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            ret_code = response.get('retCode', -1) if response else -1
            self.logger.error(f"Nie udało się pobrać tickera dla {symbol}. Kod: {ret_code}, Błąd: {error_msg}. Odpowiedź: {response}")
            return None

    def get_instrument_info(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Pobiera informacje o instrumentach (kontraktach, parach spot).

        Args:
            symbol (Optional[str]): Filtruj wg konkretnego symbolu.

        Returns:
            List[Dict]: Lista informacji o instrumentach lub pusta lista w przypadku błędu.
        """
        category = 'spot' if self.market_type == 'spot' else ('linear' if self.market_type == 'futures' else self.market_type)
        params = {
            'category': category,
            # Można dodać 'limit' i 'cursor' do paginacji
            # params['limit'] = 1000 # Max limit
        }
        if symbol:
            params['symbol'] = symbol.upper()

        response = self._make_request('GET', '/v5/market/instruments-info', params=params)

        if response and response.get('retCode') == 0 and response.get('result') and 'list' in response['result']:
            return response['result']['list']
        else:
            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            ret_code = response.get('retCode', -1) if response else -1
            self.logger.error(f"Nie udało się pobrać informacji o instrumentach dla {category}. Kod: {ret_code}, Błąd: {error_msg}. Odpowiedź: {response}")
            return []

    def check_api_key_permissions(self) -> Optional[Dict]:
        """
        Sprawdza uprawnienia bieżącego klucza API.

        Returns:
            Optional[Dict]: Słownik z informacjami o uprawnieniach lub None w przypadku błędu.
        """
        if not self._client or not self.api_key:
             self.logger.warning("Nie można sprawdzić uprawnień - brak klienta API lub klucza.")
             # Można spróbować zainicjować klienta, jeśli lazy_connect=True
             if self.lazy_connect and not self._client:
                  self._initialize_client()
                  if not self._client or not self.api_key:
                       return None # Nadal brak klienta lub klucza
             elif not self._client:
                  return None # Brak klienta i nie lazy_connect

        response = self._make_request('GET', '/v5/user/query-api') # Endpoint do sprawdzania kluczy

        if response and response.get('retCode') == 0 and response.get('result'):
            self.logger.info(f"Uprawnienia klucza API: {response['result']}")
            return response['result']
        else:
            error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
            ret_code = response.get('retCode', -1) if response else -1
            # Specyficzny błąd dla braku uprawnień do tego endpointu
            if ret_code == 10001 and "permission" in error_msg.lower():
                 self.logger.warning(f"Klucz API nie ma uprawnień do sprawdzania własnych uprawnień (endpoint /v5/user/query-api).")
                 return {"error": "Permission denied to query API key info."}
            else:
                 self.logger.error(f"Nie udało się sprawdzić uprawnień klucza API. Kod: {ret_code}, Błąd: {error_msg}. Odpowiedź: {response}")
                 return None

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None, max_retries: int = 3, backoff_factor: float = 0.5) -> Optional[Dict]:
        """
        Wykonuje żądanie do API Bybit z obsługą błędów i ponowień.

        Args:
            method (str): Metoda HTTP ('GET', 'POST', 'PUT', 'DELETE').
            endpoint (str): Endpoint API (np. '/v5/market/kline').
            params (Optional[Dict]): Parametry zapytania URL.
            data (Optional[Dict]): Ciało żądania (dla POST/PUT).
            max_retries (int): Maksymalna liczba ponowień.
            backoff_factor (float): Współczynnik opóźnienia między ponowieniami.

        Returns:
            Optional[Dict]: Odpowiedź API jako słownik lub None w przypadku błędu.
        """
        if not self._client:
            if not self.lazy_connect:
                self.logger.error("Klient API nie został zainicjowany.")
                return None
            else:
                self._initialize_client()
                if not self._client:
                    self.logger.error("Inicjalizacja klienta API nie powiodła się.")
                    return None

        request_func = getattr(self._client, method.lower(), None)
        if not request_func:
             self.logger.error(f"Nieprawidłowa metoda HTTP dla klienta pybit: {method}")
             return None

        for attempt in range(max_retries + 1):
            try:
                if method.upper() in ['GET', 'DELETE']:
                    response = request_func(endpoint, params=params or {})
                elif method.upper() in ['POST', 'PUT']:
                    response = request_func(endpoint, data=data or {})
                else:
                     self.logger.error(f"Nieobsługiwana metoda HTTP: {method}")
                     return None

                # Sprawdzenie kodu retencji i wiadomości
                ret_code = response.get('retCode', -1) # Domyślnie -1 jeśli brak retCode
                ret_msg = response.get('retMsg', 'Unknown error')
                result = response.get('result')

                if ret_code == 0:
                    self.consecutive_failures = 0 # Reset licznika błędów
                    return response # Zwróć całą odpowiedź, w tym metadane

                # Obsługa specyficznych błędów Bybit
                elif ret_code == 10002: # Request timeout
                    self.logger.warning(f"Timeout żądania ({attempt + 1}/{max_retries + 1}): {endpoint}. Ponawiam...")
                    if attempt == max_retries:
                        self.logger.error(f"Przekroczono limit czasu żądania po {max_retries + 1} próbach: {endpoint}")
                        raise TimeoutError(f"Request timed out: {endpoint}")
                    time.sleep(backoff_factor * (2 ** attempt) + random.uniform(0, 0.1)) # Exponential backoff with jitter
                    continue
                elif ret_code == 10006: # Rate limit exceeded
                    self.logger.warning(f"Przekroczono limit zapytań API ({attempt + 1}/{max_retries + 1}): {endpoint}. Czekam i ponawiam...")
                    if attempt == max_retries:
                        self.logger.error(f"Przekroczono limit zapytań API po {max_retries + 1} próbach: {endpoint}")
                        raise RateLimitExceeded(f"Rate limit exceeded for {endpoint}")
                    # Czekaj dłużej przy rate limit
                    wait_time = (backoff_factor * (2 ** attempt)) * 2 + random.uniform(0.1, 0.5)
                    self.logger.info(f"Czekam {wait_time:.2f}s przed ponowieniem...")
                    time.sleep(wait_time)
                    continue
                elif ret_code == 10001: # Invalid parameters
                    self.logger.error(f"Błąd API Bybit (Invalid Parameters): {ret_msg} dla {endpoint} z params={params}, data={data}")
                    return response # Zwróć odpowiedź z błędem, nie ponawiaj
                elif ret_code == 110007: # Insufficient balance
                     self.logger.error(f"Błąd API Bybit (Insufficient Balance): {ret_msg} dla {endpoint} z params={params}, data={data}")
                     return response # Zwróć odpowiedź z błędem, nie ponawiaj
                else:
                    # Inne błędy API Bybit
                    self.logger.error(f"Błąd API Bybit (kod: {ret_code}): {ret_msg} dla {endpoint} z params={params}, data={data}")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= 5:
                         self.logger.critical("Wystąpiło 5 kolejnych błędów API. Sprawdź status API lub konfigurację.")
                    # Dla niektórych błędów można zdecydować o nieponawianiu
                    return response # Zwróć odpowiedź z błędem

            except (requests.exceptions.RequestException, pybit.exceptions.FailedRequestError, pybit.exceptions.InvalidRequestError) as e:
                self.logger.warning(f"Błąd połączenia lub żądania ({attempt + 1}/{max_retries + 1}): {e}. Ponawiam...")
                if attempt == max_retries:
                    self.logger.error(f"Błąd sieciowy lub żądania po {max_retries + 1} próbach dla {endpoint}: {e}", exc_info=True)
                    self.consecutive_failures += 1
                    return None # Zwróć None po ostatniej próbie
                time.sleep(backoff_factor * (2 ** attempt) + random.uniform(0, 0.1)) # Exponential backoff with jitter

            except (RateLimitExceeded, TimeoutError) as e:
                 # Te wyjątki są już obsłużone logiką ponowień, ale łapiemy je, by nie wpadły w ogólny Exception
                 if attempt == max_retries:
                      self.logger.error(f"Ostateczny błąd po ponowieniach dla {endpoint}: {e}")
                      self.consecutive_failures += 1
                      return None
                 # Kontynuuj pętlę ponowień (już obsłużone wyżej)
                 continue

            except Exception as e:
                self.logger.error(f"Nieoczekiwany błąd podczas żądania do {endpoint} ({attempt + 1}/{max_retries + 1}): {e}", exc_info=True)
                if attempt == max_retries:
                    self.logger.error(f"Nieoczekiwany błąd po {max_retries + 1} próbach dla {endpoint}: {e}")
                    self.consecutive_failures += 1
                    return None # Zwróć None po ostatniej próbie
                time.sleep(backoff_factor * (2 ** attempt) + random.uniform(0, 0.1)) # Exponential backoff

        # Jeśli pętla zakończyła się bez zwrócenia wartości (co nie powinno się zdarzyć przy poprawnym przepływie)
        self.logger.error(f"Nie udało się uzyskać odpowiedzi dla {endpoint} po {max_retries + 1} próbach.")
        return None

    def _apply_rate_limit(self):
        """Ulepszona implementacja limitów API z dynamicznym dostosowaniem."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        # Dynamiczne dostosowanie czasu między zapytaniami
        if self.consecutive_failures > 0:
            self.min_time_between_calls *= 1.5  # Zwiększ czas między zapytaniami o 50%
            self.min_time_between_calls = min(self.min_time_between_calls, 30.0)  # Max 30s
        elif time_since_last_call > self.min_time_between_calls * 2:
            # Jeśli ostatnie zapytanie było znacznie później niż wymagane minimum,
            # możemy spróbować zmniejszyć opóźnienie
            self.min_time_between_calls = max(
                self.min_time_between_calls * 0.8,  # Zmniejsz o 20%
                3.0 if self.use_testnet else 10.0  # Nie schodź poniżej bezpiecznego minimum
            )

        if time_since_last_call < self.min_time_between_calls:
            sleep_time = self.min_time_between_calls - time_since_last_call
            self.logger.debug(f"Rate limit: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Reset stanu przekroczenia limitów jeśli minął okres kary
        if self.rate_limit_exceeded:
            if current_time - self.last_rate_limit_reset > self.rate_limit_reset_time:
                self.rate_limit_exceeded = False
                self.consecutive_failures = 0
                self.min_time_between_calls = 3.0 if self.use_testnet else 10.0
                self.last_rate_limit_reset = current_time
                self.logger.info("Reset stanu przekroczenia limitów API")

        self.last_api_call = time.time()

    def is_production_api(self):
        """Sprawdza czy używane jest produkcyjne API.

        Returns:
            bool: True jeśli używane jest produkcyjne API, False dla testnet.
        """
        # Sprawdzenie czy używamy produkcyjnego API
        is_prod = not self.use_testnet

        # Dodatkowy log dla produkcyjnego API (tylko jeśli nie wyświetlono wcześniej)
        if is_prod and not hasattr(self, '_production_warning_shown'):
            self.logger.warning(
                "!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit. Operacje handlowe będą mieć realne skutki finansowe!"
            )
            self.logger.warning(
                "Upewnij się, że Twoje klucze API mają właściwe ograniczenia i są odpowiednio zabezpieczone."
            )
            print("\n========== PRODUKCYJNE API BYBIT ==========")
            print("!!! UWAGA !!! Używasz PRODUKCYJNEGO API ByBit")
            print("Operacje handlowe będą mieć realne skutki finansowe!")
            print("===========================================\n")
            self._production_warning_shown = True
        elif not is_prod and not hasattr(self, '_testnet_info_shown'):
            self.logger.info("Używasz testnet API (środowisko testowe).")
            self._testnet_info_shown = True

        return is_prod

if __name__ == "__main__":
    # Przykład użycia
    connector = BybitConnector(api_key="test_key",
                               api_secret="test_secret",
                               use_testnet=True,
                               proxies={
                                   'http':'socks5h://127.0.0.1:1080',
                                   'https': 'socks5h://127.0.0.1:1080'
                               })

    server_time = connector.get_server_time()
    print(f"Czas serwera: {server_time}")

    klines = connector.get_klines(symbol="BTCUSDT")
    print(f"Pobrano {len(klines)} świec")

    order_book = connector.get_order_book(symbol="BTCUSDT")
    print(
        f"Pobrano książkę zleceń z {len(order_book['bids'])} ofertami kupna i {len(order_book['asks'])} ofertami sprzedaży"
    )

    balance = connector.get_account_balance()
    print(f"Stan konta: {balance}")

    order_result = connector.place_order(symbol="BTCUSDT",
                                         side="Buy",
                                         price=50000,
                                         quantity=0.01)
    print(f"Wynik złożenia zlecenia: {order_result}")

    order_status = connector.get_order_status(
        order_id=order_result.get("order_id"))
    print(f"Status zlecenia: {order_status}")

    cancel_result = connector.cancel_order(
        order_id=order_result.get("order_id"))
    print(f"Wynik anulowania zlecenia: {cancel_result}")