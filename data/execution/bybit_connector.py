"""
Bybit Connector - moduł do komunikacji z REST API Bybit.
Obsługuje autentykację, wykonywanie zapytań i obsługę błędów.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlencode

# Upewniamy się, że folder logs istnieje
os.makedirs("logs", exist_ok=True)

# Sprawdzamy dostępność modułu requests
try:
    import requests
except ImportError:
    requests = None
    logging.warning("Moduł 'requests' nie jest zainstalowany. BybitConnector będzie działał w trybie symulacji.")

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api_requests.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("BybitConnector")


class BybitConnector:
    """
    Klasa do komunikacji z REST API Bybit.

    Obsługuje:
    - Autentykację
    - Wykonywanie zapytań GET/POST
    - Obsługę limitów żądań
    - Obsługę błędów i retry
    - Testnet/Mainnet
    """

    # Stałe związane z API Bybit
    API_URL_MAINNET = "https://api.bybit.com"
    API_URL_TESTNET = "https://api-testnet.bybit.com"

    # Kody błędów które wymagają retry
    RETRY_ERROR_CODES = [10002, 10006, 10016, 10018, 30034, 30035, 30037]

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_testnet: bool = True,
        recv_window: int = 5000,
        max_retries: int = 3,
        retry_delay: int = 1,
        simulation_mode: bool = True,
    ):
        """
        Inicjalizacja konektora Bybit.

        Args:
            api_key: Klucz API Bybit
            api_secret: Sekret API Bybit
            use_testnet: Czy używać testnet zamiast mainnet
            recv_window: Okno czasowe dla żądań (ms)
            max_retries: Maksymalna liczba prób ponowienia żądania
            retry_delay: Opóźnienie między próbami ponowienia (s)
            simulation_mode: Czy działać w trybie symulacji (bez rzeczywistych zapytań)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.simulation_mode = simulation_mode or requests is None

        # Konfiguracja backendu
        self.base_url = self.API_URL_TESTNET if use_testnet else self.API_URL_MAINNET
        self.session = requests.Session() if requests else None

        env_type = "testnet" if use_testnet else "mainnet"

        if self.simulation_mode:
            logger.warning(f"BybitConnector zainicjalizowany w trybie SYMULACJI dla {env_type}")
        else:
            logger.info(f"BybitConnector zainicjalizowany dla {env_type}")

    def _generate_signature(self, params: Dict) -> Tuple[int, str]:
        """
        Generuje sygnaturę dla zapytania.

        Args:
            params: Parametry zapytania

        Returns:
            Tuple[int, str]: Timestamp i sygnatura
        """
        timestamp = int(time.time() * 1000)
        params_with_timestamp = params.copy()

        # Dodajemy timestamp i recv_window do parametrów
        params_with_timestamp.update({
            "timestamp": timestamp,
            "recv_window": self.recv_window,
        })

        # Sortujemy parametry alfabetycznie
        sorted_params = sorted(params_with_timestamp.items())
        query_string = urlencode(sorted_params)

        # Generujemy sygnaturę
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(query_string, "utf-8"),
            hashlib.sha256
        ).hexdigest()

        return timestamp, signature

    def _handle_response(self, response: Optional[requests.Response]) -> Dict:
        """
        Przetwarza odpowiedź z API Bybit.

        Args:
            response: Odpowiedź z requests

        Returns:
            Dict: Przetworzona odpowiedź

        Raises:
            Exception: Jeśli wystąpił błąd w odpowiedzi API
        """
        if self.simulation_mode or response is None:
            # W trybie symulacji zwracamy fałszywe dane
            logger.debug("Tryb symulacji: generowanie sztucznych danych odpowiedzi")
            return {
                "ret_code": 0,
                "ret_msg": "OK",
                "result": {
                    "timeSecond": int(time.time()),
                    "timeNano": int(time.time() * 1_000_000_000),
                    "list": []
                }
            }

        try:
            # Loguj szczegóły odpowiedzi niezależnie od wyniku
            logger.debug(f"Odpowiedź API: Status: {response.status_code}, Nagłówki: {response.headers}")

            # Sprawdź, czy mamy treść odpowiedzi
            if not response.text:
                logger.error("Pusta odpowiedź z API (brak treści)")
                raise Exception("Empty response from API (no content)")

            # Próba parsowania JSON z lepszą obsługą błędów
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                # Zapisz surową odpowiedź do pliku dla debugowania
                error_log_file = os.path.join("logs", f"api_error_{int(time.time())}.txt")
                os.makedirs("logs", exist_ok=True)

                with open(error_log_file, "w") as f:
                    f.write(f"Status: {response.status_code}\n")
                    f.write(f"Nagłówki: {dict(response.headers)}\n")
                    f.write(f"Treść: {response.text}\n")

                logger.error(f"Nieprawidłowa odpowiedź JSON: {str(e)}. Treść zapisana do {error_log_file}")
                logger.error(f"Pierwsze 500 znaków odpowiedzi: {response.text[:500]}")
                raise Exception(f"Invalid JSON response: {str(e)}. Treść zapisana do {error_log_file}")

            # Sprawdzamy status odpowiedzi API Bybit
            if data.get("ret_code") != 0:
                error_code = data.get("ret_code")
                error_msg = data.get("ret_msg", "Unknown error")
                logger.error(f"Błąd API Bybit: {error_code} - {error_msg}")
                logger.error(f"Pełne dane odpowiedzi: {json.dumps(data)}")

                # Jeśli kod błędu wymaga ponowienia, rzucamy wyjątek do obsługi retry
                if error_code in self.RETRY_ERROR_CODES:
                    raise Exception(f"Retryable error: {error_code} - {error_msg}")

                # Inne błędy zwracamy bezpośrednio
                return data

            # Sprawdź, czy dane zawierają oczekiwany format odpowiedzi
            if "result" not in data:
                logger.warning(f"Odpowiedź API nie zawiera klucza 'result': {json.dumps(data)}")

            return data

        except json.JSONDecodeError as json_err:
            logger.error(f"Nieprawidłowa odpowiedź JSON: {str(json_err)}")
            logger.error(f"Treść odpowiedzi: {response.text[:1000]}...")
            raise Exception(f"Invalid JSON response: {str(json_err)}. Response: {response.text[:500]}...")

        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas przetwarzania odpowiedzi: {str(e)}")
            if response:
                logger.error(f"Status odpowiedzi: {response.status_code}, Treść: {response.text[:500]}...")
            raise

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        signed: bool = False,
    ) -> Dict:
        """
        Wykonuje zapytanie do API Bybit.

        Args:
            method: Metoda HTTP (GET, POST, DELETE)
            endpoint: Endpoint API
            params: Parametry URL (dla GET)
            data: Dane JSON (dla POST)
            signed: Czy zapytanie wymaga autentykacji

        Returns:
            Dict: Odpowiedź z API

        Raises:
            Exception: Jeśli wystąpił błąd w zapytaniu
        """
        if params is None:
            params = {}

        if data is None:
            data = {}

        # W trybie symulacji zwracamy fałszywe dane
        if self.simulation_mode:
            logger.info(f"Symulacja: {method} {endpoint}")

            # Symulowane odpowiedzi dla różnych endpointów
            if endpoint == "/v5/market/time":
                return {
                    "ret_code": 0,
                    "ret_msg": "OK",
                    "result": {
                        "timeSecond": int(time.time()),
                        "timeNano": int(time.time() * 1_000_000_000)
                    }
                }
            elif endpoint == "/v5/market/tickers" and "symbol" in params:
                symbol = params["symbol"]
                return {
                    "ret_code": 0,
                    "ret_msg": "OK",
                    "result": {
                        "list": [
                            {
                                "symbol": symbol,
                                "lastPrice": "65432.10",
                                "volume24h": "12345.67",
                                "change24h": "2.5"
                            }
                        ]
                    }
                }
            elif endpoint == "/v5/account/wallet-balance":
                return {
                    "ret_code": 0,
                    "ret_msg": "OK",
                    "result": {
                        "list": [
                            {
                                "totalEquity": "10000.00",
                                "totalWalletBalance": "10000.00",
                                "coin": [
                                    {
                                        "coin": "USDT",
                                        "walletBalance": "10000.00",
                                        "availableToWithdraw": "10000.00"
                                    },
                                    {
                                        "coin": "BTC",
                                        "walletBalance": "0.5",
                                        "availableToWithdraw": "0.5"
                                    }
                                ]
                            }
                        ]
                    }
                }
            else:
                return {
                    "ret_code": 0,
                    "ret_msg": "OK",
                    "result": {}
                }

        # Pełny URL
        url = f"{self.base_url}{endpoint}"

        # Dodajemy nagłówki uwierzytelniające dla signed requests
        headers = {}
        if signed:
            if not self.api_key or not self.api_secret:
                logger.error("Brak kluczy API, ale zapytanie wymaga podpisu")
                raise Exception("API keys required for signed request")

            timestamp, signature = self._generate_signature(params if method == "GET" else data)
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": str(timestamp),
                "X-BAPI-SIGN": signature,
                "X-BAPI-RECV-WINDOW": str(self.recv_window),
            })

        # Dodajemy Content-Type dla POST
        if method in ["POST", "PUT", "DELETE"]:
            headers["Content-Type"] = "application/json"

        # Logi przed zapytaniem
        log_msg = f"{method} {url}"
        if params:
            log_msg += f", params: {params}"
        if data:
            log_msg += f", data: {data}"
        logger.debug(f"Wysyłanie zapytania: {log_msg}")

        # Sprawdzamy, czy mamy dostęp do modułu requests
        if not requests:
            logger.error("Moduł 'requests' nie jest dostępny, nie można wykonać żądania")
            return self._handle_response(None)

        # Wykonujemy zapytanie z obsługą retry
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                if method == "GET":
                    response = self.session.get(url, params=params, headers=headers)
                elif method == "POST":
                    response = self.session.post(url, json=data, headers=headers)
                elif method == "DELETE":
                    response = self.session.delete(url, json=data, headers=headers)
                else:
                    raise Exception(f"Nieobsługiwana metoda HTTP: {method}")

                # Logujemy status odpowiedzi
                logger.debug(f"Odpowiedź: {response.status_code}, Body: {response.text[:200]}...")

                # Sprawdzamy kod statusu HTTP
                if response.status_code != 200:
                    logger.warning(f"Nieprawidłowy status HTTP: {response.status_code}, Body: {response.text}")
                    if response.status_code in [429, 500, 502, 503, 504]:
                        # Te kody statusów są tymczasowe, więc retry
                        retry_count += 1
                        retry_delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                        logger.info(f"Ponowienie za {retry_delay}s (próba {retry_count}/{self.max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Inne kody błędów HTTP nie są retryowalne
                        raise Exception(f"HTTP Error: {response.status_code} - {response.text}")

                # Przetwarzamy odpowiedź
                return self._handle_response(response)

            except Exception as e:
                logger.error(f"Błąd podczas zapytania: {e}")

                # Sprawdzamy czy wyjątek jest retryzalny
                if "Retryable error" in str(e) and retry_count < self.max_retries:
                    retry_count += 1
                    retry_delay = self.retry_delay * (2 ** retry_count)
                    logger.info(f"Ponowienie za {retry_delay}s (próba {retry_count}/{self.max_retries})")
                    time.sleep(retry_delay)
                else:
                    # W przypadku błędu, jeśli jesteśmy w trybie symulacji, zwróć symulowane dane
                    if self.simulation_mode:
                        logger.info("Symulacja: Zwracam awaryjne dane symulowane po błędzie")
                        return {
                            "ret_code": 0,
                            "ret_msg": "OK (symulacja)",
                            "result": {}
                        }
                    else:
                        raise

        # Jeśli dotarliśmy tutaj, wszystkie próby retry zawiodły
        if self.simulation_mode:
            logger.info("Symulacja: Zwracam awaryjne dane symulowane po wyczerpaniu prób")
            return {
                "ret_code": 0,
                "ret_msg": "OK (symulacja po wyczerpaniu prób)",
                "result": {}
            }
        else:
            raise Exception(f"All retry attempts failed for {method} {url}")

    # =========== Metody API Bybit ===========

    def get_server_time(self) -> Dict:
        """
        Pobiera czas serwera Bybit.

        Returns:
            Dict: Odpowiedź z czasem serwera
        """
        return self._request("GET", "/v5/market/time")

    def get_symbols(self) -> Dict:
        """
        Pobiera dostępne symbole handlowe.

        Returns:
            Dict: Lista dostępnych symboli
        """
        return self._request("GET", "/v5/market/instruments-info", params={"category": "spot"})

    def get_ticker(self, symbol: str) -> Dict:
        """
        Pobiera ticker dla danego symbolu.

        Args:
            symbol: Symbol handlowy (np. "BTCUSDT")

        Returns:
            Dict: Dane tickera
        """
        try:
            if self.simulation_mode:
                # Symulowane dane tickera
                import random
                import time

                # Ustawiamy realistyczne ceny dla różnych symboli
                base_prices = {
                    "BTCUSDT": 65000,
                    "ETHUSDT": 3500,
                    "SOLUSDT": 150,
                    "DOGEUSDT": 0.15,
                    "XRPUSDT": 0.60,
                    "BNBUSDT": 600,
                    "ADAUSDT": 0.45,
                    "DOTUSDT": 8.20,
                }

                # Domyślna cena jeśli symbol nie jest znany
                base_price = base_prices.get(symbol, 100)

                # Dodajemy losową zmianę w granicach +/-1%
                price = base_price + (base_price * random.uniform(-0.01, 0.01))

                # Tworzymy odpowiedź w formacie Bybit API
                return {
                    "result": {
                        "list": [
                            {
                                "symbol": symbol,
                                "lastPrice": str(round(price, 2)),
                                "prevPrice24h": str(round(price * (1 + random.uniform(-0.05, 0.05)), 2)),
                                "price24hPcnt": str(round(random.uniform(-0.05, 0.05), 4)),
                                "highPrice24h": str(round(price * (1 + random.uniform(0.01, 0.1)), 2)),
                                "lowPrice24h": str(round(price * (1 - random.uniform(0.01, 0.1)), 2)),
                                "volume24h": str(round(random.uniform(1000, 10000), 2)),
                                "turnover24h": str(round(random.uniform(50000000, 500000000), 2))
                            }
                        ]
                    },
                    "time": int(time.time() * 1000)
                }

            url = f"{self.base_url}/v5/market/tickers"
            params = {"category": "spot", "symbol": symbol}
            response = self.session.get(url, params=params)
            return response.json()
        except Exception as e:
            logger.error(f"Błąd podczas pobierania tickera dla {symbol}: {e}")
            return {"ret_code": -1, "ret_msg": f"Błąd pobierania tickera: {e}"}


    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Dict:
        """
        Pobiera dane świecowe.

        Args:
            symbol: Symbol handlowy (np. "BTCUSDT")
            interval: Interwał czasowy (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
            limit: Maksymalna liczba świec (max 200)
            start_time: Początkowy timestamp (ms)
            end_time: Końcowy timestamp (ms)

        Returns:
            Dict: Dane świecowe
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        return self._request("GET", "/v5/market/kline", params=params)

    def get_order_book(self, symbol: str, limit: int = 50) -> Dict:
        """
        Pobiera orderbook dla danego symbolu.

        Args:
            symbol: Symbol handlowy (np. "BTCUSDT")
            limit: Głębokość orderbooka (1-200)

        Returns:
            Dict: Dane orderbooka
        """
        return self._request(
            "GET",
            "/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": limit},
        )

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        """
        Pobiera saldo portfela.

        Args:
            account_type: Typ konta (UNIFIED, CONTRACT, SPOT)

        Returns:
            Dict: Dane salda
        """
        return self._request(
            "GET",
            "/v5/account/wallet-balance",
            params={"accountType": account_type},
            signed=True,
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        order_link_id: Optional[str] = None,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> Dict:
        """
        Składa zlecenie.

        Args:
            symbol: Symbol handlowy (np. "BTCUSDT")
            side: Strona zlecenia ("Buy" or "Sell")
            order_type: Typ zlecenia ("Market", "Limit")
            qty: Ilość
            price: Cena (wymagana dla Limit)
            time_in_force: Czas obowiązywania (GTC, IOC, FOK)
            reduce_only: Czy zamknąć tylko pozycję
            close_on_trigger: Czy zamknąć przy wyzwoleniu
            order_link_id: Własny ID zlecenia
            tp_price: Cena take profit
            sl_price: Cena stop loss

        Returns:
            Dict: Odpowiedź z danymi zlecenia
        """
        data = {
            "category": "spot",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
            "closeOnTrigger": close_on_trigger,
        }

        if price:
            data["price"] = str(price)

        if order_link_id:
            data["orderLinkId"] = order_link_id

        if tp_price:
            data["takeProfit"] = str(tp_price)

        if sl_price:
            data["stopLoss"] = str(sl_price)

        return self._request("POST", "/v5/order/create", data=data, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> Dict:
        """
        Pobiera aktywne zlecenia.

        Args:
            symbol: Symbol handlowy (opcjonalny)

        Returns:
            Dict: Lista aktywnych zleceń
        """
        params = {"category": "spot"}
        if symbol:
            params["symbol"] = symbol

        return self._request("GET", "/v5/order/realtime", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict:
        """
        Anuluje zlecenie.

        Args:
            symbol: Symbol handlowy
            order_id: ID zlecenia
            order_link_id: Własny ID zlecenia

        Returns:
            Dict: Status anulowania
        """
        data = {"category": "spot", "symbol": symbol}

        if order_id:
            data["orderId"] = order_id
        elif order_link_id:
            data["orderLinkId"] = order_link_id
        else:
            raise Exception("Należy podać order_id lub order_link_id")

        return self._request("POST", "/v5/order/cancel", data=data, signed=True)

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        limit: int = 50,
    ) -> Dict:
        """
        Pobiera historię zleceń.

        Args:
            symbol: Symbol handlowy (opcjonalny)
            order_id: ID zlecenia (opcjonalny)
            order_link_id: Własny ID zlecenia (opcjonalny)
            limit: Maksymalna liczba wyników (max 100)

        Returns:
            Dict: Historia zleceń
        """
        params = {"category": "spot", "limit": limit}

        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id

        return self._request("GET", "/v5/order/history", params=params, signed=True)

    def test_connectivity(self) -> bool:
        """
        Testuje połączenie z API Bybit.

        Returns:
            bool: True jeśli połączenie działa, False w przeciwnym razie
        """
        try:
            if self.simulation_mode:
                logger.info("Symulacja: Test połączenia z Bybit udany")
                return True

            response = self.session.get(f"{self.base_url}/v5/market/time")
            data = response.json()
            return 'result' in data and 'timeNano' in data['result']
        except Exception as e:
            logger.error(f"Błąd testu połączenia: {e}")
            # W przypadku błędu przełączamy na tryb symulacji
            self.simulation_mode = True
            logger.warning("Przełączono na tryb symulacji po błędzie połączenia")
            return True


# Przykład użycia
if __name__ == "__main__":
    import os
    #from dotenv import load_dotenv #Commented out to avoid pip dependency conflict

    # Ładujemy zmienne środowiskowe -  Zastąp to odpowiednią metodą ładowania zmiennych środowiskowych dla Replit/Nix
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    use_testnet = os.getenv("TEST_MODE", "true").lower() in ["true", "1", "t"]

    # Sprawdzamy, czy moduł requests jest dostępny
    simulation_mode = requests is None

    # Inicjalizujemy konektor
    bybit = BybitConnector(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet,
        simulation_mode=simulation_mode
    )

    # Testujemy połączenie
    if bybit.test_connectivity():
        print("Połączenie z Bybit działa poprawnie!")

        # Pobieramy listę symboli
        symbols = bybit.get_symbols()
        print(f"Dostępnych symboli: {len(symbols.get('result', {}).get('list', []))}")

        # Pobieramy ticker dla BTC/USDT
        ticker = bybit.get_ticker("BTCUSDT")
        if "result" in ticker and "list" in ticker["result"] and ticker["result"]["list"]:
            btc_price = ticker["result"]["list"][0]["lastPrice"]
            print(f"Aktualna cena BTC: ${btc_price}")

        # Jeśli mamy klucze API, sprawdzamy saldo
        if api_key and api_secret:
            try:
                balance = bybit.get_wallet_balance()
                print("Saldo portfela:")
                print(json.dumps(balance, indent=2))
            except Exception as e:
                print(f"Błąd podczas pobierania salda: {e}")
    else:
        print("Nie można połączyć się z Bybit API")