"""
bybit_connector.py
------------------
Moduł łączący się z API ByBit, umożliwiający autoryzację, pobieranie danych oraz wykonywanie zleceń.
Obsługuje zarówno testnet jak i mainnet, z zabezpieczeniami dla różnych środowisk.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Dict, Any, Optional
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bybit_connector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bybit_connector")


class BybitConnector:
    """
    Klasa BybitConnector zapewnia interfejs do łączenia się z giełdą ByBit.

    Parameters:
        use_testnet (bool): Jeśli True, używa API testnetu zamiast mainnet.
        api_key (str): Klucz API (domyślnie z .env).
        api_secret (str): Sekret API (domyślnie z .env).
    """

    def __init__(
        self,
        use_testnet: bool = False,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        self.use_testnet = use_testnet
        self.api_key = api_key or os.getenv("BYBIT_API_KEY")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("Klucz API ByBit i sekret są wymagane. Sprawdź zmienne środowiskowe.")

        # Bazowe URL w zależności od środowiska
        self.base_url = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"

        # Podstawowa sesja HTTP
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.api_key
        })

        logger.info(f"BybitConnector zainicjalizowany. Testnet: {use_testnet}")

    def _generate_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """
        Generuje podpis dla zapytania API ByBit.

        Parameters:
            params (dict): Parametry zapytania.
            timestamp (int): Znacznik czasu w milisekundach.

        Returns:
            str: Wygenerowany podpis.
        """
        param_str = ""
        if params:
            if isinstance(params, dict):
                param_str = urlencode(sorted(params.items()))
            else:
                param_str = params

        sign_str = f"{timestamp}{self.api_key}{param_str}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        private: bool = False
    ) -> Dict[str, Any]:
        """
        Wykonuje zapytanie HTTP do API ByBit.

        Parameters:
            method (str): Metoda HTTP ('GET', 'POST', etc.).
            endpoint (str): Endpoint API.
            params (dict): Parametry zapytania.
            private (bool): Jeśli True, dodaje wymagane nagłówki autoryzacyjne.

        Returns:
            dict: Odpowiedź z API.
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        headers = {}

        if private:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(params, timestamp)
            headers.update({
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": str(timestamp),
                "X-BAPI-API-KEY": self.api_key
            })

        MAX_RETRIES = 3
        RETRY_DELAY = 1

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, headers=headers, timeout=10)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=params, headers=headers, timeout=10)
                else:
                    raise ValueError(f"Nieobsługiwana metoda HTTP: {method}")

                response.raise_for_status()
                data = response.json()

                # Logowanie z maskowaniem wrażliwych danych
                masked_params = {k: ("***" if k in ["api_key", "timestamp", "sign"] else v) for k, v in params.items()}
                logger.info(f"Zapytanie {method} {endpoint} zakończone sukcesem. Parametry: {masked_params}")

                if data.get("ret_code") != 0:
                    logger.error(f"Błąd API ByBit: {data.get('ret_msg')}")
                    raise Exception(f"ByBit API Error: {data.get('ret_msg')}")

                return data
            except requests.exceptions.RequestException as e:
                logger.warning(f"Błąd zapytania (próba {attempt}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"Przekroczono maksymalną liczbę prób dla zapytania: {endpoint}")
                    raise
                time.sleep(RETRY_DELAY)

    def get_server_time(self) -> Dict[str, Any]:
        """
        Pobiera czas serwera ByBit.

        Returns:
            dict: Czas serwera.
        """
        return self._request("GET", "/v5/market/time")

    def get_klines(
        self,
        symbol: str,
        interval: str = "1",
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Pobiera dane świec (klines) dla określonej pary handlowej.

        Parameters:
            symbol (str): Symbol pary handlowej (np. "BTCUSDT").
            interval (str): Interwał czasowy (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M).
            limit (int): Maksymalna liczba zwracanych rekordów.
            start_time (int, optional): Początkowy timestamp w milisekundach.
            end_time (int, optional): Końcowy timestamp w milisekundach.

        Returns:
            dict: Dane świec.
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        return self._request("GET", "/v5/market/kline", params)

    def get_order_book(self, symbol: str, limit: int = 25) -> Dict[str, Any]:
        """
        Pobiera aktualny stan księgi zleceń dla określonej pary handlowej.

        Parameters:
            symbol (str): Symbol pary handlowej (np. "BTCUSDT").
            limit (int): Głębokość księgi zleceń (1-200).

        Returns:
            dict: Dane księgi zleceń.
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "limit": limit
        }
        return self._request("GET", "/v5/market/orderbook", params)

    def get_account_balance(self, account_type: str = "SPOT") -> Dict[str, Any]:
        """
        Pobiera stan konta.

        Parameters:
            account_type (str): Typ konta (SPOT, CONTRACT, UNIFIED).

        Returns:
            dict: Stan konta.
        """
        params = {"accountType": account_type}
        logger.info("Próba pobrania salda konta ByBit")
        try:
            result = self._request("GET", "/v5/account/wallet-balance", params, private=True)
            logger.info(f"Odpowiedź API ByBit: {result}")
            return result["result"]
        except Exception as e:
            logger.error(f"Błąd podczas pobierania salda konta: {e}")
            # Zwracamy pusty wynik zamiast podnosić wyjątek, aby aplikacja nadal działała
            return {"coins": [], "message": f"Błąd połączenia: {str(e)}"}


    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_on_trigger: bool = False
    ) -> Dict[str, Any]:
        """
        Składa zlecenie na giełdzie ByBit.

        Parameters:
            symbol (str): Symbol pary handlowej (np. "BTCUSDT").
            side (str): Kierunek zlecenia ("Buy" lub "Sell").
            order_type (str): Typ zlecenia ("Market" lub "Limit").
            qty (float): Ilość bazowej waluty.
            price (float, optional): Cena dla zleceń Limit.
            time_in_force (str): Okres ważności zlecenia (GTC, IOC, FOK).
            reduce_only (bool): Jeśli True, zlecenie tylko zmniejsza pozycję.
            close_on_trigger (bool): Jeśli True, zamyka pozycję po aktywacji stop loss/take profit.

        Returns:
            dict: Odpowiedź API dotycząca zlecenia.
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force
        }

        if order_type.lower() == "limit" and price is not None:
            params["price"] = str(price)

        if reduce_only:
            params["reduceOnly"] = reduce_only

        if close_on_trigger:
            params["closeOnTrigger"] = close_on_trigger

        return self._request("POST", "/v5/order/create", params, private=True)

    def cancel_order(self, symbol: str, order_id: str = None, order_link_id: str = None) -> Dict[str, Any]:
        """
        Anuluje zlecenie na giełdzie ByBit.

        Parameters:
            symbol (str): Symbol pary handlowej (np. "BTCUSDT").
            order_id (str, optional): ID zlecenia.
            order_link_id (str, optional): Niestandardowe ID zlecenia.

        Returns:
            dict: Odpowiedź API dotycząca anulowania zlecenia.
        """
        params = {
            "category": "spot",
            "symbol": symbol
        }

        if order_id:
            params["orderId"] = order_id
        elif order_link_id:
            params["orderLinkId"] = order_link_id
        else:
            raise ValueError("Musisz podać order_id lub order_link_id")

        return self._request("POST", "/v5/order/cancel", params, private=True)

    def get_open_orders(self, symbol: str = None, limit: int = 50) -> Dict[str, Any]:
        """
        Pobiera listę aktywnych zleceń.

        Parameters:
            symbol (str, optional): Symbol pary handlowej (np. "BTCUSDT").
            limit (int): Maksymalna liczba zwracanych rekordów.

        Returns:
            dict: Lista aktywnych zleceń.
        """
        params = {
            "category": "spot",
            "limit": limit
        }

        if symbol:
            params["symbol"] = symbol

        return self._request("GET", "/v5/order/realtime", params, private=True)

    def get_order_history(
        self,
        symbol: str = None,
        order_id: str = None,
        order_link_id: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Pobiera historię zleceń.

        Parameters:
            symbol (str, optional): Symbol pary handlowej (np. "BTCUSDT").
            order_id (str, optional): ID zlecenia.
            order_link_id (str, optional): Niestandardowe ID zlecenia.
            limit (int): Maksymalna liczba zwracanych rekordów.

        Returns:
            dict: Historia zleceń.
        """
        params = {
            "category": "spot",
            "limit": limit
        }

        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id

        return self._request("GET", "/v5/order/history", params, private=True)


# Przykładowe użycie
if __name__ == "__main__":
    try:
        # Inicjalizacja klienta (domyślnie używa kluczy z .env)
        bybit = BybitConnector(use_testnet=False)

        # Sprawdzenie połączenia - pobranie czasu serwera
        server_time = bybit.get_server_time()
        logger.info(f"Czas serwera ByBit: {server_time}")

        # Pobranie danych świec dla BTCUSDT
        klines = bybit.get_klines(symbol="BTCUSDT", interval="15", limit=10)
        logger.info(f"Ostatnie świece BTCUSDT: {klines['result']['list'][0]}")

        # Pobranie księgi zleceń
        order_book = bybit.get_order_book(symbol="BTCUSDT", limit=5)
        logger.info(f"Księga zleceń BTCUSDT (top 5): {order_book}")

        # UWAGA: Poniższe operacje wymagają środków na koncie i są zakomentowane
        # Sprawdzenie stanu konta
        # balance = bybit.get_account_balance()
        # logger.info(f"Stan konta: {balance}")

        # Złożenie zlecenia LIMIT
        # order = bybit.place_order(
        #     symbol="BTCUSDT",
        #     side="Buy",
        #     order_type="Limit",
        #     qty=0.001,
        #     price=20000.0
        # )
        # logger.info(f"Złożone zlecenie: {order}")

    except Exception as e:
        logger.error(f"Błąd w module bybit_connector.py: {e}")
        raise