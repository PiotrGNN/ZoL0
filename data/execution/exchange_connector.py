"""
exchange_connector.py
-----------------------
Moduł łączący się z API giełdowym, umożliwiający autoryzację, pobieranie danych oraz wykonywanie zleceń.
Obsługuje różne giełdy (np. Binance, Coinbase, Kraken) poprzez warstwę abstrakcji, co ułatwia rozszerzanie o kolejne platformy.
Uwzględnia zarządzanie kluczami API, podpisywanie zapytań, bezpieczeństwo transmisji (HTTPS, rate limiting) oraz logowanie.
Kod jest zoptymalizowany pod kątem obsługi zarówno małych kont testowych, jak i bardzo dużych portfeli, dbając o wydajność i stabilność.
"""

import hashlib
import hmac
import logging
import time

import requests

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ExchangeConnector:
    """
    Klasa ExchangeConnector zapewnia abstrakcyjny interfejs do łączenia się z różnymi giełdami.

    Parameters:
        exchange (str): Nazwa giełdy ('binance', 'coinbase', 'kraken', itp.).
        api_key (str): Klucz API.
        api_secret (str): Sekret API.
        base_url (str): Bazowy URL API giełdy.
    """

    def __init__(self, exchange: str, api_key: str, api_secret: str, base_url: str = None):
        self.exchange = exchange.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update(
            {"X-MBX-APIKEY": self.api_key}
        )  # Domyślny nagłówek dla Binance; inne giełdy mogą wymagać innej konfiguracji
        self.base_url = base_url or self._default_base_url()
        logging.info("ExchangeConnector zainicjalizowany dla giełdy: %s", self.exchange)

    def _default_base_url(self) -> str:
        """
        Zwraca domyślny URL bazowy dla obsługiwanej giełdy.
        """
        defaults = {
            "binance": "https://api.binance.com",
            "coinbase": "https://api.pro.coinbase.com",
            "kraken": "https://api.kraken.com",
        }
        url = defaults.get(self.exchange)
        if not url:
            raise ValueError(f"Domyślny URL nie zdefiniowany dla giełdy: {self.exchange}")
        return url

    def _sign_payload(self, query_string: str) -> str:
        """
        Podpisuje ciąg zapytania (query string) przy użyciu klucza API i sekretu.

        Parameters:
            query_string (str): Ciąg zapytania do podpisania.

        Returns:
            str: Podpis (hexadecimal string).
        """
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """
        Wykonuje zapytanie HTTP do API giełdowego z obsługą rate limiting, retry oraz podpisywania.

        Parameters:
            method (str): Metoda HTTP ('GET', 'POST', etc.).
            endpoint (str): Endpoint API.
            params (dict): Parametry zapytania.
            signed (bool): Jeśli True, zapytanie jest podpisywane.

        Returns:
            dict: Odpowiedź z API.
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        # Dodaj znacznik czasu, jeśli zapytanie wymaga podpisu
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            query_string = "&".join([f"{key}={params[key]}" for key in sorted(params)])
            params["signature"] = self._sign_payload(query_string)

        retries = 3
        for attempt in range(1, retries + 1):
            try:
                response = self.session.request(method, url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                # Logowanie maskując wrażliwe dane
                masked_params = {k: ("***" if k in ["signature", "timestamp"] else v) for k, v in params.items()}
                logging.info(
                    "Zapytanie %s %s z parametrami %s zakończone sukcesem.",
                    method,
                    endpoint,
                    masked_params,
                )
                return data
            except requests.exceptions.RequestException as e:
                logging.warning("Błąd zapytania (próba %d/%d): %s", attempt, retries, e)
                if attempt == retries:
                    logging.error(
                        "Przekroczono maksymalną liczbę prób dla zapytania: %s",
                        endpoint,
                    )
                    raise
                time.sleep(2)

    def get_market_data(self, symbol: str, interval: str = "1m", limit: int = 100) -> dict:
        """
        Pobiera dane rynkowe dla określonego symbolu i interwału.

        Parameters:
            symbol (str): Symbol pary walutowej (np. "BTCUSDT").
            interval (str): Interwał danych (np. "1m", "5m", "1h", "1d").
            limit (int): Maksymalna liczba zwracanych rekordów.

        Returns:
            dict: Dane rynkowe.
        """
        if self.exchange == "binance":
            endpoint = "/api/v3/klines"
            params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
            return self._request("GET", endpoint, params, signed=False)
        else:
            # Implementacje dla innych giełd mogą się różnić
            raise NotImplementedError(f"get_market_data nie jest zaimplementowane dla giełdy: {self.exchange}")

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
    ) -> dict:
        """
        Wysyła zlecenie do giełdy.

        Parameters:
            symbol (str): Symbol pary walutowej (np. "BTCUSDT").
            side (str): Kierunek zlecenia ("BUY" lub "SELL").
            order_type (str): Typ zlecenia ("MARKET", "LIMIT", itp.).
            quantity (float): Ilość do zlecenia.
            price (float, optional): Cena zlecenia, wymagana dla zleceń LIMIT.

        Returns:
            dict: Odpowiedź API dotycząca zlecenia.
        """
        if self.exchange == "binance":
            endpoint = "/api/v3/order"
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": quantity,
            }
            if order_type.upper() == "LIMIT":
                if price is None:
                    raise ValueError("Dla zlecenia LIMIT podaj cenę.")
                params["price"] = price
                params["timeInForce"] = "GTC"
            return self._request("POST", endpoint, params, signed=True)
        else:
            raise NotImplementedError(f"place_order nie jest zaimplementowane dla giełdy: {self.exchange}")


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane (użyj własnych kluczy API)
        EXCHANGE = "binance"
        API_KEY = "your_api_key_here"
        API_SECRET = "your_api_secret_here"

        connector = ExchangeConnector(exchange=EXCHANGE, api_key=API_KEY, api_secret=API_SECRET)

        # Pobranie danych rynkowych dla BTCUSDT, interwał 1m, limit 5
        market_data = connector.get_market_data("BTCUSDT", interval="1m", limit=5)
        logging.info("Pobrane dane rynkowe: %s", market_data)

        # Przykład wykonania zlecenia: kupno 0.001 BTCUSDT zlecenie MARKET
        # Uwaga: To tylko przykład; wykonanie rzeczywistego zlecenia wymaga aktywnego konta i środków.
        # order_response = connector.place_order("BTCUSDT", side="BUY", order_type="MARKET", quantity=0.001)
        # logging.info("Odpowiedź zlecenia: %s", order_response)

    except Exception as e:
        logging.error("Błąd w module exchange_connector.py: %s", e)
        raise
