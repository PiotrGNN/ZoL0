"""
bybit_connector.py
------------------
Moduł do komunikacji z API ByBit.
"""

import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bybit_connector.log", mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BybitConnector:
    """Connector do API ByBit."""

    def __init__(self, api_key: str, api_secret: str, use_testnet: bool = True):
        """
        Inicjalizacja connectora ByBit.

        Parameters:
            api_key (str): Klucz API ByBit.
            api_secret (str): Sekret API ByBit.
            use_testnet (bool): Czy używać testnetu.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.base_url = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"
        logger.info(f"Zainicjalizowano connector ByBit (testnet: {use_testnet})")

    def get_server_time(self) -> Dict[str, Any]:
        """
        Pobiera czas serwera ByBit.

        Returns:
            Dict[str, Any]: Odpowiedź z czasem serwera.
        """
        # Symulacja odpowiedzi API
        server_time = int(time.time() * 1000)
        return {
            "success": True,
            "time": server_time,
            "time_formatted": datetime.fromtimestamp(server_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_klines(self, symbol: str, interval: str = "15", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Pobiera świece (klines) dla danego symbolu.

        Parameters:
            symbol (str): Symbol pary walutowej.
            interval (str): Interwał czasowy.
            limit (int): Limit wyników.

        Returns:
            List[Dict[str, Any]]: Lista świec.
        """
        # Symulacja odpowiedzi API
        klines = []
        current_time = int(time.time())
        interval_seconds = int(interval) * 60

        starting_price = random.uniform(20000, 60000)

        for i in range(limit):
            open_price = starting_price * (1 + random.uniform(-0.01, 0.01))
            high_price = open_price * (1 + random.uniform(0, 0.02))
            low_price = open_price * (1 - random.uniform(0, 0.02))
            close_price = open_price * (1 + random.uniform(-0.01, 0.01))
            volume = random.uniform(1, 100)

            kline = {
                "timestamp": current_time - (limit - i - 1) * interval_seconds,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            }
            klines.append(kline)

            starting_price = close_price

        return klines

    def get_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """
        Pobiera książkę zleceń dla danego symbolu.

        Parameters:
            symbol (str): Symbol pary walutowej.
            limit (int): Limit wyników.

        Returns:
            Dict[str, Any]: Książka zleceń.
        """
        # Symulacja odpowiedzi API
        base_price = random.uniform(20000, 60000)

        bids = []
        asks = []

        for i in range(limit):
            bid_price = base_price * (1 - 0.001 * (i + 1))
            bid_volume = random.uniform(0.1, 10)
            bids.append([bid_price, bid_volume])

            ask_price = base_price * (1 + 0.001 * (i + 1))
            ask_volume = random.uniform(0.1, 10)
            asks.append([ask_price, ask_volume])

        return {
            "timestamp": int(time.time() * 1000),
            "symbol": symbol,
            "bids": bids,
            "asks": asks
        }

    def get_account_balance(self) -> Dict[str, Any]:
        """
        Pobiera stan konta.

        Returns:
            Dict[str, Any]: Stan konta.
        """
        # Symulacja odpowiedzi API
        balances = {
            "BTC": {
                "equity": round(random.uniform(0.001, 0.1), 8),
                "available_balance": round(random.uniform(0.001, 0.1), 8),
                "wallet_balance": round(random.uniform(0.001, 0.1), 8)
            },
            "USDT": {
                "equity": round(random.uniform(1000, 10000), 2),
                "available_balance": round(random.uniform(1000, 10000), 2),
                "wallet_balance": round(random.uniform(1000, 10000), 2)
            },
            "ETH": {
                "equity": round(random.uniform(0.1, 10), 4),
                "available_balance": round(random.uniform(0.1, 10), 4),
                "wallet_balance": round(random.uniform(0.1, 10), 4)
            }
        }

        return {
            "success": True,
            "balances": balances,
            "timestamp": int(time.time() * 1000)
        }

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generuje podpis dla żądania API."""
        param_str = urlencode(sorted(params.items()))
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(param_str, "utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _send_request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Wysyła żądanie do API ByBit."""
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        params['timestamp'] = str(int(time.time() * 1000))
        signature = self._generate_signature(params)
        params['sign'] = signature
        
        logger.info(f"Wysyłanie żądania {method} do {endpoint}")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if method == "GET":
                    response = requests.get(url, params=params, timeout=10)
                elif method == "POST":
                    response = requests.post(url, json=params, timeout=10)
                else:
                    raise ValueError(f"Nieobsługiwana metoda HTTP: {method}")
                
                response.raise_for_status()  # Rzuca wyjątek dla błędnych statusów
                
                result = response.json()
                
                if result.get("ret_code") != 0:
                    error_msg = result.get('ret_msg', 'Nieznany błąd API')
                    logger.error(f"Błąd API: {error_msg}")
                    
                    # Sprawdź, czy błąd jest tymczasowy i można powtórzyć
                    if "rate limit" in error_msg.lower() or "timeout" in error_msg.lower():
                        retry_count += 1
                        time.sleep(2 ** retry_count)  # Wykładniczy backoff
                        continue
                        
                    raise Exception(f"Błąd API ByBit: {error_msg}")
                
                return result.get("result", {})
            
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logger.error(f"Błąd podczas wysyłania żądania (próba {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.error("Przekroczono maksymalną liczbę prób połączenia")
                    raise
                    
                # Opóźnienie przed ponowną próbą (wykładniczy backoff)
                time.sleep(2 ** retry_count)
        
        # Ten kod nie powinien być osiągalny, ale dla pewności
        raise Exception("Nieoczekiwany błąd podczas wysyłania żądania API")

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, time_in_force: str = "GoodTillCancel") -> Dict[str, Any]:
        """Składa zlecenie na giełdzie."""
        #MOCK IMPLEMENTATION
        return {"order_id": random.randint(1000, 9999), "symbol": symbol, "side": side, "status": "open"}

    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Pobiera listę otwartych zleceń."""
        #MOCK IMPLEMENTATION
        return []

    def cancel_order(self, order_id: str = None, symbol: str = None) -> Dict[str, Any]:
        """Anuluje zlecenie."""
        #MOCK IMPLEMENTATION
        return {"order_id": order_id, "status": "cancelled"}

    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Pobiera informacje o otwartych pozycjach."""
        #MOCK IMPLEMENTATION
        return []


import os
import requests
import hmac
import hashlib
from urllib.parse import urlencode

if __name__ == "__main__":
    # Przykład użycia
    connector = BybitConnector(
        api_key="test_key",
        api_secret="test_secret",
        use_testnet=True
    )

    server_time = connector.get_server_time()
    print(f"Czas serwera: {server_time['time_formatted']}")

    klines = connector.get_klines(symbol="BTCUSDT")
    print(f"Pobrano {len(klines)} świec")

    order_book = connector.get_order_book(symbol="BTCUSDT")
    print(f"Pobrano książkę zleceń z {len(order_book['bids'])} ofertami kupna i {len(order_book['asks'])} ofertami sprzedaży")

    balance = connector.get_account_balance()
    print(f"Stan konta: BTC={balance['balances']['BTC']['equity']}, USDT={balance['balances']['USDT']['equity']}")
"""
bybit_connector.py
-----------------
Moduł do komunikacji z API Bybit.
"""

import logging
import time
import requests
from datetime import datetime

class BybitConnector:
    """
    Klasa do komunikacji z API giełdy Bybit.
    """
    
    def __init__(self, api_key=None, api_secret=None, use_testnet=True):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        
        if use_testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
            
        self.logger.info(f"Initialized ByBit connector. Testnet: {use_testnet}")
        
    def get_server_time(self):
        """
        Pobiera czas serwera Bybit.
        
        Returns:
            dict: Odpowiedź z czasem serwera
        """
        try:
            response = requests.get(f"{self.base_url}/v2/public/time")
            data = response.json()
            self.logger.info(f"Pobrano czas serwera ByBit: {data}")
            
            # Symulacja odpowiedzi, jeśli API nie odpowiada
            if not data.get("time_now"):
                return {
                    "success": True,
                    "time": int(time.time()),
                    "timeISO": datetime.now().isoformat(),
                    "note": "Symulowane dane"
                }
                
            return {
                "success": True,
                "time": data.get("time_now"),
                "timeISO": datetime.fromtimestamp(float(data.get("time_now"))).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania czasu serwera ByBit: {e}")
            return {
                "success": False,
                "error": str(e),
                "time": int(time.time()),
                "timeISO": datetime.now().isoformat(),
                "note": "Symulowane dane z powodu błędu"
            }
            
    def get_account_balance(self):
        """
        Pobiera stan konta Bybit.
        
        Returns:
            dict: Stan konta użytkownika
        """
        # Symulowany wynik, ponieważ prawdziwe API wymaga autentykacji
        return {
            "success": True,
            "balances": {
                "BTC": {
                    "equity": 0.01,
                    "available_balance": 0.01,
                    "wallet_balance": 0.01
                },
                "USDT": {
                    "equity": 1000,
                    "available_balance": 950,
                    "wallet_balance": 1000
                }
            },
            "note": "Symulowane dane"
        }
        
    def get_klines(self, symbol, interval="15", limit=10):
        """
        Pobiera dane świecowe dla danego symbolu.
        
        Args:
            symbol (str): Symbol pary walutowej
            interval (str, optional): Interwał czasowy. Defaults to "15".
            limit (int, optional): Limit danych. Defaults to 10.
            
        Returns:
            list: Lista danych świecowych
        """
        # Symulowane dane świecowe
        klines = []
        current_time = int(time.time())
        
        for i in range(limit):
            kline = {
                "timestamp": current_time - (i * int(interval) * 60),
                "open": 30000 + (i * 100),
                "high": 30100 + (i * 100),
                "low": 29950 + (i * 100),
                "close": 30050 + (i * 100),
                "volume": 10 + (i * 2)
            }
            klines.append(kline)
            
        return klines
        
    def get_order_book(self, symbol, limit=5):
        """
        Pobiera księgę zleceń dla danego symbolu.
        
        Args:
            symbol (str): Symbol pary walutowej
            limit (int, optional): Limit danych. Defaults to 5.
            
        Returns:
            dict: Księga zleceń
        """
        # Symulowane dane księgi zleceń
        base_price = 30000
        
        bids = []
        asks = []
        
        for i in range(limit):
            bids.append([base_price - (i * 10), 1 - (i * 0.1)])
            asks.append([base_price + (i * 10), 1 - (i * 0.1)])
            
        return {
            "bids": bids,
            "asks": asks,
            "time": int(time.time())
        }
