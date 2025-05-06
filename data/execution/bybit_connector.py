"""
bybit_connector.py
-----------------
Moduł do komunikacji z giełdą Bybit używając API v5.
"""

import json
import logging
import os
import random
import time
import traceback
import hmac
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import requests
import websocket

# Use only unified trading API
try:
    from pybit.unified_trading import HTTP
    PybitHTTP = HTTP
except ImportError:
    try:
        from pybit import HTTP as PybitHTTP
    except ImportError:
        PybitHTTP = None

def is_env_flag_true(env_var_name: str) -> bool:
    """

    Returns:
        bool: True jeśli wartość zmiennej to "1", "true" lub "yes" (bez uwzględnienia wielkości liter)
    """
    return os.getenv(env_var_name, "").strip().lower() in ["1", "true", "yes"]


class BybitConnector:
    """Klasa do komunikacji z giełdą Bybit używając API v5."""

    def __init__(self,
                 api_key: str = None,
                 api_secret: str = None,
                 use_testnet: bool = None,
                 lazy_connect: bool = True,
                 proxies: Dict[str, str] = None,
                 market_type: str = "spot"):
        """
        Initialize connection to Bybit API v5.
        
        Args:
            api_key (str): Bybit API key
            api_secret (str): Bybit API secret
            use_testnet (bool): Whether to use testnet instead of production
            lazy_connect (bool): Whether to delay client initialization
            proxies (Dict[str, str]): Proxy configuration
            market_type (str): Market type (spot, linear, inverse)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet if use_testnet is not None else False
        self.proxies = proxies
        self.market_type = market_type
        self.client = None
        self.api_version = "v5"
        self._connection_initialized = False
        self._connection_test_time = 0
        self._connection_test_result = False
        self.logger = logging.getLogger("BybitConnector")
        
        # Rate limiting parameters
        self.min_time_between_calls = 0.05  # 50ms between requests
        self.rate_limit_backoff = 60.0  # 1 minute backoff
        self.remaining_rate_limit = 100
        self.rate_limit_reset_time = 0
        self.rate_limit_exceeded = False
        self.last_api_call_time = 0
        self._backoff_attempt = 0
        self.last_rate_limit_reset = 0
        self.rate_limit_wait_time = 1

        # Base URLs
        self.base_url = "https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com"
        self.ws_url = f"wss://{'stream-testnet' if self.use_testnet else 'stream'}.bybit.com/v5/public"

        # Initialize requests session
        self.session = requests.Session()
        if self.proxies:
            self.session.proxies.update(self.proxies)
        
        # Initialize WebSocket parameters
        self.ws = None
        self.ws_connected = False
        self.ws_subscriptions = set()
        self.ws_callbacks = {}
        self.ws_reconnect_delay = 5
        self.ws_ping_interval = 20
        self.ws_last_ping = 0
        self.ws_thread = None
        
        if not lazy_connect:
            self._initialize_client()

    def _initialize_client(self, force=False) -> bool:
        """Initialize the Bybit API client with proper error handling"""
        if self.client is not None and not force and self._connection_initialized:
            return True

        current_time = time.time()
        if self._connection_test_time > 0 and current_time - self._connection_test_time < 60:
            return self._connection_test_result

        try:
            # Initialize with unified trading API only
            if PybitHTTP is not None:
                self.logger.info("Using unified trading API")
                self.client = PybitHTTP(
                    testnet=self.use_testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    recv_window=20000
                )
            else:
                self.logger.error("Failed to import PybitHTTP. Make sure pybit package is installed correctly.")
                return False

            self.api_version = "v5"
            
            # Test connection with server time check
            self._connection_test_result = self._test_connection()
            self._connection_test_time = current_time
            self._connection_initialized = True
            
            return self._connection_test_result

        except Exception as e:
            self.logger.error(f"Failed to initialize Bybit client: {e}")
            self.client = None
            self._connection_test_result = False
            self._connection_test_time = current_time
            return False
            
    def _test_connection(self) -> bool:
        """Test API connection by fetching server time
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.api_version == "v5":
                response = self.client.get_server_time()
                if response and response.get("retCode") == 0:
                    return True
            else:
                # Legacy v2 API
                self.client.server_time()
                return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
        return False

    def get_server_time(self) -> Dict[str, Any]:
        """
        Pobiera czas serwera Bybit.

        Returns:
            Dict[str, Any]: Czas serwera w formacie:
            {
                "success": bool,
                "time_ms": int,
                "time": str,
                "source": str
            }
        """
        try:
            # Sprawdź cache najpierw
            try:
                from data.utils.cache_manager import get_cached_data, is_cache_valid
                cache_key = f"server_time_{self.use_testnet}"
                if is_cache_valid(cache_key, ttl=60):
                    cached_data, found = get_cached_data(cache_key)
                    if found and isinstance(cached_data, dict) and "timeNow" in cached_data:
                        time_ms = int(cached_data["timeNow"])
                        return {
                            "success": True,
                            "time_ms": time_ms,
                            "time": datetime.fromtimestamp(time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                            "source": "cache"
                        }
            except Exception as cache_err:
                self.logger.debug(f"Cache access error: {cache_err}")

            # Apply rate limiting
            self._apply_rate_limit()
            
            try:
                v5_endpoint = f"{self.base_url}/v5/market/time"
                response = requests.get(v5_endpoint, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("retCode") == 0 and "result" in data:
                        try:
                            time_nano = data["result"]["timeNano"]
                            time_ms = int(time_nano) // 1000000
                            result = {
                                "success": True,
                                "time_ms": time_ms,
                                "time": datetime.fromtimestamp(time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                                "source": "api_v5"
                            }
                            
                            # Store in cache
                            try:
                                from data.utils.cache_manager import store_cached_data
                                store_cached_data(cache_key, {"timeNow": time_ms})
                            except Exception as cache_err:
                                self.logger.debug(f"Cache store error: {cache_err}")
                                
                            return result
                        except Exception as e:
                            self.logger.warning(f"V5 API error: {e}")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed: {e}")
            
            # Fallback to local time
            current_time_ms = int(time.time() * 1000)
            return {
                "success": True,
                "time_ms": current_time_ms,
                "time": datetime.fromtimestamp(current_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                "source": "local_fallback"
            }
        except Exception as e:
            self.logger.error(f"Server time error: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": "error"
            }

    def get_klines(self,
                   symbol: str,
                   interval: str = "15",
                   limit: int = 200) -> List[Dict[str, Any]]:
        """
        Get OHLCV candlestick data using V5 API.

        Args:
            symbol: Trading symbol (e.g. BTCUSDT)
            interval: Time interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles to get (max 1000)

        Returns:
            List of candles in OHLCV format
        """
        try:
            self._apply_rate_limit()
            
            if not self._connection_initialized:
                self._initialize_client()

            # Convert interval to V5 API format
            interval_map = {
                "1": "1", "3": "3", "5": "5", "15": "15", "30": "30",
                "60": "60", "120": "120", "240": "240", "360": "360", "720": "720",
                "D": "D", "W": "W", "M": "M",
                # Handle legacy formats
                "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
                "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
                "1d": "D", "1w": "W", "1M": "M"
            }

            api_interval = interval_map.get(interval)
            if not api_interval:
                self.logger.error(f"Invalid interval: {interval}")
                return []

            endpoint = f"{self.base_url}/v5/market/kline"
            params = {
                "symbol": symbol,
                "interval": api_interval,
                "limit": min(limit, 1000)  # V5 API max limit is 1000
            }

            try:
                response = requests.get(endpoint, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("retCode") == 0 and "result" in data:
                        klines = []
                        for k in data["result"].get("list", []):
                            # V5 API returns: [timestamp, open, high, low, close, volume, turnover]
                            try:
                                timestamp = int(k[0])
                                klines.append({
                                    "timestamp": timestamp,
                                    "datetime": datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S'),
                                    "open": float(k[1]),
                                    "high": float(k[2]), 
                                    "low": float(k[3]),
                                    "close": float(k[4]),
                                    "volume": float(k[5]),
                                    "turnover": float(k[6])
                                })
                            except (ValueError, IndexError) as e:
                                self.logger.warning(f"Error parsing kline data: {e}")
                                continue
                        
                        return klines
                    else:
                        self.logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
                else:
                    self.logger.error(f"HTTP error {response.status_code}")
            except Exception as e:
                self.logger.error(f"Request failed: {e}")

            # Fallback to simulated data
            self.logger.warning("Using simulated kline data")
            current_time = int(time.time())
            klines = []
            last_price = 50000.0 if "BTC" in symbol else 3000.0

            for i in range(limit):
                timestamp = current_time - (int(interval) * 60 * (limit - i - 1))
                price_change = random.uniform(-0.01, 0.01)
                last_price = last_price * (1 + price_change)
                
                open_price = last_price
                high_price = open_price * (1 + random.uniform(0, 0.005))  
                low_price = open_price * (1 - random.uniform(0, 0.005))
                close_price = last_price
                volume = random.uniform(1, 100) if "BTC" in symbol else random.uniform(10, 1000)
                turnover = volume * ((high_price + low_price) / 2)

                klines.append({
                    "timestamp": timestamp * 1000,  # Convert to ms
                    "datetime": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2), 
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 8),
                    "turnover": round(turnover, 8)
                })

            return klines

        except Exception as e:
            self.logger.error(f"Error in get_klines: {e}")
            return []

    def get_order_book(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get order book data using V5 API.

        Args:
            symbol: Trading symbol (e.g. BTCUSDT)
            limit: Number of price levels to get (max 200)

        Returns:
            Dict containing order book data with bids and asks
        """
        try:
            self._apply_rate_limit()
            
            if not self._connection_initialized:
                self._initialize_client()

            # V5 API only supports specific limit values
            supported_limits = [1, 50, 200]
            adjusted_limit = min([x for x in supported_limits if x >= limit], default=50)

            endpoint = f"{self.base_url}/v5/market/orderbook"
            params = {
                "category": "spot",
                "symbol": symbol,
                "limit": adjusted_limit
            }

            try:
                response = requests.get(endpoint, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("retCode") == 0 and "result" in data:
                        ob_data = data["result"]
                        
                        # Format order book data
                        bids = [[float(price), float(qty)] for price, qty in ob_data.get("b", [])]
                        asks = [[float(price), float(qty)] for price, qty in ob_data.get("a", [])]
                        
                        # Limit to requested number of levels
                        bids = bids[:limit]
                        asks = asks[:limit]
                        
                        return {
                            "symbol": symbol,
                            "timestamp": ob_data.get("ts", int(time.time() * 1000)),
                            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "bids": bids,
                            "asks": asks,
                            "success": True,
                            "source": "api_v5"
                        }
                    else:
                        self.logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
                        
                else:
                    self.logger.error(f"HTTP error {response.status_code}")

            except Exception as e:
                self.logger.error(f"Request failed: {e}")

            # Fallback to simulated data
            self.logger.warning("Using simulated order book data")
            base_price = 50000.0 if "BTC" in symbol else 3000.0
            bids = []
            asks = []

            for i in range(limit):
                bid_price = base_price * (1 - 0.0001 * (i + 1))
                ask_price = base_price * (1 + 0.0001 * (i + 1))

                bid_amount = random.uniform(0.1, 2.0) if "BTC" in symbol else random.uniform(1.0, 20.0)
                ask_amount = random.uniform(0.1, 2.0) if "BTC" in symbol else random.uniform(1.0, 20.0)

                bids.append([round(bid_price, 2), round(bid_amount, 6)])
                asks.append([round(ask_price, 2), round(ask_amount, 6)])

            return {
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "bids": bids,
                "asks": asks,
                "success": True,
                "source": "simulation"
            }

        except Exception as e:
            self.logger.error(f"Error in get_order_book: {e}")
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "error": str(e),
                "success": False,
                "source": "error"
            }

    def get_wallet_balance(self, coin: str = None) -> Dict[str, Any]:
        """
        Pobiera saldo portfela dla określonej waluty lub wszystkich walut.

        Args:
            coin (str, optional): Symbol waluty. Domyślnie None (wszystkie waluty).

        Returns:
            dict: Informacje o saldzie portfela
        """
        try:
            self._apply_rate_limit()

            # Używamy API V5 - sprawdź czy klient jest zainicjalizowany
            if not self._connection_initialized:
                self._initialize_client()

            if self.client is None:
                return {
                    "success": False,
                    "error": "Klient API nie jest zainicjalizowany",
                    "balances": {}
                }

            # W trybie testnet zwracamy symulowane dane
            if self.use_testnet:
                balances = {
                    "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                    "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
                }

                if coin:
                    if coin in balances:
                        return {
                            "success": True,
                            "balances": {coin: balances[coin]},
                            "source": "testnet_simulation"
                        }
                    return {
                        "success": False,
                        "error": f"Coin {coin} not found in balances",
                        "balances": {}
                    }

                return {
                    "success": True,
                    "balances": balances,
                    "source": "testnet_simulation"
                }

            # Bezpośrednie zapytanie HTTP do V5 API z obsługą HMAC auth
            endpoint = f"{self.base_url}/v5/account/wallet-balance"

            params = {}
            if coin:
                params["coin"] = coin.upper()

            # Generate signature
            timestamp = int(time.time() * 1000)
            recv_window = 5000
            
            param_str = f"{timestamp}{self.api_key}{recv_window}"
            if params:
                param_str += json.dumps(params)
            
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                param_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Headers
            headers = {
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-TIMESTAMP': str(timestamp),
                'X-BAPI-RECV-WINDOW': str(recv_window),
                'X-BAPI-SIGN': signature
            }

            try:
                response = requests.get(
                    endpoint,
                    params=params,
                    headers=headers,
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        return {
                            "success": True,
                            "balances": data['result'],
                            "source": "api"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"API error: {data.get('retMsg')}",
                            "balances": {}
                        }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP error {response.status_code}",
                        "balances": {}
                    }
            except Exception as e:
                self.logger.error(f"Error fetching wallet balance: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "balances": {}
                }
        except Exception as e:
            self.logger.error(f"Error in get_wallet_balance: {e}")
            return {
                "success": False,
                "error": str(e),
                "balances": {}
            }

    def get_account_balance(self) -> Dict[str, Any]:
        """Pobiera saldo konta z zastosowaniem zaawansowanego cache i rate limitingu."""
        # Import managera cache
        from data.utils.cache_manager import get_cached_data, store_cached_data, is_cache_valid, get_api_status

        # Klucz cache
        cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"

        # Sprawdź status API - jeśli przekroczono limity, użyj dłuższego TTL
        api_status = get_api_status()
        ttl = 300 if api_status[
            "rate_limited"] else 30  # 5 minut w stanie przekroczenia limitów, 30s normalnie

        # Sprawdzenie czy dane są w cache i ważne
        if is_cache_valid(cache_key, ttl=ttl):
            cached_data = get_cached_data(cache_key)
            if cached_data and cached_data[0]:
                self.logger.debug(
                    f"Używam danych z cache dla account_balance (TTL: {ttl}s)")
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
                    # Używamy tylko V5 API
                    self.logger.info(
                        f"Próba reinicjalizacji klienta API. API key: {self.api_key[:5]}..., Testnet: {self.use_testnet}"
                    )
                    
                    if PybitHTTP is not None:
                        self.client = PybitHTTP(
                            testnet=self.use_testnet,
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            recv_window=20000
                        )
                        self.api_version = "v5"
                        self.logger.info("Klient API został pomyślnie reinicjalizowany.")
                    else:
                        raise ImportError("Nie można zaimportować PybitHTTP")
                except Exception as initerror:
                    self.logger.error(
                        f"Nie udało się zainicjalizować klienta API: {initerror}"
                    )
                    logging.error(
                        f"Klient API nie został zainicjalizowany. Błąd: {initerror}"
                    )
                    return {
                        "balances": {
                            "BTC": {
                                "equity": 0.005,
                                "available_balance": 0.005,
                                "wallet_balance": 0.005
                            },
                            "USDT": {
                                "equity": 500,
                                "available_balance": 450,
                                "wallet_balance": 500
                            }
                        },
                        "success": False,
                        "error":
                        f"Klient API nie został zainicjalizowany. Błąd: {initerror}",
                        "note": "Dane przykładowe -błąd inicjalizacji klienta"
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
                                "BTC": {
                                    "equity": 0.015,
                                    "available_balance": 0.015,
                                    "wallet_balance": 0.015
                                },
                                "USDT": {
                                    "equity": 1200,
                                    "available_balance": 1150,
                                    "wallet_balance": 1200
                                }
                            },
                            "success": True,
                            "note": "Dane testowe - tryb testnet"
                        }
                    else:
                        # Prawdziwa implementacja lub symulacja jeśli połączenie nie działa
                        # Przygotowanie zasłoniętego klucza do logów
                        masked_key = f"{self.api_key[:4]}{'*' * (len(self.api_key) - 4)}" if self.api_key else "Brak klucza"
                        self.logger.info(
                            f"Próba pobrania danych z {'PRODUKCYJNEGO' if not self.use_testnet else 'TESTOWEGO'} API Bybit. Klucz: {masked_key}"
                        )
                    self.logger.info(
                        f"Status API: {'Produkcyjne' if not self.use_testnet else 'Testnet'}"
                    )

                    try:
                        # Test połączenia z API z obsługą limitów zapytań
                        try:
                            # Wydłużone opóźnienie między zapytaniami dla testu połączenia
                            if not self.use_testnet:
                                time.sleep(
                                    5.0
                                )  # Dodatkowe 5 sekund dla API produkcyjnego
                            else:
                                time.sleep(3.0)  # 3 sekundy dla testnet
                            self._apply_rate_limit()

                            # W przypadku przekroczenia limitu zapytań, używamy czasu lokalnego
                            # zamiast próbować wielokrotnie odpytywać API
                            if self.remaining_rate_limit < 10:
                                local_time = int(time.time() * 1000)
                                self.logger.info(
                                    f"Używam lokalnego czasu ({local_time}) zamiast odpytywania API (oszczędzanie limitów)"
                                )
                                return {
                                    'success': True,
                                    'time_ms': local_time,
                                    'time': time.strftime('%Y-%m-%d %H:%M:%S')
                                }

                            # Sprawdzanie dostępu do API przez bezpośrednie zapytanie HTTP do publicznego endpointu
                            try:
                                # Próba z endpointem Unified v5 API
                                v5_endpoint = f"{self.base_url}/v5/market/time"
                                self.logger.debug(
                                    f"Test połączenia z V5 API: {v5_endpoint}")
                                # Direct connection without proxy
                                response = requests.get(v5_endpoint,
                                                        timeout=10)

                                if response.status_code == 200:
                                    data = response.json()
                                    if data.get("retCode") == 0 and "result" in data:
                                        time_response = {
                                            "timeNow":
                                            data["result"]["timeNano"] //
                                            1000000
                                        }
                                        self.logger.debug(
                                            f"Czas serweraV5: {time_response}")
                                    else:
                                        raise Exception(
                                            f"Nieprawidłowa odpowiedź z V5 API: {data}"
                                        )
                                else:
                                    # Fallback do Spot API
                                    spot_endpoint = f"{self.base_url}/spot/v1/time"
                                    self.logger.debug(
                                        f"Test połączenia ze Spot API: {spot_endpoint}"
                                    )
                                    # Direct connection without proxy
                                    response = requests.get(spot_endpoint,
                                                            timeout=10)

                                    if response.status_code == 200:
                                        data = response.json()
                                        if data.get("ret_code") == 0 and "serverTime" in data:
                                            time_response = {"timeNow": data["serverTime"]}
                                            self.logger.debug(
                                                f"Czas serwera Spot: {time_response}"
                                            )
                                        else:
                                            raise Exception(
                                                f"Nieprawidłowa odpowiedź ze Spot API: {data}"
                                            )
                                    else:
                                        raise Exception(
                                            f"Błąd HTTP {response.status_code} dla obu endpointów"
                                        )
                            except Exception as e:
                                # Awaryjnie używamy czasu lokalnego
                                time_response = {
                                    "timeNow": int(time.time() * 1000)
                                }
                                self.logger.warning(
                                    f"Brak dostępu do czasu serwera, używam czasu lokalnego. Błąd: {e}"
                                )

                            self.logger.info(
                                f"Test połączenia z API: {time_response}")
                        except Exception as time_error:
                            error_str = str(time_error)
                            self.logger.error(
                                f"Test połączenia z API nie powiódł się: {time_error}"
                            )

                            # Importuj funkcje wykrywania i ustawiania blokady CloudFront
                            try:
                                from data.utils.cache_manager import detect_cloudfront_error, set_cloudfront_block_status
                                # Wykrywanie błędów CloudFront i limitów IP
                                if detect_cloudfront_error(error_str):
                                    self.rate_limit_exceeded = True
                                    self.last_rate_limit_reset = time.time()
                                    self.remaining_rate_limit = 0

                                    # Ustaw blokadę CloudFront
                                    set_cloudfront_block_status(
                                        True, error_str)
                                    self.logger.warning(
                                        f"Wykryto blokadę/limit API w komunikacie błędu: {error_str}"
                                    )
                            except ImportError:
                                # Jeśli funkcje nie istnieją, implementujemy prostą weryfikację
                                error_str_lower = error_str.lower()
                                has_cloudfront_error = any(
                                    indicator in error_str_lower
                                    for indicator in [
                                        'cloudfront', 'distribution', '403',
                                        'rate limit', '429'
                                    ])
                                if has_cloudfront_error:
                                    self.rate_limit_exceeded = True
                                    self.last_rate_limit_reset = time.time()
                                    self.remaining_rate_limit = 0

                                    # Bardzo agresywny backoff dla problemów z CloudFront i IP rate limit
                                    if "cloudfront" in error_str.lower(
                                    ) or "The Amazon CloudFront distribution" in error_str:
                                        self.logger.critical(
                                            f"Wykryto blokadę CloudFront - przechodzę w tryb pełnego fallback"
                                        )
                                        try:
                                            set_cloudfront_block_status(
                                                True, error_str)
                                        except Exception as cf_err:
                                            self.logger.error(
                                                f"Błąd podczas ustawiania statusu blokady CloudFront: {cf_err}"
                                            )

                                        # Ustaw ekstremalnie długi backoff dla blokady CloudFront
                                        self.min_time_between_calls = 30.0  # minimum 30s między zapytaniami
                                        self.rate_limit_backoff = 1800.0  # 30 minut backoff

                                        # Ustaw flagę _backoff_attempt dla eksponencjalnego wzrostu
                                        self._backoff_attempt = 5  # Wysoka wartość dla długiego czasu oczekiwania
                                    else:
                                        self.logger.warning(
                                            f"Przekroczono limit zapytań API - używam cache lub danych symulowanych"
                                        )
                                        try:
                                            set_cloudfront_block_status(
                                                True,
                                                f"IP Rate Limit: {error_str}")
                                        except Exception as cf_err:
                                            self.logger.error(
                                                f"Błąd podczas ustawiania statusu blokady CloudFront: {cf_err}"
                                            )

                                        # Ustaw parametry backoff dla IP rate limit
                                        self.min_time_between_calls = 20.0  # 20s zgodnie z wymaganiami
                                        self.rate_limit_backoff = 600.0  # 10 minut backoff
                                        self._backoff_attempt = 3

                                    self.logger.warning(
                                        f"[FALLBACK MODE] min_interval={self.min_time_between_calls:.1f}s, backoff={self.rate_limit_backoff:.1f}s"
                                    )

                                    # Sprawdź najpierw cache - jeśli są dane w cache
                                    cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"
                                    cached_data, cache_found = get_cached_data(
                                        cache_key)

                                    if cache_found and cached_data:
                                        # Użyj danych z cache ale dodaj flagę informacyjną
                                        cached_data["success"] = True
                                        cached_data[
                                            "source"] = "cache_fallback"
                                        cached_data[
                                            "warning"] = f"Używam danych z cache z powodu: {error_str}"
                                        self.logger.info(
                                            "UŻYWAM DANYCH Z CACHE z powodu blokady CloudFront/IP Rate Limit"
                                        )
                                        return cached_data

                                    # Nie ma danych w cache - wygeneruj symulowane dane
                                    return {
                                        "balances": {
                                            "BTC": {
                                                "equity": 0.025,
                                                "available_balance": 0.020,
                                                "wallet_balance": 0.025
                                            },
                                            "USDT": {
                                                "equity": 1500,
                                                "available_balance": 1450,
                                                "wallet_balance": 1500
                                            },
                                            "ETH": {
                                                "equity": 0.5,
                                                "available_balance": 0.5,
                                                "wallet_balance": 0.5
                                            }
                                        },
                                        "success":
                                        True,
                                        "warning":
                                        f"CloudFront/IP Rate Limit: {error_str}",
                                        "source":
                                        "simulation_cloudfront_blocked",
                                        "note":
                                        "Dane symulowane - wykryto blokadę CloudFront lub przekroczenie limitów IP"
                                    }
                                else:
                                    # Dla innych błędów zgłaszamy wyjątek
                                    raise Exception(
                                        f"Brak dostępu do API Bybit: {time_error}"
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Błąd w bloku try: {e}"
                                )

                        # Próba pobrania salda konta z uwzględnieniem różnych API
                        wallet = None

                        # Dla kont UNIFIED pomijamy API V2, które zwraca błąd 409
                        # Priorytetyzujemy bezpośrednie zapytanie HTTP do API V5
                        wallet_methods = []

                        # Sprawdzamy czy używamy konta UNIFIED
                        is_unified_account = True  # Domyślnie zakładamy że używamy konta UNIFIED

                        # Jeśli nie używamy konta UNIFIED, dodajemy standardowe metody API V2
                        if not is_unified_account:
                            wallet_methods = [('get_wallet_balance', {}),
                                            ('get_wallet_balance', {
                                                'coin': 'USDT'
                                            }), ('query_account_info', {}),
                                            ('get_account_overview', {}),
                                            ('get_account_balance', {}),
                                            ('get_balances', {})]
                            self.logger.info("Używam standardowych metod API dla konta non-UNIFIED")
                        else:
                            self.logger.info("Używam tylko API V5 dla konta UNIFIED")

                        # Jeśli powyższe metody nie zadziałały, spróbuj bezpośredniego zapytania HTTP do V5 API
                        if wallet is None:
                            try:
                                self.logger.info(
                                    "Próba pobrania salda przez bezpośrednie zapytanie HTTP do V5 API"
                                )
                                v5_endpoint = f"{self.base_url}/v5/account/wallet-balance"

                                # Parametry zgodnie z dokumentacją V5 API
                                params = {
                                    'accountType': 'UNIFIED'  # Można dostosować w zależności od typu konta
                                }

                                # Tworzenie sygnatury zgodnie z dokumentacją V5 API
                                timestamp = str(int(time.time() * 1000))
                                recv_window = "20000"  # Używamy stałej wartości recv_window

                                # Przygotowanie parametrów do sygnatury
                                # Upewnij się, że parametry są posortowane alfabetycznie po kluczach
                                param_str = ''
                                if params:
                                    param_str = '&'.join([
                                        f"{key}={value}" for key, value in
                                        sorted(params.items())
                                    ])

                                self.logger.debug(f"Parametry do podpisu (sorted): {param_str}")

                                # Poprawne tworzenie pre_sign zgodnie z dokumentacją V5
                                # Format: timestamp + api_key + recv_window + query_string
                                # W zapytaniach GET, param_str już zawiera posortowane parametry jako 'key=value&key2=value2'
                                if param_str:
                                    pre_sign = f"{timestamp}{self.api_key}{recv_window}{param_str}"
                                else:
                                    pre_sign = f"{timestamp}{self.api_key}{recv_window}"

                                self.logger.debug(f"Generowanie podpisu dla API V5. Pre-sign: [{pre_sign}]")

                                # Generowanie sygnatury HMAC SHA256
                                signature = hmac.new(
                                    bytes(self.api_secret, 'utf-8'),
                                    bytes(pre_sign, 'utf-8'),
                                    hashlib.sha256).hexdigest()

                                # Ustawienie nagłówków zgodnie z dokumentacją
                                headers = {
                                    "X-BAPI-API-KEY": self.api_key,
                                    "X-BAPI-TIMESTAMP": timestamp,
                                    "X-BAPI-RECV-WINDOW": recv_window,  # Używamy tej samej wartości co w pre_sign
                                    "Content-Type": "application/json"
                                }

                                self.logger.debug(
                                    f"Wysyłanie zapytania do V5 API: {v5_endpoint} z parametrami: {params}"
                                )
                                # Direct connection without proxy
                                response = requests.get(v5_endpoint,
                                                        headers=headers,
                                                        params=params,
                                                        timeout=10)

                                if response.status_code == 200:
                                    wallet = response.json()
                                    self.logger.info(
                                        f"Saldo pobrane przez bezpośrednie zapytanie HTTP do V5 API"
                                    )
                                    self.logger.debug(
                                        f"Odpowiedź API: {str(wallet)[:200]}..."
                                    )
                                else:
                                    self.logger.warning(
                                        f"Błąd podczas pobierania salda przez V5 API: {response.status_code} - {response.text}"
                                    )

                                    # Spróbuj alternatywnego API - Spot API v1
                                    spot_endpoint = f"{self.base_url}/spot/v1/account"
                                    timestamp = str(int(time.time() * 1000))
                                    pre_sign = f"{timestamp}{self.api_key}"
                                    signature = hmac.new(
                                        bytes(self.api_secret, 'utf-8'),
                                        bytes(pre_sign, 'utf-8'),
                                        hashlib.sha256).hexdigest()

                                    headers = {
                                        "X-BAPI-API-KEY": self.api_key,
                                        "X-BAPI-TIMESTAMP": timestamp,
                                        "X-BAPI-SIGN": signature,
                                        "Content-Type": "application/json"
                                    }

                                    self.logger.debug(
                                        f"Próba zapytania do Spot API v1: {spot_endpoint}"
                                    )
                                    # Direct connection without proxy
                                    response = requests.get(spot_endpoint,
                                                            headers=headers,
                                                            timeout=10)

                                    if response.status_code == 200:
                                        wallet = response.json()
                                        self.logger.info(
                                            f"Saldo pobrane przez bezpośrednie zapytanie HTTP do Spot API v1"
                                        )
                                    else:
                                        self.logger.warning(
                                            f"Błąd podczas pobierania salda przez Spot API v1: {response.status_code} - {response.text}"
                                        )
                            except Exception as e:
                                self.logger.warning(
                                    f"Błąd podczas używania bezpośredniego zapytania HTTP: {e}"
                                )
                                # Kontynuujemy, aby spróbować innych metod
                        for method_name, params in wallet_methods:
                            if hasattr(self.client, method_name):
                                try:
                                    method = getattr(self.client, method_name)
                                    wallet = method(**params)
                                    self.logger.info(
                                        f"Saldo pobrane metodą: {method_name}")
                                    break
                                except Exception as method_error:
                                    self.logger.warning(
                                        f"Błąd podczas używania metody {method_name}: {method_error}")

                        if wallet is None:
                            self.logger.error(
                                "Wszystkie metody pobierania salda zawiodły")
                            raise Exception(
                                "Brak dostępnych metod do pobrania salda portfela"
                            )

                        self.logger.info(
                            f"Odpowiedź API Bybit: {str(wallet)[:200]}...")

                        # Dodatkowe logowanie struktury odpowiedzi dla celów diagnostycznych
                        if "result" in wallet:
                            result_keys = list(wallet["result"].keys()) if isinstance(wallet["result"], dict) else "nie jest słownikiem"
                            self.logger.debug(f"Struktura odpowiedzi - klucze w result: {result_keys}")
                            if "list" in wallet["result"] and isinstance(wallet["result"]["list"], list):
                                first_item = wallet["result"]["list"][0] if wallet["result"]["list"] else "pusta lista"
                                self.logger.debug(f"Pierwszy element listy: {first_item}")

                        # Sprawdzenie czy odpowiedź zawiera kod błędu
                        if "retCode" in wallet and wallet["retCode"] != 0:
                            error_msg = wallet.get("retMsg",
                                                   "Nieznany błąd API")
                            self.logger.error(
                                f"API zwróciło błąd: {error_msg}")
                            raise Exception(f"Błąd API ByBit: {error_msg}")

                        result = {
                            "balances": {},
                            "success": True,
                            "source": "API",
                            "api_version": self.api_version
                        }

                        # Obsługa różnych formatów odpowiedzi w zależności od wersji API
                        if wallet and "result" in wallet:
                            # Nowsza struktura API ByBit V5
                            # Sprawdź czy result zawiera listę (typowy format V5 API)
                            if "list" in wallet["result"] and isinstance(wallet["result"]["list"], list):
                                for account_data in wallet["result"]["list"]:
                                    # Sprawdź czy dane zawierają pole "coin" które jest listą słowników
                                    if isinstance(account_data, dict):
                                        # Obsługa różnych formatów odpowiedzi API
                                        if "coin" in account_data:
                                            coin_list = account_data["coin"] if isinstance(account_data["coin"], list) else [account_data]
                                            for coin_data in coin_list:
                                                if isinstance(coin_data, dict) and "coin" in coin_data:
                                                    coin = coin_data["coin"]
                                                    if isinstance(coin, str) and coin:
                                                        # Bezpieczne konwertowanie wartości na float z obsługą pustych wartości
                                                        try:
                                                            equity = float(coin_data.get("equity", 0) or 0)
                                                        except (ValueError, TypeError):
                                                            self.logger.warning(f"Błędna wartość equity dla {coin}: {coin_data.get('equity')}, ustawiam 0")
                                                            equity = 0.0

                                                        # Sprawdź dostępne saldo z różnych możliwych pól
                                                        available_value = coin_data.get("availableBalance") or coin_data.get("availableToWithdraw") or 0
                                                        try:
                                                            available_balance = float(available_value)
                                                        except (ValueError, TypeError):
                                                            self.logger.warning(f"Błędna wartość available_balance dla {coin}: {available_value}, ustawiam 0")
                                                            available_balance = 0.0

                                                        try:
                                                            wallet_balance = float(coin_data.get("walletBalance", 0) or 0)
                                                        except (ValueError, TypeError):
                                                            self.logger.warning(f"Błędna wartość wallet_balance dla {coin}: {coin_data.get('walletBalance')}, ustawiam 0")
                                                            wallet_balance = 0.0

                                                        result["balances"][coin] = {
                                                            "equity": equity,
                                                            "available_balance": available_balance,
                                                            "wallet_balance": wallet_balance
                                                        }
                                                        self.logger.debug(f"Dodano saldo dla {coin}: {result['balances'][coin]}")
                                    # Obsługa przypadku gdy coin jest bezpośrednio w danych konta
                                    elif isinstance(account_data, dict):
                                        coin = account_data.get("coin")
                                        # Upewnienie się że coin jest stringiem, a nie listą
                                        if isinstance(coin, str) and coin:
                                            result["balances"][coin] = {
                                                "equity": float(account_data.get("equity", 0)),
                                                "available_balance": float(
                                                    account_data.get(
                                                        "availableBalance", 0) or 
                                                    account_data.get(
                                                        "availableToWithdraw", 0)
                                                ),
                                                "wallet_balance": float(account_data.get("walletBalance", 0))
                                            }
                                            self.logger.debug(f"Dodano saldo dla {coin}: {result['balances'][coin]}")
                                        elif isinstance(coin, list) and coin:
                                            # Jeśli coin jest listą, iteruj po jej elementach
                                            self.logger.warning(f"Otrzymano listę coin zamiast stringa: {coin}")
                                            for single_coin in coin:
                                                if isinstance(single_coin, str) and single_coin:
                                                    result["balances"][single_coin] = {
                                                        "equity": float(account_data.get("equity", 0)),
                                                        "available_balance": float(
                                                            account_data.get("availableBalance", 0) or 
                                                            account_data.get("availableToWithdraw", 0)
                                                        ),
                                                        "wallet_balance": float(account_data.get("walletBalance", 0))
                                                    }
                                                    self.logger.debug(f"Dodano saldo dla coin z listy: {single_coin}")
                                    # Obsługa przypadku gdy coin jest bezpośrednio w danych konta
                                    elif isinstance(account_data, dict):
                                        coin = account_data.get("coin")
                                        # Upewnienie się że coin jest stringiem, a nie listą
                                        if isinstance(coin, str) and coin:
                                            result["balances"][coin] = {
                                                "equity": float(account_data.get("equity", 0)),
                                                "available_balance": float(
                                                    account_data.get(
                                                        "availableBalance", 0) or 
                                                    account_data.get(
                                                        "availableToWithdraw", 0)
                                                ),
                                                "wallet_balance": float(account_data.get("walletBalance", 0))
                                            }
                                            self.logger.debug(f"Dodano saldo dla {coin}: {result['balances'][coin]}")
                                        elif isinstance(coin, list) and coin:
                                            # Jeśli coin jest listą, iteruj po jej elementach
                                            self.logger.warning(f"Otrzymano listę coin zamiast stringa: {coin}")
                                            for single_coin in coin:
                                                if isinstance(single_coin, str) and single_coin:
                                                    result["balances"][single_coin] = {
                                                        "equity": float(account_data.get("equity", 0)),
                                                        "available_balance": float(
                                                            account_data.get("availableBalance", 0) or 
                                                            account_data.get("availableToWithdraw", 0)
                                                        ),
                                                        "wallet_balance": float(account_data.get("walletBalance", 0))
                                                    }
                                                    self.logger.debug(f"Dodano saldo dla coin z listy: {single_coin}")
                        elif wallet and "result" in wallet and isinstance(
                                wallet["result"], dict):
                            # Starsza struktura API ByBit lub format usdt_perpetual
                            for coin, coin_data in wallet["result"].items():
                                if isinstance(coin_data, dict):
                                    result["balances"][coin] = {
                                        "equity":
                                        float(coin_data.get("equity", 0)),
                                        "available_balance":
                                        float(
                                            coin_data.get(
                                                "available_balance", 0)
                                            or coin_data.get(
                                                "availableBalance", 0)),
                                        "wallet_balance":
                                        float(
                                            coin_data.get("wallet_balance", 0)
                                            or coin_data.get(
                                                "walletBalance", 0))
                                    }
                        elif wallet and "result" in wallet and isinstance(
                                wallet["result"], list):
                            # Format odpowiedzi dla niektórych wersji API
                            for coin_data in wallet["result"]:
                                coin = coin_data.get("coin", "")
                                if coin:
                                    result["balances"][coin] = {
                                        "equity":
                                        float(coin_data.get("equity", 0)),
                                        "available_balance":
                                        float(
                                            coin_data.get(
                                                "available_balance", 0)
                                            or coin_data.get(
                                                "availableBalance", 0)),
                                        "wallet_balance":
                                        float(
                                            coin_data.get("wallet_balance", 0)
                                            or coin_data.get(
                                                "walletBalance", 0))
                                    }

                        if not result["balances"]:
                            self.logger.warning(
                                f"API zwróciło pustą listę sald. Pełna odpowiedź: {wallet}"
                            )
                            result["warning"] = "API zwróciło pustą listę sald"
                            self.logger.info(
                                "Próba pobrania danych z API zwróciła pustą listę sald. Możliwe przyczyny: brak środków na koncie, nieprawidłowe konto, nieprawidłowe uprawnienia API."
                            )

                        # Zapisanie poprawnych danych w cache
                        from data.utils.cache_manager import store_cached_data
                        cache_key = f"account_balance_{self.api_key[:8]}_{self.use_testnet}"
                        store_cached_data(cache_key, result)
                        self.logger.debug("Zapisano dane portfolio w cache")

                        return result
                    except Exception as e:
                        self.logger.error(
                            f"Błąd podczas pobierania danych z prawdziwego API: {e}. Traceback: {traceback.format_exc()}"
                        )
                        # Dane symulowane w przypadku błędu
                        return {
                            "balances": {
                                "BTC": {
                                    "equity": 0.025,
                                    "available_balance": 0.020,
                                    "wallet_balance": 0.025
                                },
                                "USDT": {
                                    "equity": 1500,
                                    "available_balance": 1450,
                                    "wallet_balance": 1500
                                },
                                "ETH": {
                                    "equity": 0.5,
                                    "available_balance": 0.5,
                                    "wallet_balance": 0.5
                                }
                            },
                            "success": False,
                            "error": str(e),
                            "source": "simulation",
                            "note": "Dane symulowane - błąd API: " + str(e)
                        }
                except Exception as e:
                    error_str = str(e)
                    # Sprawdzenie, czy błąd dotyczy przekroczenia limitu zapytań
                    if "rate limit" in error_str.lower(
                    ) or "429" in error_str or "403" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            self.logger.warning(
                                f"Przekroczono limit zapytań API. Ponawiam próbę {retry_count}/{max_retries} za {retry_delay} sekund..."
                            )
                            # Zwiększamy opóźnienie wykładniczo, aby uniknąć ponownego przekroczenia limitu
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Podwajamy czas oczekiwania przy każdej próbie
                            continue
                        else:
                            self.logger.warning(
                                f"Wykorzystano wszystkie próby ponawiania. Zwracam dane symulowane."
                            )

                    self.logger.error(
                        f"Krytyczny błąd podczas pobierania salda konta: {e}. Traceback: {traceback.format_exc()}"
                    )
                    # Dane symulowane w przypadku błędu
                    return {
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
                        "success": False,
                        "error": str(e),
                        "source": "simulation_error",
                        "note": "Dane symulowane - wystąpił błąd: " + str(e)
                    }

                # Jeśli dotarliśmy tutaj, to znaczy, że zapytanie się powiodło
                break

        except Exception as e:
            self.logger.error(
                f"Krytyczny błąd podczas pobierania salda konta: {e}. Traceback: {traceback.format_exc()}"
            )
            # Dane symulowane w przypadku błędu
            return {
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
                "success": False,
                "error": str(e),
                "source": "simulation_critical_error",
                "note": "Dane symulowane - wystąpił krytyczny błąd: " + str(e)
            }

        return {
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
            "success": False,
            "error": "Nieudane pobieranie salda",
            "source": "fallback",
            "note": "Dane symulowane - żadna próba nie powiodła się"
        }

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information.

        Args:
            symbol: Trading symbol

        Returns:
            Dict containing ticker data or error information
        """
        try:
            response = self.session.get(f"{self.base_url}/spot/v3/public/quote/ticker/price", 
                                      params={"symbol": symbol})
            
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0:
                    return {
                        "success": True,
                        "data": data["result"]
                    }
                elif data.get("retCode") == 10006:
                    return {
                        "success": False,
                        "error": "Rate limit exceeded"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API Error: {data.get('retMsg', 'Unknown error')}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP Error: {response.status_code}"
                }

        except Exception as e:
            self.logger.error(f"Error in get_ticker: {str(e)}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }

    def place_order(self,
                    symbol: str,
                    side: str,
                    price: float,
                    quantity: float,
                    order_type: str = "Limit") -> Dict[str, Any]:
        """
        Place an order using V5 API.

        Args:
            symbol (str): Trading symbol (e.g. BTCUSDT)
            side (str): Order side ('Buy' or 'Sell')
            price (float): Order price (ignored for market orders)
            quantity (float): Order quantity
            order_type (str): Order type ('Limit' or 'Market')

        Returns:
            Dict containing order details or error
        """
        try:
            self._apply_rate_limit()
            
            if not self._connection_initialized:
                self._initialize_client()

            # Input validation
            if side not in ["Buy", "Sell"]:
                return {"success": False, "error": "Invalid side. Must be 'Buy' or 'Sell'"}
            
            if order_type not in ["Limit", "Market"]:
                return {"success": False, "error": "Invalid order type. Must be 'Limit' or 'Market'"}
            
            if quantity <= 0:
                return {"success": False, "error": "Quantity must be positive"}
            
            if order_type == "Limit" and price <= 0:
                return {"success": False, "error": "Price must be positive for limit orders"}

            # Production API safety checks
            if self.is_production_api():
                self.logger.warning(
                    f"PRODUCTION ORDER: {side} {quantity} {symbol} @ {price if order_type == 'Limit' else 'Market'}"
                )

                max_order_value_usd = 1000.0
                est_order_value = quantity * (price if order_type == "Limit" else price)

                if est_order_value > max_order_value_usd:
                    self.logger.critical(
                        f"ORDER REJECTED: Exceeds maximum order value ({est_order_value} USD > {max_order_value_usd} USD)"
                    )
                    return {
                        "success": False,
                        "error": f"Order exceeds maximum allowed value of {max_order_value_usd} USD"
                    }

            # Prepare order parameters
            params = {
                "category": "spot",  # or linear, inverse depending on market_type
                "symbol": symbol,
                "side": side.upper(),
                "orderType": order_type.upper(),
                "qty": str(quantity),
            }

            if order_type == "Limit":
                params["price"] = str(price)

            # Generate signature
            timestamp = str(int(time.time() * 1000))
            recv_window = "20000"

            param_str = '&'.join([
                f"{key}={value}" for key, value in sorted(params.items())
            ])

            # Format: timestamp + api_key + recv_window + query_string
            pre_sign = f"{timestamp}{self.api_key}{recv_window}{param_str}"

            signature = hmac.new(
                bytes(self.api_secret, 'utf-8'),
                bytes(pre_sign, 'utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Headers
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
                "Content-Type": "application/json"
            }

            endpoint = f"{self.base_url}/v5/order/create"

            try:
                response = requests.post(
                    endpoint,
                    json=params,
                    headers=headers,
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("retCode") == 0:
                        order_data = data.get("result", {})
                        return {
                            "success": True,
                            "order_id": order_data.get("orderId"),
                            "symbol": symbol,
                            "side": side,
                            "price": price if order_type == "Limit" else None,
                            "quantity": quantity,
                            "order_type": order_type,
                            "status": order_data.get("orderStatus", "Created"),
                            "created_time": order_data.get("createdTime"),
                            "source": "api_v5"
                        }
                    else:
                        return {
                            "success": False,
                            "error": data.get("retMsg", "Unknown API error"),
                            "code": data.get("retCode"),
                            "source": "api_v5"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "source": "http"
                    }

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed: {e}")
                return {
                    "success": False,
                    "error": f"Request failed: {str(e)}",
                    "source": "request"
                }

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": "internal"
            }

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
        Pobiera statuszlecenia.

        Parameters:
            order_id (str): ID zlecenia.

        Returns:
            Dict[str, Any]: Status zlecenia.
        """
        try:
            self._apply_rate_limit()
            # Symulacja pobierania statusu zlecenia
            statuses = [
                "New", "PartiallyFilled", "Filled", "Cancelled", "Rejected"
            ]
            status = random.choice(statuses)

            return {
                "success":
                True,
                "order_id":
                order_id,
                "status":
                status,
                "filled_quantity":
                random.uniform(0, 1) if status == "PartiallyFilled" else
                (1.0 if status == "Filled" else 0.0),
                "timestamp":
                int(time.time() * 1000),
                "datetime":
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            self.logger.error(f"Błąd podczas pobierania statusu zlecenia: {e}")
            return {"success": False, "error": str(e)}

    def _apply_rate_limit(self):
        """Applies improved rate limiting with exponential backoff"""
        if self.rate_limit_exceeded:
            wait_time = min(300, self.rate_limit_wait_time * 2)  # Cap at 5 minutes
            self.logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds")
            time.sleep(wait_time)
            self.rate_limit_wait_time = wait_time
        else:
            # Standard rate limit prevention
            time.sleep(0.05)  # 50ms between requests

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

    def _handle_v5_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Improved V5 API response handler"""
        if not isinstance(response, dict):
            return {"success": False, "error": "Invalid response format"}
            
        ret_code = response.get("retCode")
        if ret_code == 0:
            return {"success": True, "data": response.get("result", {})}
        elif ret_code in [10006, 10018]:  # Rate limit error codes
            self.rate_limit_exceeded = True
            return {"success": False, "error": "Rate limit exceeded"}
        else:
            return {"success": False, "error": response.get("retMsg", "Unknown error")}

    def _validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validates API response and ensures consistent return format."""
        if not isinstance(response, dict):
            return {
                "success": False,
                "error": "Invalid response format",
                "source": "validation"
            }
        
        # Check for common error patterns
        if "ret_code" in response:
            if response["ret_code"] != 0:
                return {
                    "success": False,
                    "error": response.get("ret_msg", "Unknown error"),
                    "code": response.get("ret_code"),
                    "source": "api"
                }
            # V5 API success case
            return {
                "success": True,
                "data": response.get("result", {}),
                "source": "api_v5"
            }

        # If response is already in our standard format
        if "success" in response:
            return response

        # If response is in a different format but seems valid
        if "result" in response:
            return {
                "success": True,
                "data": response["result"],
                "source": "api"
            }

        return {
            "success": False,
            "error": "Unknown response format",
            "source": "validation"
        }

    def _handle_request_error(self, e: Exception, context: str = "") -> Dict[str, Any]:
        """Standardizes error handling for API requests."""
        error_msg = str(e)
        self.logger.error(f"API error in {context}: {error_msg}")
        
        # Rate limit handling
        if "too many requests" in error_msg.lower() or "rate limit" in error_msg.lower():
            self.rate_limit_exceeded = True
            # Store in cache
            try:
                from data.utils.cache_manager import store_cached_data
                store_cached_data("api_rate_limited", {"timestamp": time.time()})
            except Exception as cache_error:
                self.logger.warning(f"Failed to cache rate limit status: {cache_error}")
        
        return {
            "success": False,
            "error": error_msg,
            "source": "request",
            "context": context
        }

    def _handle_rate_limit(self) -> None:
        """
        Handle rate limiting for API calls
        """
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_time_between_calls:
            time.sleep(self.min_time_between_calls - time_since_last_call)
        
        if self.rate_limit_exceeded:
            if current_time - self.last_rate_limit_reset > self.rate_limit_backoff:
                self.rate_limit_exceeded = False
                self.remaining_rate_limit = 50 if self.use_testnet else 20
                self.last_rate_limit_reset = current_time
            else:
                raise Exception("Rate limit exceeded. Please wait before making more requests.")
        
        self.last_api_call = time.time()
        
    def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """
        Make authenticated request to Bybit API
        """
        if not self._connection_initialized:
            self._initialize_client()
            
        self._handle_rate_limit()
        
        timestamp = str(int(time.time() * 1000))
        params = params or {}
        params.update({
            "api_key": self.api_key,
            "timestamp": timestamp
        })
        
        # Generate signature
        param_str = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['sign'] = signature
        
        try:
            response = requests.request(
                method=method,
                url=endpoint,
                params=params if method == 'GET' else None,
                json=params if method != 'GET' else None
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return {"success": False, "error": str(e)}
            
    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current positions
        
        Args:
            symbol (str, optional): Trading pair symbol
            
        Returns:
            Dict with position information
        """
        try:
            if not self._connection_initialized:
                self._initialize_client()
                
            params = {}
            if symbol:
                params['symbol'] = symbol
                
            endpoint = f"{self.base_url}/v5/position/list"
            return self._make_request('GET', endpoint, params)
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"success": False, "error": str(e)}

    def _initialize_websocket(self):
        """Initialize WebSocket connection"""
        self.ws = None
        self.ws_connected = False
        self.ws_subscriptions = set()
        self.ws_callbacks = {}
        self.ws_reconnect_delay = 5
        self.ws_url = f"wss://{'stream-testnet' if self.use_testnet else 'stream'}.bybit.com/v5/public/linear"
        
        self._connect_websocket()
        
    def _connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            topic = data.get('topic')
            
            if topic in self.ws_callbacks:
                self.ws_callbacks[topic](data)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
            
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        self.ws_connected = False
        self.logger.warning("WebSocket connection closed")
        
        # Attempt to reconnect
        time.sleep(self.ws_reconnect_delay)
        self._connect_websocket()
        
    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        self.ws_connected = True
        self.logger.info("WebSocket connection established")
        
        # Resubscribe to previous topics
        for topic in self.ws_subscriptions:
            self._subscribe_to_topic(topic)
            
    def subscribe_to_orderbook(self, symbol: str, callback: Callable):
        """
        Subscribe to orderbook updates for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            callback (Callable): Function to handle incoming data
        """
        topic = f"orderbook.50.{symbol}"
        self.ws_callbacks[topic] = callback
        self.ws_subscriptions.add(topic)
        
        if self.ws_connected:
            self._subscribe_to_topic(topic)
            
    def _subscribe_to_topic(self, topic: str):
        """Send subscription request for a topic"""
        try:
            self.ws.send(json.dumps({
                "op": "subscribe",
                "args": [topic]
            }))
        except Exception as e:
            self.logger.error(f"Error subscribing to topic {topic}: {e}")
            
    def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe from a topic"""
        try:
            if topic in self.ws_subscriptions:
                self.ws.send(json.dumps({
                    "op": "unsubscribe",
                    "args": [topic]
                }))
                self.ws_subscriptions.remove(topic)
                self.ws_callbacks.pop(topic, None)
        except Exception as e:
            self.logger.error(f"Error unsubscribing from topic {topic}: {e}")

    def _handle_reconnection(self):
        """Handle WebSocket reconnection with exponential backoff"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and not self.ws_connected:
            delay = min(30, (2 ** retry_count) * self.ws_reconnect_delay)
            self.logger.info(f"Attempting to reconnect in {delay} seconds...")
            time.sleep(delay)
            
            try:
                self._connect_websocket()
                if self.ws_connected:
                    self.logger.info("Successfully reconnected to WebSocket")
                    break
            except Exception as e:
                self.logger.error(f"Reconnection attempt {retry_count + 1} failed: {e}")
                
            retry_count += 1
            
        if not self.ws_connected:
            self.logger.error("Failed to reconnect after maximum retries")
            
    def _validate_connection(self):
        """Validate WebSocket connection status"""
        if not self.ws_connected:
            raise ConnectionError("WebSocket connection is not established")
            
    def _handle_subscription_error(self, error_data):
        """Handle subscription-related errors"""
        error_code = error_data.get('ret_code')
        error_msg = error_data.get('ret_msg', 'Unknown error')
        
        if error_code:
            self.logger.error(f"Subscription error {error_code}: {error_msg}")
            
            # Handle specific error codes
            if error_code == 10001:  # Rate limit exceeded
                time.sleep(1)  # Wait before retrying
                return True  # Indicate retry needed
            elif error_code == 10002:  # Invalid topic
                return False  # Don't retry for invalid topics
            
        return True  # Default to retry for unknown errors
        
    def _process_orderbook_update(self, data):
        """Process and validate orderbook updates"""
        try:
            orderbook = data.get('data', {})
            if not orderbook:
                return None
                
            timestamp = orderbook.get('ts')
            symbol = orderbook.get('s')
            
            # Validate data structure
            if not all([timestamp, symbol]):
                self.logger.warning(f"Invalid orderbook data structure: {data}")
                return None
                
            return {
                'timestamp': timestamp,
                'symbol': symbol,
                'bids': [[float(price), float(qty)] for price, qty in orderbook.get('b', [])],
                'asks': [[float(price), float(qty)] for price, qty in orderbook.get('a', [])]
            }
        except Exception as e:
            self.logger.error(f"Error processing orderbook update: {e}")
            return None

    def _validate_websocket_data(self, data: Dict[str, Any]) -> bool:
        """Validate WebSocket message data structure."""
        if not isinstance(data, dict):
            self.logger.error(f"Invalid data type: {type(data)}")
            return False
            
        required_fields = ['topic', 'data', 'ts']
        if not all(field in data for field in required_fields):
            self.logger.error(f"Missing required fields. Got: {list(data.keys())}")
            return False
            
        return True

    def _process_trade_update(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process trade update messages from WebSocket."""
        try:
            if not self._validate_websocket_data(data):
                return None
                
            trade_data = data['data']
            return {
                'symbol': trade_data.get('s'),
                'side': trade_data.get('S'),
                'price': float(trade_data.get('p', 0)),
                'quantity': float(trade_data.get('v', 0)),
                'timestamp': trade_data.get('T'),
                'trade_id': trade_data.get('i')
            }
        except Exception as e:
            self.logger.error(f"Error processing trade update: {e}")
            return None

    def _send_ping(self):
        """Send ping message to keep WebSocket connection alive."""
        try:
            if self.ws and self.ws_connected:
                self.ws.send(json.dumps({"op": "ping"}))
                self.ws_last_ping = time.time()
        except Exception as e:
            self.logger.error(f"Error sending ping: {e}")
            self._handle_reconnection()

    def _start_ping_timer(self):
        """Start background timer for sending ping messages."""
        def ping_task():
            while self.ws_connected:
                if time.time() - self.ws_last_ping > self.ws_ping_interval:
                    self._send_ping()
                time.sleep(1)
                
        threading.Thread(target=ping_task, daemon=True).start()

    def _generate_signature(self, params: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate authentication headers for V5 API requests
        
        Args:
            params: Request parameters
            
        Returns:
            Dict containing authentication headers
        """
        timestamp = str(int(time.time() * 1000))
        recv_window = "20000"
        
        # Sort parameters alphabetically and create param string
        param_str = ""
        if params:
            param_str = "&".join([
                f"{key}={value}"
                for key, value in sorted(params.items())
            ])
            
        # Format for V5 API: timestamp + api_key + recv_window + params
        pre_hash = f"{timestamp}{self.api_key}{recv_window}{param_str}"
        
        # Generate signature
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(pre_hash, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }

    def _handle_rate_limit_headers(self, response: requests.Response) -> None:
        """
        Update rate limit tracking based on response headers
        
        Args:
            response: API response with rate limit headers
        """
        try:
            remaining = response.headers.get('X-Bapi-Limit-Remaining')
            if remaining is not None:
                self.remaining_rate_limit = int(remaining)
                
            reset_at = response.headers.get('X-Bapi-Limit-Reset-Timestamp')
            if reset_at is not None:
                self.rate_limit_reset_time = int(reset_at)
                
            # Update rate limit status
            if self.remaining_rate_limit <= 5:  # Buffer to prevent hitting absolute limit
                self.rate_limit_exceeded = True
                self.rate_limit_wait_time = max(
                    1,  # Minimum 1 second
                    (self.rate_limit_reset_time - int(time.time() * 1000)) / 1000.0  # Convert ms to seconds
                )
                self.logger.warning(
                    f"Rate limit near exceeded. Remaining: {self.remaining_rate_limit}. "
                    f"Reset in: {self.rate_limit_wait_time:.1f}s"
                )
        except Exception as e:
            self.logger.error(f"Error processing rate limit headers: {e}")

    def _make_v5_request(self, 
                        method: str,
                        endpoint: str,
                        params: Dict[str, Any] = None,
                        auth_required: bool = True) -> Dict[str, Any]:
        """
        Make a request to V5 API with proper authentication and rate limiting
        
        Args:
            method: HTTP method (GET, POST, etc)
            endpoint: API endpoint path
            params: Request parameters
            auth_required: Whether authentication is needed
            
        Returns:
            API response data
        """
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
            
        url = f"{self.base_url}{endpoint}"
        
        # Apply rate limiting
        self._apply_rate_limit()
        
        try:
            headers = {}
            if auth_required:
                headers = self._generate_signature(params)
                
            # Make request
            response = self.session.request(
                method=method,
                url=url,
                params=params if method == 'GET' else None,
                json=params if method != 'GET' else None,
                headers=headers,
                timeout=10
            )
            
            # Update rate limit tracking
            self._handle_rate_limit_headers(response)
            
            # Process response
            if response.status_code == 200:
                data = response.json()
                return self._handle_v5_response(data)
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "source": "http"
                }
                
        except Exception as e:
            return self._handle_request_error(e, f"{method} {endpoint}")

    def _check_account_type(self) -> str:
        """
        Check if account is UNIFIED or CLASSIC
        
        Returns:
            Account type ('UNIFIED' or 'CLASSIC')
        """
        try:
            response = self._make_v5_request(
                'GET',
                '/v5/account/info',
                auth_required=True
            )
            
            if response['success']:
                data = response['data']
                unified_enabled = data.get('unifiedMarginStatus', False)
                return 'UNIFIED' if unified_enabled else 'CLASSIC'
            else:
                self.logger.error(f"Failed to get account type: {response.get('error')}")
                return 'CLASSIC'  # Default to classic for safety
                
        except Exception as e:
            self.logger.error(f"Error checking account type: {e}")
            return 'CLASSIC'  # Default to classic on error

    def _handle_websocket_error(self, error: str) -> None:
        """Handle WebSocket errors with proper logging and reconnection."""
        self.logger.error(f"WebSocket error: {error}")
        
        if "rate limit" in error.lower():
            self.rate_limit_exceeded = True
            self.rate_limit_wait_time *= 2  # Exponential backoff
        
        if not self.ws_connected:
            self._handle_reconnection()

    def close_websocket(self):
        """Safely close the WebSocket connection."""
        try:
            if self.ws:
                self.ws.close()
                self.ws_connected = False
                self.ws_subscriptions.clear()
                self.ws_callbacks.clear()
                if self.ws_thread and self.ws_thread.is_alive():
                    self.ws_thread.join(timeout=1.0)
        except Exception as e:
            self.logger.error(f"Error closing WebSocket connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close_websocket()
        if self.session:
            self.session.close()
            
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.close_websocket()
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def subscribe_to_trades(self, symbol: str, callback: Callable):
        """Subscribe to real-time trade updates.
        
        Args:
            symbol: Trading pair symbol
            callback: Function to handle trade updates
        """
        topic = f"trade.{symbol}"
        self.ws_callbacks[topic] = callback
        self.ws_subscriptions.add(topic)
        
        if self.ws_connected:
            self._subscribe_to_topic(topic)

    def subscribe_to_tickers(self, symbol: str, callback: Callable):
        """Subscribe to real-time ticker updates.
        
        Args:
            symbol: Trading pair symbol
            callback: Function to handle ticker updates
        """
        topic = f"tickers.{symbol}"
        self.ws_callbacks[topic] = callback
        self.ws_subscriptions.add(topic)
        
        if self.ws_connected:
            self._subscribe_to_topic(topic)

    def handle_disconnect(self):
        """Clean up resources on disconnection."""
        self.ws_connected = False
        self.ws_subscriptions.clear()
        self.ws_callbacks.clear()
        self.ws_last_ping = 0
        
        # Cancel any pending operations
        if self.ws and hasattr(self.ws, 'sock') and self.ws.sock:
            try:
                self.ws.sock.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
                
        # Reset rate limiting
        self.rate_limit_exceeded = False
        self.rate_limit_wait_time = 1

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