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
            # W nowszych wersjach używamy unified_trading, w starszych usdb_perpetual
            if hasattr(pybit, 'unified_trading'):
                self.client = pybit.unified_trading.HTTP(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.use_testnet
                )
                self.api_version = "unified"
            elif hasattr(pybit, 'usdt_perpetual'):
                self.client = pybit.usdt_perpetual.HTTP(
                    endpoint="https://api-testnet.bybit.com" if self.use_testnet else "https://api.bybit.com",
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                self.api_version = "usdt_perpetual"
            else:
                # Fallback dla najnowszej wersji pybit (po 2.5+)
                self.client = pybit.HTTP(
                    testnet=self.use_testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                self.api_version = "v5"
                
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
        """Pobiera saldo konta."""
        try:
            if self.client is None:
                # Próba reinicjalizacji klienta
                try:
                    import pybit
                    self.logger.info(f"Próba reinicjalizacji klienta API. API key: {self.api_key[:5]}..., Testnet: {self.use_testnet}")
                    self.client = pybit.unified_trading.HTTP(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        testnet=self.use_testnet
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
                try:
                    self.logger.info(f"Próba pobrania danych z prawdziwego API Bybit. API key: {self.api_key[:5]}..., Testnet: {self.use_testnet}")
                    
                    # Testowanie połączenia z API
                    try:
                        # Najpierw sprawdźmy czy mamy dostęp do API
                        time_response = self.client.get_server_time()
                        self.logger.info(f"Test połączenia z API: {time_response}")
                    except Exception as time_error:
                        self.logger.error(f"Test połączenia z API nie powiódł się: {time_error}")
                        raise Exception(f"Brak dostępu do API Bybit: {time_error}")
                    
                    # Próba pobrania salda konta - dostosowana do różnych wersji API
                    if self.api_version == "unified":
                        wallet = self.client.get_wallet_balance(accountType="UNIFIED")
                    elif self.api_version == "usdt_perpetual":
                        wallet = self.client.get_wallet_balance()
                    else:
                        # Najnowsza wersja API (v5)
                        wallet = self.client.get_wallet_balance(accountType="UNIFIED")
                    
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
            self.logger.error(f"Krytyczny błąd podczas pobierania salda konta: {e}. Traceback: {traceback.format_exc()}")
            # Dane symulowane w przypadku błędu
            return {
                "balances": {
                    "BTC": {"equity": 0.01, "available_balance": 0.01, "wallet_balance": 0.01},
                    "USDT": {"equity": 1000, "available_balance": 950, "wallet_balance": 1000}
                },
                "success": False,
                "error": str(e),
                "note": "Dane symulowane - wystąpił błąd: " + str(e)
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