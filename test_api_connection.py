
#!/usr/bin/env python3
"""
test_api_connection.py - Skrypt do testowania połączenia z API Bybit
"""

import os
import sys
import time
import logging
from pprint import pprint
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def test_connection():
    """Testuje połączenie z API Bybit."""
    try:
        # Ładowanie zmiennych środowiskowych
        load_dotenv()
        
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
        
        if not api_key or not api_secret:
            logger.error("Brak kluczy API w zmiennych środowiskowych.")
            return False
        
        # Import modułu BybitConnector
        try:
            from data.execution.bybit_connector import BybitConnector
        except ImportError:
            logger.error("Nie można zaimportować modułu BybitConnector. Sprawdź ścieżki importu.")
            return False
        
        # Inicjalizacja klienta Bybit
        client = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=testnet,
            lazy_connect=False,
            market_type=os.getenv("MARKET_TYPE", "spot")
        )
        
        # Wyświetlenie informacji o połączeniu
        masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 4)}"
        logger.info(f"Inicjalizacja klienta Bybit - Klucz: {masked_key}, Testnet: {testnet}")
        
        # Test 1: Pobieranie czasu serwera
        logger.info("Test 1: Pobieranie czasu serwera...")
        server_time = client.get_server_time()
        logger.info(f"Czas serwera: {server_time}")
        
        # Test 2: Pobieranie danych świecowych
        logger.info("Test 2: Pobieranie danych świecowych...")
        klines = client.get_klines(symbol="BTCUSDT", interval="15", limit=5)
        logger.info(f"Pobrano {len(klines)} świec dla BTCUSDT (15m)")
        
        # Test 3: Pobieranie księgi zleceń
        logger.info("Test 3: Pobieranie księgi zleceń...")
        order_book = client.get_order_book(symbol="BTCUSDT", limit=5)
        logger.info(f"Pobrano księgę zleceń dla BTCUSDT - bids: {len(order_book.get('bids', []))}, asks: {len(order_book.get('asks', []))}")
        
        # Test 4: Pobieranie stanu portfela
        logger.info("Test 4: Pobieranie stanu portfela...")
        account_balance = client.get_account_balance()
        if account_balance.get("success", False):
            balances = account_balance.get("balances", {})
            currencies = list(balances.keys())
            logger.info(f"Pobrano stan portfela - waluty: {currencies}")
        else:
            logger.error(f"Błąd podczas pobierania stanu portfela: {account_balance.get('error', 'Nieznany błąd')}")
        
        # Test 5: Pobieranie stanu portfela z metodą get_wallet_balance
        logger.info("Test 5: Pobieranie stanu portfela z metodą get_wallet_balance...")
        wallet_balance = client.get_wallet_balance()
        if wallet_balance.get("success", False):
            balances = wallet_balance.get("balances", {})
            currencies = list(balances.keys())
            logger.info(f"Pobrano stan portfela (get_wallet_balance) - waluty: {currencies}")
        else:
            logger.error(f"Błąd podczas pobierania stanu portfela (get_wallet_balance): {wallet_balance.get('error', 'Nieznany błąd')}")
        
        logger.info("Testy zakończone pomyślnie.")
        return True
    
    except Exception as e:
        logger.error(f"Błąd podczas testowania połączenia z API: {e}")
        return False

if __name__ == "__main__":
    logger.info("Test połączenia z API Bybit")
    
    result = test_connection()
    
    if result:
        logger.info("✅ Połączenie z API działa poprawnie.")
        sys.exit(0)
    else:
        logger.error("❌ Test połączenia z API nie powiódł się.")
        sys.exit(1)
