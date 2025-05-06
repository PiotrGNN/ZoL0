"""
test_bybit_connection.py
-----------------------
Skrypt do testowania połączenia z API Bybit.
"""

import os
import logging
import pytest
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List
import backoff  # for exponential backoff

# Konfiguracja podstawowego loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    """Create a test Bybit client"""
    load_dotenv()
    
    from data.execution.bybit_connector import BybitConnector
    
    # Use test credentials
    client = BybitConnector(
        api_key="test_key",
        api_secret="test_secret",
        use_testnet=True
    )
    
    return client

@pytest.fixture
def symbol():
    """Test trading symbol"""
    return "BTCUSDT"

def validate_response(response: Dict[str, Any]) -> bool:
    """Validates API response"""
    if not isinstance(response, dict):
        return False
    return response.get("success", False) or bool(response.get("balances"))

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def test_initialization(client):
    """Test client initialization"""
    assert client is not None, "Client should not be None"
    assert client.use_testnet is True, "Test client should use testnet"
    assert client.api_version == "v5", "Client should use API v5"
    assert "testnet" in client.base_url, "Base URL should point to testnet"

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def test_server_time(client) -> Dict[str, Any]:
    """Test server time with retries"""
    response = client.get_server_time()
    assert validate_response(response), f"Invalid server time response: {response}"
    assert "time_ms" in response, "Response should contain time_ms"
    return response

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def test_account_balance(client) -> Dict[str, Any]:
    """Test account balance with retries"""
    balance = client.get_wallet_balance()
    assert validate_response(balance), f"Invalid balance response: {balance}"
    assert "balances" in balance, "Missing balances in response"
    assert isinstance(balance["balances"], dict), "balances should be a dictionary"
    
    # Test that each balance has the required fields
    for coin, coin_balance in balance["balances"].items():
        assert "equity" in coin_balance, f"Missing equity for {coin}"
        assert "available_balance" in coin_balance, f"Missing available_balance for {coin}"
        assert "wallet_balance" in coin_balance, f"Missing wallet_balance for {coin}"
    
    return balance

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def test_market_data(client, symbol: str) -> Dict[str, Any]:
    """Test market data endpoints with retries"""
    # Test order book
    order_book = client.get_order_book(symbol=symbol)
    assert "error" not in order_book, f"Error getting orderbook: {order_book.get('error')}"
    assert "bids" in order_book and "asks" in order_book, "Missing bids/asks in orderbook"
    assert len(order_book["bids"]) > 0, "Empty bids in orderbook"
    assert len(order_book["asks"]) > 0, "Empty asks in orderbook"
    
    # Test klines (candlestick) data
    klines = client.get_klines(symbol=symbol, interval="15", limit=5)
    assert klines and len(klines) > 0, "No klines data received"
    assert all(k.get("timestamp") for k in klines), "Invalid kline data structure"
    assert len(klines) == 5, f"Expected 5 klines, got {len(klines)}"
    
    # Test kline data structure
    for kline in klines:
        assert "timestamp" in kline, "Missing timestamp in kline"
        assert "open" in kline, "Missing open in kline"
        assert "high" in kline, "Missing high in kline"
        assert "low" in kline, "Missing low in kline"
        assert "close" in kline, "Missing close in kline"
        assert "volume" in kline, "Missing volume in kline"
    
    return {"order_book": order_book, "klines": klines}

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def test_websocket(client, symbol: str):
    """Test WebSocket functionality"""
    received_data = []
    
    def callback(data):
        received_data.append(data)
    
    # Subscribe to orderbook updates
    client.subscribe_to_orderbook(symbol, callback)
    
    # Wait for some data
    time.sleep(5)
    
    # Cleanup
    client.close_websocket()
    
    assert len(received_data) > 0, "No WebSocket data received"
    for data in received_data:
        assert client._validate_websocket_data(data), "Invalid WebSocket data format"

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def test_rate_limiting(client):
    """Test rate limiting functionality"""
    # Make multiple rapid requests to trigger rate limiting
    for _ in range(5):
        response = client.get_server_time()
        assert validate_response(response), "Rate limited request failed"
        time.sleep(0.1)  # Small delay to avoid overwhelming the API

def run_all_tests():
    """Run all test cases"""
    # Ładowanie zmiennych środowiskowych
    load_dotenv()

    print("\n==== Test połączenia z Bybit API ====\n")

    # Sprawdzanie zmiennych środowiskowych
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"

    if not api_key or not api_secret:
        print("❌ BŁĄD: Brak kluczy API w zmiennych środowiskowych.")
        print("Upewnij się, że masz plik .env z poprawnymi kluczami.")
        return

    # Wyświetlanie informacji o środowisku
    env_type = "TESTNET" if use_testnet else "PRODUKCYJNE"
    masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 4)}" if api_key else "Brak klucza"
    masked_secret = f"{api_secret[:4]}{'*' * (len(api_secret) - 4)}" if api_secret else "Brak sekretu"

    print(f"🔑 Klucz API: {masked_key}")
    print(f"🔒 Sekret API: {masked_secret}")
    print(f"🌐 Środowisko: {env_type}")

    try:
        from data.execution.bybit_connector import BybitConnector

        print("\n📡 Inicjalizacja klienta Bybit...")
        client = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet
        )

        success_count = 0
        total_tests = 6  # Updated number of tests

        # Test 1: Client initialization
        print("\n🔧 Test 1: Inicjalizacja klienta...")
        test_initialization(client)
        print("✅ Inicjalizacja klienta zakończona pomyślnie")
        success_count += 1

        # Test 2: Server time
        print("\n🕒 Test 2: Pobieranie czasu serwera...")
        server_time = test_server_time(client)
        print(f"✅ Czas serwera: {server_time.get('time', 'brak')}")
        print(f"✅ Źródło czasu: {server_time.get('source', 'brak')}")
        success_count += 1

        # Test 3: Account balance
        print("\n💰 Test 3: Pobieranie salda konta...")
        account_balance = test_account_balance(client)
        if account_balance.get("balances"):
            print("✅ Saldo konta pobrane pomyślnie:")
            for coin, balance in account_balance.get("balances", {}).items():
                print(f"   {coin}: {balance.get('equity', 0)} (dostępne: {balance.get('available_balance', 0)})")
            success_count += 1
        else:
            print(f"❌ Błąd podczas pobierania salda konta: {account_balance.get('error', 'Nieznany błąd')}")

        # Test 4: Market data
        print("\n📊 Test 4: Pobieranie danych rynkowych...")
        symbol = "BTCUSDT"
        try:
            market_data = test_market_data(client, symbol)
            order_book = market_data["order_book"]
            klines = market_data["klines"]
            
            print(f"✅ Książka zleceń dla {symbol} pobrana pomyślnie")
            print(f"   Liczba ofert kupna: {len(order_book.get('bids', []))}")
            print(f"   Liczba ofert sprzedaży: {len(order_book.get('asks', []))}")
            
            print(f"✅ Dane świecowe dla {symbol} pobrane pomyślnie")
            print(f"   Liczba świec: {len(klines)}")
            success_count += 1
        except Exception as e:
            print(f"❌ Błąd podczas pobierania danych rynkowych: {e}")

        # Test 5: WebSocket
        print("\n🔌 Test 5: Testowanie WebSocket...")
        try:
            test_websocket(client, symbol)
            print("✅ Test WebSocket zakończony pomyślnie")
            success_count += 1
        except Exception as e:
            print(f"❌ Błąd podczas testu WebSocket: {e}")

        # Test 6: Rate limiting
        print("\n⚡ Test 6: Testowanie limitów API...")
        try:
            test_rate_limiting(client)
            print("✅ Test limitów API zakończony pomyślnie")
            success_count += 1
        except Exception as e:
            print(f"❌ Błąd podczas testu limitów API: {e}")

        # Podsumowanie
        print(f"\n==== Podsumowanie testów ====")
        print(f"✅ Pomyślnie wykonane testy: {success_count}/{total_tests}")
        if success_count == total_tests:
            print("🎉 Wszystkie testy zakończone sukcesem!")
        else:
            print(f"⚠️ {total_tests - success_count} testów nie powiodło się")

    except ImportError as e:
        print(f"❌ Błąd importu modułu BybitConnector: {e}")
        print("Upewnij się, że struktura projektu jest poprawna.")
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    run_all_tests()