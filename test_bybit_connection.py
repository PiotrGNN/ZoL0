"""
test_bybit_connection.py
-----------------------
Skrypt do testowania połączenia z API Bybit.
"""

import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Konfiguracja podstawowego loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api_test.log"),
        logging.StreamHandler()
    ]
)

def main():
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

    # Próba zaimportowania BybitConnector
    try:
        from data.execution.bybit_connector import BybitConnector

        # Inicjalizacja klienta
        print("\n📡 Inicjalizacja klienta Bybit...")
        client = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet
        )

        # Test 1: Czas serwera
        print("\n🕒 Test 1: Pobieranie czasu serwera...")
        server_time = client.get_server_time()
        if server_time.get("success", False):
            print(f"✅ Czas serwera: {server_time.get('time', 'brak')}")
            print(f"✅ Źródło czasu: {server_time.get('source', 'brak')}")
        else:
            print(f"❌ Błąd podczas pobierania czasu serwera: {server_time}")

        # Test 2: Saldo konta
        print("\n💰 Test 2: Pobieranie salda konta...")
        account_balance = client.get_account_balance()
        if account_balance.get("success", False) or account_balance.get("balances"):
            print("✅ Saldo konta pobrane pomyślnie:")
            for coin, balance in account_balance.get("balances", {}).items():
                print(f"   {coin}: {balance.get('equity', 0)} (dostępne: {balance.get('available_balance', 0)})")
        else:
            print(f"❌ Błąd podczas pobierania salda konta: {account_balance.get('error', 'Nieznany błąd')}")
            print(f"   Źródło danych: {account_balance.get('source', 'brak')}")
            print(f"   Notatka: {account_balance.get('note', 'brak')}")

        # Test 3: Dane rynkowe
        print("\n📊 Test 3: Pobieranie danych rynkowych...")
        symbol = "BTCUSDT"
        try:
            # Książka zleceń
            order_book = client.get_order_book(symbol=symbol)
            if "error" not in order_book:
                print(f"✅ Książka zleceń dla {symbol} pobrana pomyślnie.")
                print(f"   Liczba ofert kupna: {len(order_book.get('bids', []))}")
                print(f"   Liczba ofert sprzedaży: {len(order_book.get('asks', []))}")
            else:
                print(f"❌ Błąd podczas pobierania książki zleceń: {order_book.get('error', 'Nieznany błąd')}")

            # Świece
            klines = client.get_klines(symbol=symbol, interval="15", limit=5)
            if klines and len(klines) > 0:
                print(f"✅ Dane świecowe dla {symbol} pobrane pomyślnie.")
                print(f"   Liczba świec: {len(klines)}")
                print(f"   Ostatnia świeca: {klines[-1]['datetime']} - Open: {klines[-1]['open']}, Close: {klines[-1]['close']}")
            else:
                print(f"❌ Błąd podczas pobierania danych świecowych: {klines}")
        except Exception as e:
            print(f"❌ Błąd podczas pobierania danych rynkowych: {e}")

        print("\n==== Podsumowanie ====")
        print(f"🌐 API Bybit ({env_type}): {'✅ Połączenie działa' if server_time.get('success', False) else '❌ Problem z połączeniem'}")
        print(f"🔑 Autoryzacja API: {'✅ Działa poprawnie' if account_balance.get('balances') else '❌ Problem z autoryzacją'}")

    except ImportError as e:
        print(f"❌ Błąd importu modułu BybitConnector: {e}")
        print("Upewnij się, że struktura projektu jest poprawna.")
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    main()