#!/usr/bin/env python3
"""
Skrypt testujący połączenie z API Bybit.
Pozwala na diagnozę problemów z połączeniem.
"""
import os
import logging
import json
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/api_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Zapewnienie istnienia katalogu logs
os.makedirs("logs", exist_ok=True)

def test_environment_variables():
    """Sprawdza, czy zmienne środowiskowe są poprawnie skonfigurowane."""
    # Próba ładowania .env z kilku lokalizacji
    load_dotenv(override=True)
    load_dotenv('.env', override=True)

    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"

    if not api_key or not api_secret:
        logger.error("❌ Brak kluczy API Bybit w zmiennych środowiskowych")
        print("\n❌ BŁĄD: Brak kluczy API w zmiennych środowiskowych")
        print("Upewnij się, że plik .env zawiera:")
        print("BYBIT_API_KEY=twój_klucz")
        print("BYBIT_API_SECRET=twój_sekret")
        print("BYBIT_USE_TESTNET=true  # lub false dla produkcji\n")

        # Stwórz przykładowy plik .env, jeśli nie istnieje
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write("BYBIT_API_KEY=twój_klucz_api\n")
                f.write("BYBIT_API_SECRET=twój_sekret_api\n")
                f.write("BYBIT_USE_TESTNET=true\n")
                f.write("FLASK_APP=main.py\n")
                f.write("FLASK_DEBUG=true\n")
                f.write("PORT=8080\n")
            logger.info("📝 Utworzono przykładowy plik .env")
            print("📝 Utworzono przykładowy plik .env. Uzupełnij go swoimi danymi.")

        return False

    # Maskowanie kluczy w logach
    masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if api_key else "Brak klucza"
    masked_secret = f"{api_secret[:4]}{'*' * (len(api_secret) - 8)}{api_secret[-4:]}" if api_secret else "Brak sekretu"

    logger.info(f"✅ Znaleziono klucze API: {masked_key}")
    logger.info(f"🌐 Tryb API: {'TESTNET' if use_testnet else 'PRODUKCJA'}")

    print(f"\n✅ Znaleziono klucze API Bybit: {masked_key}")
    print(f"🌐 Tryb API: {'TESTNET' if use_testnet else 'PRODUKCJA'}")

    return True

def test_api_connection():
    """Testuje połączenie z API Bybit."""
    # Ładowanie modułu BybitConnector
    try:
        from data.execution.bybit_connector import BybitConnector

        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"

        logger.info("🔄 Inicjalizacja klienta Bybit API...")
        connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet
        )

        # Test połączenia bez autentykacji (czas serwera)
        logger.info("🔄 Testowanie połączenia podstawowego (czas serwera)...")
        server_time = connector.get_server_time()

        # Test połączenia z autentykacją (saldo konta)
        logger.info("🔄 Testowanie połączenia z autentykacją (saldo konta)...")
        account_balance = connector.get_account_balance()

        # Zapisujemy wyniki do pliku dla debugowania
        results = {
            "server_time": server_time,
            "account_info": account_balance,
            "testnet": use_testnet
        }

        with open('logs/bybit_connection_test.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Wyświetlamy wyniki
        print("\n🔍 WYNIKI TESTU API:\n")
        print(f"✅ Czas serwera Bybit: {server_time}")

        if account_balance.get("success") is True:
            print("\n✅ Pomyślnie pobrano informacje o koncie")
            print("\nDostępne salda:")
            for currency, balance in account_balance.get("balances", {}).items():
                print(f"  • {currency}: {balance.get('available_balance', 0)}")
        else:
            print("\n❌ Błąd podczas pobierania informacji o koncie:")
            print(f"  • {account_balance.get('error', 'Nieznany błąd')}")

        print(f"\n📋 Pełne wyniki zapisano w pliku logs/bybit_connection_test.json")

        return True

    except Exception as e:
        logger.error(f"❌ Błąd podczas testowania API: {e}", exc_info=True)
        print(f"\n❌ BŁĄD POŁĄCZENIA: {e}")
        return False

if __name__ == "__main__":
    print("\n🔌 TEST POŁĄCZENIA Z BYBIT API 🔌\n")

    if test_environment_variables():
        if test_api_connection():
            print("\n✅ TEST ZAKOŃCZONY SUKCESEM: Połączenie z Bybit API działa poprawnie!\n")
        else:
            print("\n❌ TEST NIEUDANY: Wystąpił błąd podczas łączenia z API.\n")
    else:
        print("\n❌ TEST NIEUDANY: Błąd w konfiguracji zmiennych środowiskowych.\n")

    print("📌 Sprawdź logs/api_test.log, aby zobaczyć szczegóły.")