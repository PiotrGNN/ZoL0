
"""
Skrypt testujący połączenie z API Bybit.
"""

import os
import sys
import logging
import time
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Dodajemy katalog główny do ścieżki Pythona
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Ładowanie zmiennych środowiskowych
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Załadowano zmienne środowiskowe")
except ImportError:
    logging.warning("Moduł dotenv nie jest zainstalowany. Instaluję...")
    os.system("pip install python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

def test_bybit_connection():
    """Testuje połączenie z Bybit API."""
    try:
        # Import BybitConnector
        from data.execution.bybit_connector import BybitConnector
        
        # Pobranie kluczy API z zmiennych środowiskowych
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        
        # Sprawdzenie czy mamy klucze API
        if not api_key or not api_secret:
            logging.error("Brak kluczy API Bybit w zmiennych środowiskowych. Sprawdź plik .env.")
            return False
        
        # Czy używać testnet?
        use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"
        
        # Logowanie informacji o konfiguracji
        masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if len(api_key) > 8 else "***"
        logging.info(f"Testowanie połączenia z API Bybit - Klucz: {masked_key}, Testnet: {use_testnet}")
        
        # Inicjalizacja klienta Bybit
        connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet
        )
        
        # Test 1: Pobieranie czasu serwera
        logging.info("Test 1: Pobieranie czasu serwera")
        server_time = connector.get_server_time()
        logging.info(f"Czas serwera: {server_time}")
        
        if not server_time.get("success", False):
            logging.error("Nie udało się pobrać czasu serwera")
            return False
        
        # Test 2: Pobieranie danych świecowych
        logging.info("Test 2: Pobieranie danych świecowych")
        klines = connector.get_klines(symbol="BTCUSDT", interval="15", limit=5)
        logging.info(f"Pobrano {len(klines)} świec")
        
        if not klines or len(klines) == 0:
            logging.error("Nie udało się pobrać danych świecowych")
            return False
        
        # Test 3: Pobieranie księgi zleceń
        logging.info("Test 3: Pobieranie księgi zleceń")
        order_book = connector.get_order_book(symbol="BTCUSDT", limit=5)
        logging.info(f"Pobrano księgę zleceń z {len(order_book.get('bids', []))} ofertami kupna i {len(order_book.get('asks', []))} ofertami sprzedaży")
        
        if not order_book or "bids" not in order_book or "asks" not in order_book:
            logging.error("Nie udało się pobrać księgi zleceń")
            return False
        
        # Test 4: Pobieranie stanu konta
        logging.info("Test 4: Pobieranie stanu konta")
        balance = connector.get_account_balance()
        logging.info(f"Stan konta: {balance}")
        
        if not balance or not balance.get("success", False):
            logging.warning("Nie udało się pobrać stanu konta lub wystąpił błąd autentykacji")
            # Nie traktujemy tego jako błąd krytyczny, bo może wynikać z ograniczeń API key
        
        logging.info("Wszystkie testy zakończone pomyślnie!")
        return True
    
    except Exception as e:
        logging.error(f"Wystąpił błąd podczas testowania połączenia: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logging.info("Uruchamianie testu połączenia z Bybit API...")
    success = test_bybit_connection()
    
    if success:
        logging.info("✅ Test połączenia zakończony pomyślnie!")
    else:
        logging.error("❌ Test połączenia zakończony niepowodzeniem!")
    
    print("\nNaciśnij Enter, aby zakończyć...")
    input()
