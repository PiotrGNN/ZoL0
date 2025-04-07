
#!/usr/bin/env python
"""
Skrypt diagnostyczny do testowania połączenia z API Bybit.
Sprawdza połączenie, załadowanie zmiennych środowiskowych i zwracane dane.
"""
import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api_test.log")
    ]
)
logger = logging.getLogger(__name__)

def test_env_vars():
    """Testuje zmienne środowiskowe."""
    try:
        # Spróbuj załadować dotenv
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("✅ python-dotenv załadowany poprawnie")
        except ImportError:
            logger.error("❌ Nie można zaimportować dotenv. Instaluję...")
            os.system(f"{sys.executable} -m pip install python-dotenv")
            try:
                from dotenv import load_dotenv
                load_dotenv()
                logger.info("✅ python-dotenv zainstalowany i załadowany")
            except ImportError:
                logger.error("❌ Instalacja dotenv nie powiodła się")
                return False
        
        # Kluczowe zmienne środowiskowe do sprawdzenia
        key_vars = [
            "BYBIT_API_KEY", 
            "BYBIT_API_SECRET", 
            "BYBIT_USE_TESTNET",
            "API_MIN_INTERVAL",
            "API_MAX_CALLS_PER_MINUTE"
        ]
        
        all_vars_present = True
        
        # Sprawdź zmienne
        for var in key_vars:
            value = os.environ.get(var)
            if value:
                # Zamaskuj wrażliwe dane
                if "API_KEY" in var or "SECRET" in var:
                    masked = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"
                    logger.info(f"✅ {var}: {masked}")
                else:
                    logger.info(f"✅ {var}: {value}")
            else:
                logger.error(f"❌ {var}: BRAK")
                all_vars_present = False
        
        return all_vars_present
    
    except Exception as e:
        logger.error(f"❌ Błąd podczas testowania zmiennych środowiskowych: {e}")
        return False

def test_api_connection():
    """Testuje połączenie z API Bybit."""
    try:
        # Spróbuj zaimportować BybitConnector
        try:
            from data.execution.bybit_connector import BybitConnector
        except ImportError:
            logger.error("❌ Nie udało się zaimportować BybitConnector. Sprawdź ścieżki importu.")
            return False
        
        # Pobierz zmienne środowiskowe
        api_key = os.environ.get("BYBIT_API_KEY")
        api_secret = os.environ.get("BYBIT_API_SECRET")
        use_testnet = os.environ.get("BYBIT_USE_TESTNET", "true").lower() == "true"
        
        if not api_key or not api_secret:
            logger.error("❌ Brak kluczy API. Sprawdź zmienne środowiskowe.")
            return False
        
        # Inicjalizuj connector
        logger.info("🔄 Inicjalizacja BybitConnector...")
        connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet,
            lazy_connect=False
        )
        
        # Test 1: Czas serwera
        logger.info("🔄 Test 1: Pobieranie czasu serwera...")
        server_time = connector.get_server_time()
        logger.info(f"✅ Czas serwera: {json.dumps(server_time, indent=2)}")
        
        # Test 2: Saldo konta
        logger.info("🔄 Test 2: Pobieranie salda konta...")
        time.sleep(5)  # Opóźnienie dla uniknięcia rate limiting
        balance = connector.get_account_balance()
        logger.info(f"✅ Saldo konta: {json.dumps(balance, indent=2)}")
        
        # Test 3: Dane rynkowe
        logger.info("🔄 Test 3: Pobieranie danych rynkowych...")
        time.sleep(5)  # Opóźnienie dla uniknięcia rate limiting
        symbol = "BTCUSDT"
        klines = connector.get_klines(symbol=symbol, limit=5)
        logger.info(f"✅ Dane świecowe ({len(klines)} świec): {json.dumps(klines[:2], indent=2)}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Błąd podczas testowania API: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🔍 DIAGNOSTYKA POŁĄCZENIA API BYBIT")
    logger.info(f"📅 Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    # Sprawdź zmienne środowiskowe
    logger.info("\n📋 SPRAWDZANIE ZMIENNYCH ŚRODOWISKOWYCH")
    env_result = test_env_vars()
    if env_result:
        logger.info("✅ Wszystkie wymagane zmienne środowiskowe są ustawione")
    else:
        logger.warning("⚠️ Brakuje niektórych zmiennych środowiskowych")
    
    # Testuj połączenie API
    logger.info("\n📡 TESTOWANIE POŁĄCZENIA API")
    api_result = test_api_connection()
    if api_result:
        logger.info("✅ Testy API zakończone sukcesem")
    else:
        logger.error("❌ Testy API nie powiodły się")
    
    logger.info("=" * 50)
    logger.info("🏁 KONIEC DIAGNOSTYKI API")
    logger.info("=" * 50)
