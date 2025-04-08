
#!/usr/bin/env python3
"""
Skrypt do testowania połączenia z Bybit API
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
        logging.FileHandler("logs/bybit_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Dodanie katalogu głównego do ścieżki Pythona
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_environment_variables():
    """Ładuje zmienne środowiskowe z .env"""
    try:
        from dotenv import load_dotenv
        # Próbujemy różnych sposobów załadowania zmiennych
        load_dotenv(override=True)  # Standardowe podejście
        
        # Sprawdzenie czy zmienne zostały załadowane
        api_key = os.getenv("BYBIT_API_KEY")
        if not api_key:
            # Próba alternatywnego podejścia
            dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path=dotenv_path, override=True)
                logger.info(f"Załadowano zmienne ze ścieżki: {dotenv_path}")
        
        # Logowanie stanu zmiennych (z maskowaniem)
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"
        
        masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 4)}" if len(api_key) > 4 else "Brak klucza"
        masked_secret = f"{api_secret[:4]}{'*' * (len(api_secret) - 4)}" if len(api_secret) > 4 else "Brak sekretu"
        
        logger.info(f"API Key: {masked_key}")
        logger.info(f"API Secret: {masked_secret}")
        logger.info(f"Testnet: {use_testnet}")
        
        return api_key, api_secret, use_testnet
    except Exception as e:
        logger.error(f"Błąd podczas ładowania zmiennych środowiskowych: {e}")
        traceback.print_exc()
        return None, None, True

def test_direct_http_request():
    """Test bezpośredniego zapytania HTTP do API Bybit"""
    try:
        import requests
        import hmac
        import hashlib
        import time
        
        # Pobierz zmienne środowiskowe
        api_key, api_secret, use_testnet = load_environment_variables()
        
        # Wybierz odpowiedni endpoint
        base_url = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"
        
        # 1. Test publicznego endpointu (bez autoryzacji)
        logger.info("Test 1: Publiczny endpoint (czas serwera)")
        v5_time_endpoint = f"{base_url}/v5/market/time"
        
        try:
            response = requests.get(v5_time_endpoint, timeout=10)
            logger.info(f"Status: {response.status_code}")
            logger.info(f"Odpowiedź: {response.text}")
            
            if response.status_code == 200:
                logger.info("✅ Test publicznego endpointu udany")
            else:
                logger.warning(f"⚠️ Test publicznego endpointu nieudany: {response.text}")
        except Exception as e:
            logger.error(f"❌ Błąd podczas testu publicznego endpointu: {e}")
        
        # 2. Test prywatnego endpointu (z autoryzacją)
        if api_key and api_secret:
            logger.info("\nTest 2: Prywatny endpoint (saldo konta)")
            v5_balance_endpoint = f"{base_url}/v5/account/wallet-balance"
            
            # Przygotowanie nagłówków z podpisem
            timestamp = str(int(time.time() * 1000))
            signature_payload = timestamp + api_key + "20000"  # recv_window=20000
            signature = hmac.new(
                bytes(api_secret, 'utf-8'),
                bytes(signature_payload, 'utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers = {
                "X-BAPI-API-KEY": api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": "20000",
                "X-BAPI-SIGN": signature
            }
            
            try:
                response = requests.get(v5_balance_endpoint, headers=headers, timeout=10)
                logger.info(f"Status: {response.status_code}")
                logger.info(f"Odpowiedź: {response.text[:200]}...")  # Wyświetl pierwsze 200 znaków
                
                if response.status_code == 200:
                    logger.info("✅ Test prywatnego endpointu udany")
                else:
                    logger.warning(f"⚠️ Test prywatnego endpointu nieudany: {response.text}")
            except Exception as e:
                logger.error(f"❌ Błąd podczas testu prywatnego endpointu: {e}")
        else:
            logger.warning("⚠️ Pominięto test prywatnego endpointu - brak kluczy API")
        
        return True
    except Exception as e:
        logger.error(f"❌ Wystąpił błąd podczas testu bezpośredniego HTTP: {e}")
        traceback.print_exc()
        return False

def test_bybit_connector():
    """Test używając BybitConnector"""
    try:
        # Importuj BybitConnector
        from data.execution.bybit_connector import BybitConnector
        
        # Pobierz zmienne środowiskowe
        api_key, api_secret, use_testnet = load_environment_variables()
        
        # Inicjalizacja konektora
        logger.info("\nTest 3: BybitConnector")
        logger.info(f"Inicjalizacja BybitConnector z parametrami: API Key: {api_key[:4]}*** Testnet: {use_testnet}")
        
        connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet,
            lazy_connect=False
        )
        
        # Test 1: Czas serwera
        logger.info("Test 3.1: Pobieranie czasu serwera")
        server_time = connector.get_server_time()
        logger.info(f"Wynik: {json.dumps(server_time, indent=2)}")
        
        # Jeśli jest API key, test 2: Saldo konta
        if api_key and api_secret:
            logger.info("Test 3.2: Pobieranie salda konta")
            time.sleep(5)  # Opóźnienie dla uniknięcia rate limiting
            balance = connector.get_account_balance()
            # Wyświetl tylko fragment wyniku dla bezpieczeństwa
            truncated_balance = {
                "success": balance.get("success", False),
                "source": balance.get("source", "unknown"),
                "balances_count": len(balance.get("balances", {})),
                "available_currencies": list(balance.get("balances", {}).keys())
            }
            logger.info(f"Wynik: {json.dumps(truncated_balance, indent=2)}")
        else:
            logger.warning("Pominięto test salda konta - brak kluczy API")
        
        return True
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas testu BybitConnector: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🔍 DIAGNOSTYKA POŁĄCZENIA API BYBIT")
    logger.info(f"📅 Data i czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    # Załaduj zmienne środowiskowe
    api_key, api_secret, use_testnet = load_environment_variables()
    
    # Pierwszy test - bezpośrednie zapytanie HTTP
    logger.info("\n🌐 TEST BEZPOŚREDNIEGO ZAPYTANIA HTTP")
    http_result = test_direct_http_request()
    
    # Drugi test - z wykorzystaniem BybitConnector
    logger.info("\n🔌 TEST PRZY UŻYCIU BYBITCONNECTOR")
    connector_result = test_bybit_connector()
    
    # Podsumowanie
    logger.info("\n" + "=" * 50)
    logger.info("📊 PODSUMOWANIE TESTÓW")
    logger.info(f"Bezpośrednie zapytanie HTTP: {'✅ SUKCES' if http_result else '❌ BŁĄD'}")
    logger.info(f"BybitConnector: {'✅ SUKCES' if connector_result else '❌ BŁĄD'}")
    logger.info("=" * 50)
    
    if http_result and connector_result:
        logger.info("🎉 WSZYSTKIE TESTY ZAKOŃCZONE SUKCESEM!")
    else:
        logger.warning("⚠️ NIEKTÓRE TESTY ZAKOŃCZYŁY SIĘ NIEPOWODZENIEM")
        
        # Sugestie rozwiązania problemów
        logger.info("\n🔧 SUGESTIE ROZWIĄZANIA PROBLEMÓW:")
        if use_testnet:
            logger.info("1. Obecnie używasz API testnet. Jeśli chcesz używać produkcyjnego API, ustaw BYBIT_USE_TESTNET=false w pliku .env")
        else:
            logger.info("1. Obecnie używasz produkcyjnego API. Jeśli masz problemy, spróbuj przełączyć się na testnet (BYBIT_USE_TESTNET=true)")
        
        logger.info("2. Sprawdź, czy klucze API są poprawne i mają odpowiednie uprawnienia")
        logger.info("3. Sprawdź połączenie internetowe i upewnij się, że firewall nie blokuje dostępu do API Bybit")
        logger.info("4. Jeśli używasz produkcyjnego API, upewnij się, że Twój adres IP jest dozwolony w ustawieniach API Bybit")
#!/usr/bin/env python3
"""
test_bybit_connection.py
----------------------
Prosty skrypt do testowania połączenia z API ByBit.
"""

import os
import logging
import json
from datetime import datetime

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

# Dodanie katalogu głównego do ścieżki Pythona
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ładowanie zmiennych środowiskowych
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    os.system("pip install python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

def test_bybit_connection():
    """Test połączenia z API ByBit."""
    try:
        from data.execution.bybit_connector import BybitConnector
        
        # Pobieranie kluczy API z zmiennych środowiskowych
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        use_testnet = os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true"
        
        if not api_key or not api_secret:
            print("❌ BŁĄD: Brak kluczy API ByBit w zmiennych środowiskowych")
            print("   Dodaj BYBIT_API_KEY i BYBIT_API_SECRET do pliku .env")
            return False
        
        # Informacja o środowisku
        env_type = "TESTNET" if use_testnet else "PRODUKCYJNYM"
        masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 4)}" if api_key else "Brak klucza"
        masked_secret = f"{api_secret[:4]}{'*' * (len(api_secret) - 4)}" if api_secret else "Brak sekretu"
        
        print(f"\n📡 Test połączenia z API ByBit ({env_type})")
        print(f"🔑 Klucz API: {masked_key}")
        print(f"🔐 Sekret API: {masked_secret}")
        
        # Inicjalizacja klienta ByBit
        bybit_client = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet
        )
        
        # Test połączenia - pobranie czasu serwera
        server_time = bybit_client.get_server_time()
        
        if "error" in server_time:
            print(f"❌ Błąd połączenia: {server_time['error']}")
            return False
        
        print(f"✅ Połączenie udane! Czas serwera: {server_time['time']}")
        
        # Test pobierania danych konta (wymaga uwierzytelnienia)
        balance = bybit_client.get_account_balance()
        
        if balance.get("success", False):
            print("✅ Uwierzytelnienie poprawne, pobrano dane konta")
            print(f"💰 Dostępne środki: {json.dumps(balance['balances'], indent=2)}")
        else:
            print(f"❌ Błąd pobierania danych konta: {balance.get('error', 'Nieznany błąd')}")
        
        return True
    
    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
        print("   Sprawdź instalację wymaganych pakietów")
        return False
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd: {e}")
        return False

if __name__ == "__main__":
    print("\n🚀 Test połączenia z API ByBit\n")
    
    # Utworzenie katalogu na logi, jeśli nie istnieje
    os.makedirs("logs", exist_ok=True)
    
    # Uruchomienie testu
    result = test_bybit_connection()
    
    if result:
        print("\n✅ Test zakończony pomyślnie")
    else:
        print("\n❌ Test zakończony niepowodzeniem")
