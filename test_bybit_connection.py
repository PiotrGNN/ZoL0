
"""
test_bybit_connection.py
-----------------------
Prosty skrypt do testowania połączenia z API Bybit.
"""

import os
import sys
import time
import logging
import json
import hmac
import hashlib
import requests
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

# Dodaj katalog główny do ścieżki Pythona
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Ładowanie zmiennych środowiskowych
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("Nie można załadować dotenv. Upewnij się, że zainstalowano python-dotenv.")
    logging.info("Instalowanie python-dotenv...")
    os.system("pip install python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

def get_bybit_server_time(use_testnet=True):
    """Pobiera czas serwera Bybit."""
    base_url = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"
    
    try:
        # Najpierw próbujemy z V5 API
        response = requests.get(f"{base_url}/v5/market/time", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data["retCode"] == 0:
                time_nano = int(data["result"]["timeNano"])
                server_time = time_nano // 1000000
                return {
                    "success": True,
                    "time_ms": server_time,
                    "time": datetime.fromtimestamp(server_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "v5_api"
                }
        
        # Jeśli V5 nie zadziała, próbujemy z Spot API
        response = requests.get(f"{base_url}/spot/v1/time", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data["ret_code"] == 0:
                server_time = data["serverTime"]
                return {
                    "success": True,
                    "time_ms": server_time,
                    "time": datetime.fromtimestamp(server_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "spot_api"
                }
        
        # Jeśli nic nie zadziała, zwracamy błąd
        return {
            "success": False,
            "error": f"Błąd HTTP: {response.status_code}",
            "response": response.text
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def test_auth(api_key, api_secret, use_testnet=True):
    """Testuje autentykację do API Bybit."""
    base_url = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"
    
    try:
        # Parametry dla podpisu
        timestamp = str(int(time.time() * 1000))
        params = {"accountType": "UNIFIED"}
        param_str = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
        
        # Tworzenie podpisu
        pre_sign = f"{timestamp}{api_key}{param_str}"
        signature = hmac.new(
            bytes(api_secret, 'utf-8'),
            bytes(pre_sign, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Nagłówki
        headers = {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "20000"
        }
        
        # Wysłanie żądania
        url = f"{base_url}/v5/account/wallet-balance"
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data["retCode"] == 0:
                return {
                    "success": True,
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "error": f"Błąd API: {data['retCode']} - {data['retMsg']}",
                    "response": data
                }
        else:
            return {
                "success": False,
                "error": f"Błąd HTTP: {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Główna funkcja testu."""
    # Sprawdzanie zmiennych środowiskowych
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    use_testnet_str = os.getenv("BYBIT_USE_TESTNET", "true")
    use_testnet = use_testnet_str.lower() in ["true", "1", "yes", "t"]
    
    if not api_key or not api_secret:
        logging.error("Brak kluczy API w zmiennych środowiskowych. Sprawdź plik .env")
        return
    
    # Maskowanie kluczy dla logów
    masked_key = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"
    masked_secret = f"{api_secret[:4]}{'*' * (len(api_secret) - 8)}{api_secret[-4:]}"
    
    logging.info(f"Testowanie połączenia z Bybit API")
    logging.info(f"Tryb: {'Testnet' if use_testnet else 'Produkcja'}")
    logging.info(f"API Key: {masked_key}")
    logging.info(f"API Secret: {masked_secret}")
    
    # Test czasu serwera
    server_time = get_bybit_server_time(use_testnet)
    if server_time["success"]:
        logging.info(f"Czas serwera: {server_time['time']} ({server_time['source']})")
    else:
        logging.error(f"Błąd podczas pobierania czasu serwera: {server_time['error']}")
        return
    
    # Test autentykacji
    auth_result = test_auth(api_key, api_secret, use_testnet)
    if auth_result["success"]:
        logging.info("Autentykacja powiodła się!")
        
        # Wyświetlenie sald
        try:
            balances = {}
            if "result" in auth_result["data"] and "list" in auth_result["data"]["result"]:
                for account in auth_result["data"]["result"]["list"]:
                    if "coin" in account:
                        for coin in account["coin"]:
                            if coin["availableToWithdraw"] != "0":
                                coin_name = coin["coin"]
                                balances[coin_name] = {
                                    "balance": coin["walletBalance"],
                                    "available": coin["availableToWithdraw"]
                                }
            
            if balances:
                logging.info("Dostępne salda:")
                for coin, data in balances.items():
                    logging.info(f"  {coin}: {data['balance']} (dostępne: {data['available']})")
            else:
                logging.info("Brak dostępnych sald lub serwer zwrócił pustą listę")
        except Exception as e:
            logging.error(f"Błąd podczas przetwarzania sald: {e}")
    else:
        logging.error(f"Błąd podczas autentykacji: {auth_result['error']}")

if __name__ == "__main__":
    main()
