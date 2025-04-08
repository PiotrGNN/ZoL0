
#!/usr/bin/env python3
"""
Skrypt do testowania połączenia przez SOCKS5 proxy
"""

import requests
import json
import logging
import os
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('proxy_test')

# Załaduj zmienne środowiskowe
load_dotenv()
proxy_url = os.getenv('PROXY_URL', 'socks5h://127.0.0.1:1080')

def test_direct_connection():
    """Test połączenia bezpośredniego do API Bybit"""
    try:
        logger.info("Test połączenia bezpośredniego do API Bybit...")
        response = requests.get('https://api.bybit.com/v5/market/time', timeout=10)
        
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Odpowiedź: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Błąd połączenia bezpośredniego: {e}")
        return False

def test_proxy_connection():
    """Test połączenia przez proxy do API Bybit"""
    try:
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        
        logger.info(f"Test połączenia przez proxy {proxy_url} do API Bybit...")
        response = requests.get('https://api.bybit.com/v5/market/time', 
                               proxies=proxies, 
                               timeout=10)
        
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Odpowiedź: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Błąd połączenia przez proxy: {e}")
        return False

def check_my_ip():
    """Sprawdza publiczny adres IP (z i bez proxy)"""
    try:
        # Sprawdź IP bez proxy
        logger.info("Sprawdzanie publicznego IP bez proxy...")
        direct_ip = requests.get('https://api.ipify.org?format=json', timeout=10).json()
        logger.info(f"Adres IP bez proxy: {direct_ip['ip']}")
        
        # Sprawdź IP z proxy
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"Sprawdzanie publicznego IP przez proxy {proxy_url}...")
        proxy_ip = requests.get('https://api.ipify.org?format=json', 
                               proxies=proxies, 
                               timeout=10).json()
        logger.info(f"Adres IP przez proxy: {proxy_ip['ip']}")
        
        # Porównaj adresy
        if direct_ip['ip'] != proxy_ip['ip']:
            logger.info("✅ Proxy działa poprawnie! Adresy IP są różne.")
            return True
        else:
            logger.warning("⚠️ Proxy może nie działać poprawnie - te same adresy IP.")
            return False
    except Exception as e:
        logger.error(f"Błąd podczas sprawdzania IP: {e}")
        return False

def main():
    """Główna funkcja skryptu"""
    logger.info("=== TEST POŁĄCZENIA PROXY ===")
    
    direct_ok = test_direct_connection()
    proxy_ok = test_proxy_connection()
    
    logger.info("\n=== WYNIKI TESTÓW ===")
    logger.info(f"Połączenie bezpośrednie: {'✅ OK' if direct_ok else '❌ BŁĄD'}")
    logger.info(f"Połączenie przez proxy: {'✅ OK' if proxy_ok else '❌ BŁĄD'}")
    
    logger.info("\n=== TEST ADRESU IP ===")
    check_my_ip()
    
    if proxy_ok:
        logger.info("\n✅ Proxy SOCKS5 działa poprawnie!")
    else:
        logger.error("\n❌ Proxy SOCKS5 nie działa. Sprawdź konfigurację tunelu SSH.")

if __name__ == "__main__":
    main()
