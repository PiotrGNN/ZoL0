
#!/usr/bin/env python3
"""
Skrypt do zestawiania tunelu SSH SOCKS5 proxy
Umożliwia przekierowanie ruchu API przez zewnętrzny serwer VPS
"""

import os
import sys
import time
import logging
import subprocess
import signal
import atexit
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ssh_tunnel')

def load_config():
    """Ładuje konfigurację z .env lub zmiennych środowiskowych"""
    load_dotenv()
    
    # Odczytaj zmienne konfiguracyjne
    config = {
        'VPS_USER': os.getenv('VPS_USER'),
        'VPS_HOST': os.getenv('VPS_HOST'),
        'VPS_PORT': os.getenv('VPS_PORT', '22'),
        'VPS_KEY_PATH': os.getenv('VPS_KEY_PATH', None),
        'SOCKS_PORT': os.getenv('SOCKS_PORT', '1080')
    }
    
    # Sprawdź czy kluczowe zmienne są ustawione
    if not config['VPS_USER'] or not config['VPS_HOST']:
        logger.error("Brak wymaganych zmiennych środowiskowych: VPS_USER, VPS_HOST")
        logger.info("Dodaj te zmienne do pliku .env lub ustaw je w środowisku")
        sys.exit(1)
        
    return config

def start_ssh_tunnel(config):
    """Uruchamia tunel SSH SOCKS5"""
    ssh_cmd = ['ssh', '-N', '-D', config['SOCKS_PORT']]
    
    # Dodaj ścieżkę do klucza SSH, jeśli podano
    if config['VPS_KEY_PATH']:
        ssh_cmd.extend(['-i', config['VPS_KEY_PATH']])
    
    # Dodaj port SSH, jeśli inny niż domyślny
    if config['VPS_PORT'] != '22':
        ssh_cmd.extend(['-p', config['VPS_PORT']])
    
    # Dodaj adres serwera docelowego
    ssh_cmd.append(f"{config['VPS_USER']}@{config['VPS_HOST']}")
    
    logger.info(f"Uruchamianie tunelu SSH SOCKS5 na porcie {config['SOCKS_PORT']}")
    logger.info(f"Komenda: {' '.join(ssh_cmd)}")
    
    # Uruchom proces SSH
    process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    # Funkcja do czyszczenia przy wyjściu
    def cleanup():
        if process.poll() is None:
            logger.info("Zatrzymywanie tunelu SSH")
            process.terminate()
            process.wait()
    
    # Zarejestruj funkcję czyszczącą przy wyjściu
    atexit.register(cleanup)
    
    # Obsługa sygnałów SIGINT i SIGTERM
    def signal_handler(sig, frame):
        logger.info(f"Otrzymano sygnał {sig}, zamykanie tunelu")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    return process

def test_connection():
    """Testuje połączenie przez proxy"""
    import requests
    
    try:
        proxies = {
            'http': 'socks5h://127.0.0.1:1080',
            'https': 'socks5h://127.0.0.1:1080'
        }
        
        logger.info("Testowanie połączenia przez proxy...")
        response = requests.get('https://api.bybit.com/v5/market/time', 
                                proxies=proxies, 
                                timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Połączenie działa! Czas serwera: {data}")
            return True
        else:
            logger.error(f"Błąd połączenia: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Nie można połączyć się przez proxy: {e}")
        return False

def main():
    """Główna funkcja skryptu"""
    logger.info("Konfiguracja tunelu SSH SOCKS5 proxy")
    
    # Załaduj konfigurację
    config = load_config()
    
    # Uruchom tunel SSH
    process = start_ssh_tunnel(config)
    
    # Poczekaj na ustanowienie tunelu
    logger.info("Czekam 5 sekund na ustanowienie tunelu...")
    time.sleep(5)
    
    # Przeprowadź test połączenia
    if test_connection():
        logger.info("Tunel SSH SOCKS5 działa poprawnie!")
    else:
        logger.warning("Test połączenia nie powiódł się, ale tunel może nadal działać")
    
    logger.info(f"Tunel SSH SOCKS5 uruchomiony na porcie {config['SOCKS_PORT']}")
    logger.info("Naciśnij Ctrl+C, aby zakończyć")
    
    # Czekaj na zakończenie procesu
    try:
        process.wait()
    except KeyboardInterrupt:
        logger.info("Otrzymano Ctrl+C, kończenie pracy")
    
    # Sprawdź kod wyjścia
    if process.returncode != 0:
        stderr = process.stderr.read().decode('utf-8')
        logger.error(f"Tunel SSH zakończył się z błędem (kod {process.returncode}): {stderr}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
