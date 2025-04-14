
#!/usr/bin/env python3
"""
run.py - Skrypt do lokalnego uruchomienia systemu tradingowego
"""

import os
import sys
import logging
import subprocess
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Przygotowanie środowiska do uruchomienia."""
    # Upewnij się, że potrzebne katalogi istnieją
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("static/img", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("python_libs/__pycache__", exist_ok=True)
    os.makedirs("ai_models/__pycache__", exist_ok=True)
    
    # Sprawdź czy mamy plik .env, jeśli nie - utwórz z .env.example
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        logger.info("Tworzenie pliku .env na podstawie .env.example")
        with open('.env.example', 'r') as src, open('.env', 'w') as dest:
            dest.write(src.read())
    
    # Załaduj zmienne środowiskowe
    load_dotenv()
    
    # Sprawdź ustawienia API
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
    
    logger.info(f"Tryb API: {'Testnet' if testnet else 'Produkcja'}")
    if api_key and api_secret:
        logger.info(f"Klucz API: {api_key[:4]}{'*' * (len(api_key) - 4)}")
    else:
        logger.warning("Brak kluczy API. System będzie działał w trybie symulacji.")
    
    return True

def run_application():
    """Uruchomienie głównej aplikacji."""
    try:
        import main
        logger.info("Aplikacja uruchomiona pomyślnie.")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania aplikacji: {e}")
        return False

if __name__ == "__main__":
    logger.info("Uruchamianie lokalnego systemu tradingowego...")
    
    # Przygotowanie środowiska
    if not setup_environment():
        logger.error("Błąd podczas przygotowywania środowiska.")
        sys.exit(1)
    
    # Uruchomienie aplikacji
    logger.info("Uruchamianie aplikacji...")
    if not run_application():
        logger.error("Błąd podczas uruchamiania aplikacji.")
        sys.exit(1)
    
    logger.info("System zakończył działanie.")
