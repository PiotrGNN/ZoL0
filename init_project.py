"""
init_project.py
--------------
Skrypt inicjalizujący projekt, instalujący zależności i konfigurujący środowisko.
"""

import os
import sys
import time
import json
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/init_project.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Tworzy wymagane katalogi jeśli nie istnieją."""
    directories = [
        "logs",
        "data/cache",
        "reports",
        "saved_models",
        "python_libs", # Added for potential library storage
        "data/execution",
        "data/indicators",
        "data/logging",
        "data/optimization",
        "data/risk_management",
        "data/strategies",
        "data/tests",
        "data/utils",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Utworzono katalog: {directory}")

def install_dependencies():
    """Instaluje wymagane zależności z kontrolą wersji."""
    logger.info("Instalowanie wymaganych pakietów...")

    # Instalacja pakietów z precyzyjnie określonymi wersjami
    os.system("pip install --upgrade requests==2.31.0")
    os.system("pip install --upgrade websocket-client==1.7.0")
    os.system("pip install --upgrade typing-extensions==4.5.0")
    os.system("pip install --upgrade pandas") #Example added dependency
    os.system("pip install --upgrade numpy") #Example added dependency


    # Instalacja pozostałych zależności z requirements.txt
    os.system("pip install -r requirements.txt")

    logger.info("Instalacja pakietów zakończona.")

def setup_env_file():
    """Tworzy plik .env jeśli nie istnieje."""
    if not os.path.exists(".env"):
        logger.info("Tworzenie pliku .env...")

        with open(".env.example", "r") as example_file:
            env_content = example_file.read()

        with open(".env", "w") as env_file:
            env_file.write(env_content)

        logger.info("Utworzono plik .env na podstawie .env.example")
        logger.info("WAŻNE: Edytuj plik .env, aby dodać prawdziwe klucze API")
    else:
        logger.info("Plik .env już istnieje")

def verify_api_keys():
    """Sprawdza, czy klucze API są poprawnie skonfigurowane."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        use_testnet = os.getenv("BYBIT_USE_TESTNET", "false").lower() == "true"

        # Sprawdź czy klucze istnieją
        if not api_key or api_key == "YourApiKeyHere":
            logger.warning("BYBIT_API_KEY nie jest ustawiony poprawnie w pliku .env")
            return False

        if not api_secret or api_secret == "YourApiSecretHere":
            logger.warning("BYBIT_API_SECRET nie jest ustawiony poprawnie w pliku .env")
            return False

        logger.info(f"Klucze API skonfigurowane. Używam {'testnet' if use_testnet else 'produkcyjnego API'}")

        # Sprawdź połączenie z API
        try:
            import requests
            from datetime import datetime
            import hmac
            import hashlib

            endpoint = "https://api-testnet.bybit.com" if use_testnet else "https://api.bybit.com"
            time_endpoint = f"{endpoint}/v5/market/time"

            response = requests.get(time_endpoint, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0 and "result" in data:
                    server_time = data["result"]["timeNano"] // 1000000
                    logger.info(f"Połączenie z API Bybit działa. Czas serwera: {datetime.fromtimestamp(server_time/1000)}")
                    return True
            else:
                logger.warning(f"Błąd podczas łączenia z API Bybit: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Błąd podczas testowania API Bybit: {e}")
            return False

    except Exception as e:
        logger.error(f"Błąd podczas weryfikacji kluczy API: {e}")
        return False

def main():
    """Główna funkcja inicjalizująca projekt."""
    logger.info("Rozpoczynam inicjalizację projektu...")

    # Utworzenie katalogów
    create_directories()

    # Instalacja zależności
    install_dependencies()

    # Konfiguracja pliku .env
    setup_env_file()

    # Weryfikacja kluczy API
    api_connected = verify_api_keys()

    if api_connected:
        logger.info("✅ Projekt został zainicjalizowany pomyślnie. Połączenie z API działa.")
    else:
        logger.warning("⚠️ Projekt został zainicjalizowany, ale połączenie z API nie działa.")
        logger.warning("Sprawdź poprawność kluczy API w pliku .env oraz dostęp do internetu.")

    logger.info("Inicjalizacja zakończona.")

if __name__ == "__main__":
    main()