"""
settings.py
-----------
Skrypt definiujący ustawienia aplikacji jako zmienne i słowniki.
Wprowadza mechanizm dynamicznego przełączania środowisk (dev, test, production)
oraz integruje się z modułem ConfigLoader, umożliwiając ładowanie dodatkowych konfiguracji
z plików JSON/YAML. Kod został zaprojektowany z myślą o skalowalności i szybkim uruchamianiu dużych systemów.
"""

import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from .config_loader import \
    ConfigLoader  # Import względny, dopasuj do swojej struktury projektu

# -----------------------------------
# 1. Wczytanie zmiennych z pliku .env
# -----------------------------------
load_dotenv()

# -----------------------------------
# 2. Konfiguracja wstępna logowania
# -----------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------------
# 3. Określenie środowiska
# -----------------------------------
APP_ENV: str = os.getenv("APP_ENV", "development").lower()
logging.info("Ładowanie ustawień dla środowiska: %s", APP_ENV)

# -----------------------------------
# 4. Domyślne ustawienia
# -----------------------------------
default_settings: Dict[str, Any] = {
    "development": {
        "DEBUG": True,
        "DATABASE": {
            "host": "localhost",
            "port": 5432,
            "user": "devuser",
            "password": "devpassword",
            "name": "dev_db",
        },
        # Binance klucze (np. do testnetu):
        "API": {
            # Jeśli chcesz wczytywać z .env:
            "key": os.getenv("BINANCE_API_KEY", "binance_public_key"),
            "secret": os.getenv("BINANCE_API_SECRET", "binance_public_secret"),
            "base_url": "https://testnet.binance.vision",
        },
        # Bybit klucze – pobierane z .env
        "BYBIT": {
            "api_key": os.getenv("BYBIT_API_KEY", "bybit_dummy_key"),
            "api_secret": os.getenv("BYBIT_API_SECRET", "bybit_dummy_secret"),
            "use_testnet": os.getenv("BYBIT_USE_TESTNET", "true").lower() == "true",
        },
        "TRADING": {"commission": 0.001, "spread": 0.0005, "slippage": 0.0005},
        "SECURITY": {"access_restrictions": False, "password_encryption": False},
        "LOGGING": {"level": "DEBUG", "file": "dev_app.log"},
        "PATHS": {"logs_dir": "./logs", "data_dir": "./data"},
    },
    "test": {
        "DEBUG": True,
        "DATABASE": {
            "host": "localhost",
            "port": 5432,
            "user": "testuser",
            "password": "testpassword",
            "name": "test_db",
        },
        "API": {
            "key": "test_key",
            "secret": "test_secret",
            "base_url": "https://api.binance.com",
        },
        "BYBIT": {
            "api_key": "test_bybit_key",
            "api_secret": "test_bybit_secret",
            "use_testnet": True,
        },
        "TRADING": {"commission": 0.001, "spread": 0.0005, "slippage": 0.0005},
        "SECURITY": {"access_restrictions": True, "password_encryption": True},
        "LOGGING": {"level": "INFO", "file": "test_app.log"},
        "PATHS": {"logs_dir": "./logs", "data_dir": "./data"},
    },
    "production": {
        "DEBUG": False,
        "DATABASE": {
            "host": "prod-db-host",
            "port": 5432,
            "user": "produser",
            "password": "prodpassword",
            "name": "prod_db",
        },
        "API": {
            "key": "prod_key",
            "secret": "prod_secret",
            "base_url": "https://api.binance.com",
        },
        "BYBIT": {
            "api_key": "prod_bybit_key",
            "api_secret": "prod_bybit_secret",
            "use_testnet": False,
        },
        "TRADING": {"commission": 0.001, "spread": 0.0005, "slippage": 0.0005},
        "SECURITY": {"access_restrictions": True, "password_encryption": True},
        "LOGGING": {"level": "WARNING", "file": "prod_app.log"},
        "PATHS": {"logs_dir": "./logs", "data_dir": "./data"},
    },
}

# -----------------------------------
# 5. Wybór ustawień dla aktualnego środowiska
# -----------------------------------
env_settings: Dict[str, Any] = default_settings.get(
    APP_ENV, default_settings["development"]
)

# -----------------------------------
# 6. Ładowanie dodatkowych ustawień z plików JSON/YAML przy pomocy ConfigLoader
# -----------------------------------
config_files = []
if os.path.exists("config/settings.yaml"):
    config_files.append("config/settings.yaml")
if os.path.exists("config/settings.json"):
    config_files.append("config/settings.json")

# Uwaga: Import i użycie ConfigLoader
loader = ConfigLoader(config_files=config_files, env_prefix="APP", cache_enabled=True)
try:
    file_settings: Dict[str, Any] = loader.load()
except Exception as e:
    logging.error("Błąd ładowania dodatkowych ustawień: %s", e)
    file_settings = {}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rekurencyjnie scala dwa słowniki, gdzie wartości z 'override' nadpisują te z 'base'.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


# -----------------------------------
# 7. Scalanie ustawień (domyślne + pliki + zmienne środowiskowe)
# -----------------------------------
settings: Dict[str, Any] = merge_dicts(env_settings, file_settings)

# -----------------------------------
# 8. Ostateczna konfiguracja
# -----------------------------------
CONFIG: Dict[str, Any] = settings

logging.info("Ostateczna konfiguracja: %s", CONFIG)
logging.info("BINANCE klucz: %s", CONFIG["API"]["key"])
logging.info("BYBIT klucz: %s", CONFIG["BYBIT"]["api_key"])
