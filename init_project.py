
#!/usr/bin/env python
"""
Skrypt inicjalizacyjny dla projektu Replit Trading Bot.
Ustawia środowisko i sprawdza, czy wszystkie zależności są poprawnie zainstalowane.
"""
import os
import sys
import importlib.util
import subprocess
import time
import json

def check_package(package_name):
    """Sprawdza, czy pakiet jest zainstalowany."""
    spec = importlib.util.find_spec(package_name.replace('-', '_'))
    return spec is not None

def check_env_vars():
    """Sprawdza zmienne środowiskowe i wyświetla ich status."""
    print("\n=== DIAGNOSTYKA ZMIENNYCH ŚRODOWISKOWYCH ===")
    
    # Sprawdź czy .env istnieje
    env_exists = os.path.exists('.env')
    print(f".env file exists: {env_exists}")
    
    # Spróbuj załadować dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("python-dotenv załadowany poprawnie")
    except ImportError:
        print("BŁĄD: Nie można zaimportować dotenv. Instaluję...")
        subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"])
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("python-dotenv zainstalowany i załadowany")
        except ImportError:
            print("BŁĄD: Instalacja dotenv nie powiodła się")
    
    # Kluczowe zmienne środowiskowe do sprawdzenia
    key_vars = [
        "BYBIT_API_KEY", 
        "BYBIT_API_SECRET", 
        "BYBIT_USE_TESTNET",
        "API_MIN_INTERVAL",
        "API_MAX_CALLS_PER_MINUTE",
        "FLASK_SECRET_KEY"
    ]
    
    # Sprawdź zmienne
    for var in key_vars:
        value = os.environ.get(var)
        if value:
            # Zamaskuj wrażliwe dane
            if "API_KEY" in var or "SECRET" in var:
                masked = value[:4] + "*" * (len(value) - 4) if len(value) > 4 else "****"
                print(f"{var}: {masked}")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: BRAK")
    
    print("=== KONIEC DIAGNOSTYKI ===\n")

def create_directories():
    """Tworzy wymagane katalogi."""
    directories = [
        'logs',
        'data/cache',
        'saved_models',
        'reports'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Utworzono katalog: {directory}")

def create_env_file():
    """Tworzy plik .env, jeśli nie istnieje."""
    if not os.path.exists('.env'):
        with open('.env.example', 'r') as example_file:
            example_content = example_file.read()
        
        with open('.env', 'w') as env_file:
            env_file.write(example_content)
        print("✓ Utworzono plik .env z ustawieniami przykładowymi")
    else:
        print("✓ Plik .env już istnieje")

def main():
    """Główna funkcja inicjalizacyjna."""
    print("=== Inicjalizacja projektu Trading Bot ===")
    
    # Sprawdzenie wymaganych pakietów
    required_packages = [
        'python-dotenv', 'flask', 'pybit', 'requests',
        'numpy', 'pandas', 'matplotlib', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Brakujące pakiety: {', '.join(missing_packages)}")
        print("Instalowanie brakujących pakietów...")
        
        # W Replit pakiety powinny być zainstalowane przez replit.nix
        print("ℹ️ W środowisku Replit pakiety powinny być zdefiniowane w replit.nix")
        print("ℹ️ Upewnij się, że plik replit.nix zawiera wszystkie wymagane zależności")
    else:
        print("✓ Wszystkie wymagane pakiety są zainstalowane")
    
    # Tworzenie katalogów
    create_directories()
    
    # Tworzenie pliku .env
    create_env_file()
    
    print("\n=== Inicjalizacja zakończona ===")
    print("Możesz teraz uruchomić aplikację za pomocą:")
    print("1. Przycisku 'Run' w Replit")
    print("2. Workflow 'Run Server' dla aplikacji webowej")
    print("3. Workflow 'Run' dla głównego skryptu")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
init_project.py
-------------
Skrypt do inicjalizacji projektu, tworzenia wymaganych katalogów i naprawy importów.
"""

import os
import sys
import logging
import subprocess
from typing import List

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/init_project.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_required_directories():
    """Tworzy wymagane katalogi projektu."""
    directories = [
        "logs",
        "data/cache",
        "reports",
        "static/css",
        "static/js",
        "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Utworzono katalog: {directory}")

def check_required_packages():
    """Sprawdza i instaluje wymagane pakiety."""
    required_packages = [
        "flask",
        "python-dotenv",
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "pytest",
        "flake8",
        "black",
        "psutil"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Brakujące pakiety: {', '.join(missing_packages)}")
        print(f"📦 Instalacja brakujących pakietów: {', '.join(missing_packages)}")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info(f"Zainstalowano pakiety: {', '.join(missing_packages)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Błąd instalacji pakietów: {e}")
    else:
        logger.info("Wszystkie wymagane pakiety są zainstalowane")

def fix_imports():
    """Uruchamia skrypt naprawy importów."""
    try:
        if os.path.exists("fix_imports.py"):
            logger.info("Uruchamianie skryptu naprawy importów")
            subprocess.check_call([sys.executable, "fix_imports.py"])
        else:
            logger.warning("Brak skryptu fix_imports.py")
    except subprocess.CalledProcessError as e:
        logger.error(f"Błąd podczas naprawy importów: {e}")

def create_example_env_file():
    """Tworzy przykładowy plik .env, jeśli nie istnieje."""
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Konfiguracja Flask\n")
            f.write("FLASK_APP=main.py\n")
            f.write("FLASK_ENV=development\n")
            f.write("FLASK_DEBUG=1\n")
            f.write("PORT=5000\n\n")
            f.write("# API ByBit\n")
            f.write("BYBIT_API_KEY=YourApiKeyHere\n")
            f.write("BYBIT_API_SECRET=YourApiSecretHere\n")
            f.write("BYBIT_USE_TESTNET=true\n")
        
        logger.info("Utworzono przykładowy plik .env")

def main():
    """Główna funkcja inicjalizacji projektu."""
    print("\n🚀 Inicjalizacja projektu\n")
    
    # Utworzenie katalogów
    create_required_directories()
    print("✅ Utworzono wymagane katalogi")
    
    # Sprawdzenie pakietów
    check_required_packages()
    print("✅ Sprawdzono wymagane pakiety")
    
    # Naprawa importów
    fix_imports()
    print("✅ Naprawiono importy")
    
    # Utworzenie przykładowego pliku .env
    create_example_env_file()
    print("✅ Utworzono przykładowy plik .env")
    
    print("\n🎉 Inicjalizacja projektu zakończona pomyślnie!\n")
    print("👉 Teraz możesz uruchomić aplikację używając \"Run Server\"")

if __name__ == "__main__":
    main()
