"""
Skrypt do instalacji pakietów w środowisku Replit z obsługą ograniczeń uprawnień
"""

import os
import platform
import subprocess
import sys
import logging
import time
import json

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("install_replit.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def is_installed(package_name):
    """Sprawdza, czy pakiet jest już zainstalowany"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """Instaluje pojedynczy pakiet z obsługą błędów"""
    try:
        # Próba instalacji z flagą --user
        cmd = [sys.executable, "-m", "pip", "install", "--user", package]
        logging.info(f"Instalacja pakietu {package}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Pakiet {package} zainstalowany pomyślnie")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Błąd podczas instalacji pakietu {package}: {e.stderr}")
        logging.info(f"Próba alternatywnego podejścia dla {package}...")
        try:
            # Próba użycia alternatywnego podejścia bez flagi --user
            cmd = [sys.executable, "-m", "pip", "install", package]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logging.info(f"Pakiet {package} zainstalowany alternatywnie")
            return True
        except subprocess.CalledProcessError as e2:
            logging.error(f"Druga próba instalacji pakietu {package} nieudana: {e2.stderr}")
            return False

def install_requirements(requirements_file):
    """Odczytuje i instaluje pakiety z pliku requirements.txt"""
    if not os.path.exists(requirements_file):
        logging.error(f"Plik {requirements_file} nie istnieje")
        return False

    # Odczytanie pakietów z requirements.txt
    with open(requirements_file, 'r') as f:
        requirements = f.readlines()

    # Filtracja komentarzy i pustych linii
    requirements = [r.strip() for r in requirements if r.strip() and not r.strip().startswith('#')]

    # Podstawowe pakiety potrzebne do działania aplikacji
    core_packages = [
        "flask", "numpy", "pandas", "python-dotenv", "requests", 
        "pyyaml", "psutil", "colorama", "tqdm", "typing-extensions"
    ]

    # Pakiety zaawansowane (opcjonalne - nie blokują działania aplikacji)
    optional_packages = [
        "ccxt", "ta", "mplfinance", "pybit", "aiohttp", "websockets", 
        "pyOpenSSL", "pycryptodome", "sentry-sdk", "prometheus-client"
    ]

    # Najpierw instaluj core packages
    logging.info("Instalacja podstawowych pakietów...")
    for package in core_packages:
        if not is_installed(package.split('>=')[0].split('==')[0]):
            install_package(package)

    # Następnie instaluj pozostałe pakiety
    logging.info("Instalacja pozostałych pakietów...")
    remaining_packages = [p for p in requirements if not any(p.startswith(core) for core in core_packages + optional_packages)]
    for package in remaining_packages:
        if not is_installed(package.split('>=')[0].split('==')[0]):
            install_package(package)

    # Na koniec opcjonalne pakiety
    logging.info("Instalacja opcjonalnych pakietów...")
    for package in optional_packages:
        if not is_installed(package.split('>=')[0].split('==')[0]):
            install_package(package)

    return True

def create_env_if_missing():
    """Tworzy plik .env jeśli nie istnieje"""
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        logging.info("Tworzenie pliku .env z szablonu .env.example...")
        with open('.env.example', 'r') as src:
            with open('.env', 'w') as dst:
                dst.write(src.read())
        logging.info("Plik .env utworzony pomyślnie")

def create_replit_config():
    """Tworzy lub aktualizuje plik konfiguracyjny .replit"""
    if not os.path.exists('.replit'):
        logging.info("Tworzenie pliku konfiguracyjnego .replit...")
        config = """
run = "python main.py"
entrypoint = "main.py"

[nix]
channel = "stable-23_11"

[env]
PYTHONPATH = "${PYTHONPATH}:${workspaceFolder}"
PATH = "${PATH}:${HOME}/.local/bin"

[languages]
python = "python3"

[languages.python]
pattern = "**/*.py"

[languages.python.languageServer]
start = "pylsp"
"""
        with open('.replit', 'w') as f:
            f.write(config.strip())
        logging.info("Plik .replit utworzony pomyślnie")

def create_env_files():
    """Tworzy wszystkie potrzebne pliki konfiguracyjne"""
    create_env_if_missing()
    create_replit_config()

def fix_imports_in_file(filepath):
    """Naprawia problematyczne importy w pliku"""
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Przykładowe naprawy importów
    replacements = [
        ('import tensorflow as tf', '# import tensorflow as tf  # Zakomentowano - opcjonalny pakiet'),
        ('import torch', '# import torch  # Zakomentowano - opcjonalny pakiet'),
        ('import optuna', '# import optuna  # Zakomentowano - opcjonalny pakiet')
    ]

    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        logging.info(f"Naprawiono importy w pliku {filepath}")
        return True

    return False

def fix_ai_model_imports():
    """Naprawia problematyczne importy w modelach AI"""
    ai_models_dir = 'ai_models'
    if not os.path.exists(ai_models_dir):
        return

    for filename in os.listdir(ai_models_dir):
        if filename.endswith('.py'):
            filepath = os.path.join(ai_models_dir, filename)
            fix_imports_in_file(filepath)

def main():
    """Główna funkcja instalacyjna"""
    start_time = time.time()
    logging.info("Rozpoczęcie procesu instalacji dla Replit...")

    # Wyświetl informacje o środowisku
    logging.info(f"Python: {sys.version}")
    logging.info(f"System: {platform.system()} {platform.release()}")

    # Stwórz potrzebne katalogi
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('python_libs', exist_ok=True)

    # Utwórz pliki konfiguracyjne
    create_env_files()

    # Napraw importy w modelach AI
    logging.info("Naprawianie problematycznych importów w modelach AI...")
    fix_ai_model_imports()

    # Instalacja pakietów
    success = install_requirements('requirements.txt')

    elapsed_time = time.time() - start_time
    if success:
        logging.info(f"Instalacja zakończona pomyślnie w {elapsed_time:.2f} sekund")
    else:
        logging.warning(f"Instalacja zakończona z problemami w {elapsed_time:.2f} sekund")

    logging.info("Możesz teraz uruchomić aplikację poleceniem: python main.py")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
install_replit.py - Skrypt instalujący zależności dla Replit
"""

import os
import sys
import subprocess
import logging
import time
import re
import json
from typing import List, Dict, Any, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def install_package(package_name: str) -> bool:
    """
    Instaluje pakiet Python.
    
    Args:
        package_name: Nazwa pakietu
        
    Returns:
        bool: Czy instalacja się powiodła
    """
    logger.info(f"Instalacja pakietu {package_name}...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Pakiet {package_name} zainstalowany pomyślnie")
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Błąd podczas instalacji pakietu {package_name}")
        
        # Próba alternatywnego podejścia
        logger.info(f"Próba alternatywnego podejścia dla {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package_name])
            logger.info(f"Pakiet {package_name} zainstalowany pomyślnie (podejście alternatywne)")
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Druga próba instalacji pakietu {package_name} nieudana")
            return False

def fix_ai_model_imports():
    """Naprawia problematyczne importy w modelach AI."""
    logger.info("Naprawianie problematycznych importów w modelach AI...")
    
    # Ścieżka do katalogu modeli AI
    ai_models_dir = "ai_models"
    
    # Sprawdź czy katalog istnieje
    if not os.path.exists(ai_models_dir):
        os.makedirs(ai_models_dir, exist_ok=True)
        logger.info(f"Utworzono katalog {ai_models_dir}")
    
    # Znajdź pliki .py w katalogu modeli
    for filename in os.listdir(ai_models_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            file_path = os.path.join(ai_models_dir, filename)
            
            try:
                # Odczytaj zawartość pliku
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Napraw importy
                fixed_content = content
                
                # Zastąp problematyczne importy
                patterns = [
                    (r"from tensorflow import", "# from tensorflow import"),
                    (r"import tensorflow as", "# import tensorflow as"),
                    (r"from torch import", "# from torch import"),
                    (r"import torch", "# import torch"),
                    (r"from sklearn\.ensemble import IsolationForest", "# from sklearn.ensemble import IsolationForest")
                ]
                
                for pattern, replacement in patterns:
                    fixed_content = re.sub(pattern, replacement, fixed_content)
                
                # Zapisz poprawioną zawartość
                if fixed_content != content:
                    with open(file_path, "w") as f:
                        f.write(fixed_content)
                    logger.info(f"Naprawiono importy w pliku {file_path}")
            except Exception as e:
                logger.error(f"Błąd podczas naprawiania importów w pliku {file_path}: {e}")

def setup_python_libs():
    """Konfiguruje katalog python_libs."""
    python_libs_dir = "python_libs"
    
    # Sprawdź czy katalog istnieje
    if not os.path.exists(python_libs_dir):
        os.makedirs(python_libs_dir, exist_ok=True)
        logger.info(f"Utworzono katalog {python_libs_dir}")
    
    # Utwórz plik __init__.py, jeśli nie istnieje
    init_file = os.path.join(python_libs_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Inicjalizacja pakietu python_libs\n")
        logger.info(f"Utworzono plik {init_file}")

def install_requirements():
    """Instaluje wymagane pakiety z pliku requirements.txt."""
    try:
        requirements_file = "requirements.txt"
        
        if os.path.exists(requirements_file):
            logger.info("Instalacja pakietów z requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            logger.info("Zainstalowano pakiety z requirements.txt")
        else:
            logger.warning(f"Plik {requirements_file} nie istnieje")
    except subprocess.CalledProcessError:
        logger.error("Błąd podczas instalacji pakietów z requirements.txt")

def install_core_packages():
    """Instaluje podstawowe pakiety."""
    logger.info("Instalacja podstawowych pakietów...")
    
    core_packages = [
        "pytz", "typing-extensions", "pyyaml", "requests", "python-dotenv", "psutil", "colorama", "tqdm"
    ]
    
    for package in core_packages:
        install_package(package)

def install_optional_packages():
    """Instaluje opcjonalne pakiety."""
    logger.info("Instalacja opcjonalnych pakietów...")
    
    optional_packages = [
        "ccxt", "ta", "mplfinance", "pybit", "aiohttp", "websockets", "pyOpenSSL", "pycryptodome", "sentry-sdk", "prometheus-client"
    ]
    
    for package in optional_packages:
        install_package(package)

def install_packages_from_requirements():
    """Instaluje pakiety z pliku requirements.txt."""
    logger.info("Instalacja pozostałych pakietów...")
    
    requirements_file = "requirements.txt"
    
    if os.path.exists(requirements_file):
        with open(requirements_file, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            # Pomiń komentarze i puste linie
            if line.strip() and not line.strip().startswith("#"):
                # Usuń komentarze na końcu linii
                package = line.split("#")[0].strip()
                
                # Pomiń już zainstalowane podstawowe pakiety
                if not any(p in package.lower() for p in ["python-dotenv", "pyyaml", "requests", "psutil", "colorama", "tqdm", "typing-extensions"]):
                    install_package(package)

def main():
    """Główna funkcja."""
    start_time = time.time()
    
    logger.info("Rozpoczęcie procesu instalacji dla Replit...")
    
    # Informacje o środowisku
    logger.info(f"Python: {sys.version}")
    logger.info(f"System: {os.uname().sysname} {os.uname().release}")
    
    # Napraw problematyczne importy w modelach AI
    fix_ai_model_imports()
    
    # Konfiguracja katalogu python_libs
    setup_python_libs()
    
    # Instalacja podstawowych pakietów
    install_core_packages()
    
    # Instalacja pakietów z requirements.txt
    install_packages_from_requirements()
    
    # Instalacja opcjonalnych pakietów
    install_optional_packages()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Instalacja zakończona pomyślnie w {elapsed_time:.2f} sekund")
    logger.info("Możesz teraz uruchomić aplikację poleceniem: python main.py")

if __name__ == "__main__":
    main()
