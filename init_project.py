
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

def check_package(package_name):
    """Sprawdza, czy pakiet jest zainstalowany."""
    spec = importlib.util.find_spec(package_name.replace('-', '_'))
    return spec is not None

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
