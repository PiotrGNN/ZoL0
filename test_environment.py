
#!/usr/bin/env python3

import sys
import importlib
import subprocess
import os

def check_module(module_name):
    """Sprawdza, czy moduł może być zaimportowany."""
    try:
        importlib.import_module(module_name)
        print(f"✅ Moduł {module_name} jest dostępny")
        return True
    except ImportError as e:
        print(f"❌ Błąd importu modułu {module_name}: {e}")
        return False

def check_environment():
    """Sprawdza konfigurację środowiska."""
    print("\n=== SPRAWDZANIE ŚRODOWISKA PYTHON ===")
    
    # Wersja Pythona
    print(f"Python: {sys.version}")
    
    # Ścieżka Pythona
    print(f"Ścieżka Pythona: {sys.executable}")
    
    # Sprawdź PYTHONPATH
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Nie ustawiony')}")
    
    # Sprawdź kluczowe moduły
    modules_to_check = [
        "flask", "pandas", "numpy", "sklearn", "matplotlib", 
        "xgboost", "requests", "dotenv", "flask_cors",
        "flask_socketio", "json", "pytz"
    ]
    
    print("\n=== SPRAWDZANIE MODUŁÓW ===")
    missing_modules = []
    for module in modules_to_check:
        if not check_module(module):
            missing_modules.append(module)
    
    # Podsumowanie
    print("\n=== PODSUMOWANIE ===")
    if missing_modules:
        print(f"❌ Brakujące moduły: {', '.join(missing_modules)}")
        print("\nWykonaj następujące polecenia, aby zainstalować brakujące pakiety:")
        for module in missing_modules:
            print(f"pip3 install {module}")
    else:
        print("✅ Wszystkie wymagane moduły są dostępne")
    
    print("\nAby przetestować pełne środowisko, uruchom:")
    print("python3 main.py")

if __name__ == "__main__":
    check_environment()
