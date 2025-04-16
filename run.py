
#!/usr/bin/env python3
"""
run.py - Skrypt do lokalnego uruchomienia systemu tradingowego
"""

import os
import sys
import logging
import time
import subprocess
from dotenv import load_dotenv

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run.log")
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Przygotowanie środowiska do uruchomienia."""
    logger.info("Rozpoczynam przygotowanie środowiska...")
    
    # Upewnij się, że potrzebne katalogi istnieją
    for directory in ["logs", "data/cache", "reports", "static/img", "saved_models", 
                      "models", "python_libs/__pycache__", "ai_models/__pycache__"]:
        os.makedirs(directory, exist_ok=True)
    
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

def fix_imports():
    """Naprawia znane problemy z importami w projekcie."""
    logger.info("Sprawdzam i naprawiam importy...")
    
    # Sprawdzenie importu datetime w model_recognition.py
    model_recognition_path = 'ai_models/model_recognition.py'
    if os.path.exists(model_recognition_path):
        with open(model_recognition_path, 'r') as f:
            content = f.read()
        
        if 'from datetime import datetime' not in content:
            logger.info("Dodaję import datetime do model_recognition.py")
            lines = content.split('\n')
            import_index = next((i for i, line in enumerate(lines) if line.startswith('import') or line.startswith('from')), 0)
            lines.insert(import_index, 'from datetime import datetime')
            
            with open(model_recognition_path, 'w') as f:
                f.write('\n'.join(lines))
    
    return True

def run_application():
    """Uruchomienie głównej aplikacji."""
    try:
        # Utwórz flagę, że aplikacja jest uruchomiona, aby uniknąć podwójnego uruchomienia
        with open('running.lock', 'w') as f:
            f.write(str(time.time()))
        
        # Utwórz też flagę dla testowania modeli
        with open('models_tested.lock', 'w') as f:
            f.write(str(time.time()))
        
        logger.info("Importuję główny moduł aplikacji...")
        
        # Import main powinien być wykonany tylko raz
        import main
        
        # Usuń plik blokady po zakończeniu
        if os.path.exists('running.lock'):
            os.remove('running.lock')
        
        logger.info("Aplikacja uruchomiona pomyślnie.")
        return True
    except Exception as e:
        logger.error(f"Błąd podczas uruchamiania aplikacji: {e}", exc_info=True)
        
        # Usuń plik blokady w przypadku błędu
        if os.path.exists('running.lock'):
            os.remove('running.lock')
        
        return False

if __name__ == "__main__":
    logger.info("Uruchamianie lokalnego systemu tradingowego...")
    
    # Sprawdź czy aplikacja już jest uruchomiona
    if os.path.exists('running.lock'):
        with open('running.lock', 'r') as f:
            start_time = float(f.read().strip())
        
        # Jeśli plik jest starszy niż 2 godziny, prawdopodobnie pozostał po poprzedniej nieudanej sesji
        if time.time() - start_time < 7200:  # 2 godziny w sekundach
            logger.error("Aplikacja jest już uruchomiona. Jeśli jesteś pewien, że nie działa, usuń plik 'running.lock'.")
            sys.exit(1)
        else:
            logger.warning("Znaleziono plik blokady, ale jest stary. Usuwam i kontynuuję.")
            os.remove('running.lock')
    
    # Przygotowanie środowiska
    if not setup_environment():
        logger.error("Błąd podczas przygotowywania środowiska.")
        sys.exit(1)
    
    # Naprawianie znanych problemów
    if not fix_imports():
        logger.error("Błąd podczas naprawiania importów.")
        sys.exit(1)
    
    # Uruchomienie aplikacji
    logger.info("Uruchamianie aplikacji...")
    if not run_application():
        logger.error("Błąd podczas uruchamiania aplikacji.")
        sys.exit(1)
    
    logger.info("System zakończył działanie.")
