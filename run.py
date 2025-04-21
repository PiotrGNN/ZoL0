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
import socket

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

# Funkcja do znajdowania wolnego portu
def find_free_port(start_port=5000, max_attempts=20):
    """Znajdź wolny port TCP zaczynając od start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                logging.info(f"Znaleziono wolny port: {port}")
                return port
            except OSError:
                logging.info(f"Port {port} jest zajęty, próbuję następny...")
                continue
    # Jeśli nie znaleziono wolnego portu, zwróć None
    logging.error(f"Nie znaleziono wolnego portu w zakresie {start_port}-{start_port + max_attempts - 1}")
    return None

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
        
        # Znajdź wolny port zaczynając od domyślnego portu 5000
        default_port = int(os.environ.get("PORT", 5000))
        port = find_free_port(start_port=default_port)
        if port is None:
            logger.error("Nie znaleziono wolnego portu. Zakończono działanie.")
            return False
        
        # Ustawienie zmiennej środowiskowej PORT na znaleziony wolny port
        os.environ["PORT"] = str(port)
        
        host = "0.0.0.0"  # Używamy 0.0.0.0 dla dostępu zewnętrznego w środowisku Replit
        
        try:
            logger.info(f"Uruchamianie serwera Flask na {host}:{port}")
            # Uruchamiamy serwer Flask w osobnym wątku, aby nie blokować głównego programu
            import threading
            threading.Thread(target=lambda: main.app.run(host=host, port=port, debug=False)).start()
            logger.info(f"Serwer Flask uruchomiony na {host}:{port}")
            
            # Wyświetlenie adresu dostępu dla użytkownika
            print(f"\n============================================")
            print(f"Aplikacja dostępna pod adresem: http://localhost:{port}")
            print(f"============================================\n")
            
            # Utrzymuj główny wątek aktywny, aby program nie zakończył się od razu
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Otrzymano sygnał zakończenia pracy (Ctrl+C).")
        except Exception as flask_err:
            logger.error(f"Błąd podczas uruchamiania serwera Flask: {flask_err}", exc_info=True)
            
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
