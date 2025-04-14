
#!/usr/bin/env python3
"""
install_replit.py - Skrypt do bezpiecznej instalacji pakietów w środowisku Replit.

Ten skrypt instaluje tylko niezbędne pakiety potrzebne do działania aplikacji
w środowisku Replit, pomijając ciężkie zależności jak TensorFlow i PyTorch,
które mogą powodować problemy z uprawnieniami.
"""

import os
import sys
import subprocess
import logging

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def install_packages():
    """Instaluje niezbędne pakiety z użyciem flagi --user."""
    
    logging.info("Rozpoczynam instalację pakietów dla środowiska Replit...")
    
    # Użyj flagi --user, aby zainstalować pakiety w katalogu użytkownika
    cmd = [sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"]
    
    try:
        subprocess.run(cmd, check=True)
        logging.info("Instalacja pakietów zakończona pomyślnie.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Błąd podczas instalacji pakietów: {e}")
        sys.exit(1)

def setup_local_directories():
    """Tworzy niezbędne katalogi dla aplikacji."""
    directories = [
        "logs",
        "data/cache",
        "reports",
        "static/img",
        "saved_models",
        "python_libs/__pycache__",
        "ai_models/__pycache__",
        "data/utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Utworzono katalog: {directory}")
        
    # Upewnij się, że katalogi zawierają pliki __init__.py
    init_directories = [
        "python_libs",
        "ai_models",
        "data",
        "data/cache",
        "data/utils"
    ]
    
    for directory in init_directories:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Auto-generated __init__.py file\n")
            logging.info(f"Utworzono plik inicjalizacyjny: {init_file}")

    # Utwórz prosty performance_monitor jeśli nie istnieje
    performance_monitor_path = os.path.join("data", "utils", "performance_monitor.py")
    if not os.path.exists(performance_monitor_path):
        with open(performance_monitor_path, 'w') as f:
            f.write('''"""
performance_monitor.py
--------------------
Moduł do monitorowania wydajności systemu i strategii tradingowych.
"""

import logging
import os
import time
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional

# Konfiguracja logowania
logger = logging.getLogger("performance_monitor")
if not logger.handlers:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "performance_monitor.log"))
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class PerformanceMonitor:
    """
    Klasa do monitorowania wydajności systemu i strategii tradingowych.
    """

    def __init__(self, log_interval: int = 60):
        """
        Inicjalizuje monitor wydajności.

        Parameters:
            log_interval (int): Interwał logowania w sekundach.
        """
        self.log_interval = log_interval
        self.last_log_time = 0
        self.system_stats = {}
        self.strategy_performance = {}
        self.start_time = time.time()
        logger.info(f"Zainicjalizowano monitor wydajności z interwałem logowania: {log_interval}s")

    def monitor_system(self) -> Dict[str, Any]:
        """
        Monitoruje wydajność systemu.
        
        Returns:
            Dict[str, Any]: Statystyki systemowe
        """
        try:
            # Pobierz informacje o CPU, pamięci i dysku
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            stats = {
                'cpu': {
                    'percent': cpu_percent,
                    'cores': psutil.cpu_count()
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'uptime': time.time() - self.start_time
            }
            
            # Aktualizuj statystyki systemowe
            self.system_stats.update(stats)
            
            current_time = time.time()
            if current_time - self.last_log_time > self.log_interval:
                logger.info(f"Wydajność systemu: CPU: {cpu_percent}%, RAM: {memory.percent}%")
                self.last_log_time = current_time
                
            return stats
        except Exception as e:
            logger.error(f"Błąd podczas monitorowania systemu: {e}")
            return {}
    
    def log_execution_time(self, function_name: str, execution_time: float):
        """
        Zapisuje czas wykonania funkcji.

        Parameters:
            function_name (str): Nazwa funkcji.
            execution_time (float): Czas wykonania w sekundach.
        """
        logger.info(f"Czas wykonania {function_name}: {execution_time:.6f}s")
        
# Singleton instancja dla łatwego dostępu z różnych modułów
performance_monitor = PerformanceMonitor()

# Dekorator do pomiaru czasu wykonania funkcji
def measure_time(func):
    """
    Dekorator do pomiaru czasu wykonania funkcji.
    
    Parameters:
        func: Funkcja do zmierzenia.
        
    Returns:
        Funkcja wrapper.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        performance_monitor.log_execution_time(func.__name__, execution_time)
        return result
    return wrapper


def get_system_stats() -> Dict[str, Any]:
    """
    Funkcja pomocnicza do pobierania statystyk systemu.
    
    Returns:
        Dict[str, Any]: Statystyki systemu.
    """
    return performance_monitor.monitor_system()
''')
        logging.info(f"Utworzono prosty moduł performance_monitor.py")

def main():
    """Funkcja główna."""
    logging.info("Rozpoczynam konfigurację projektu dla środowiska Replit...")
    
    # Instalacja pakietów
    install_packages()
    
    # Tworzenie katalogów
    setup_local_directories()
    
    logging.info("Konfiguracja projektu zakończona.")
    logging.info("Możesz teraz uruchomić aplikację za pomocą komendy: python main.py")

if __name__ == "__main__":
    main()
