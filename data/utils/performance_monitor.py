
"""
performance_monitor.py
--------------------
Moduł do monitorowania wydajności systemu i strategii tradingowych.
"""

import logging
import os
import time
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

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


import time
import logging
import os
import psutil
from typing import Dict, Any, List, Callable
from functools import wraps
import traceback

# Inicjalizacja logowania
logger = logging.getLogger(__name__)

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
            return {} systemu.

        Returns:
            Dict[str, Any]: Statystyki systemu.
        """
        try:
            # Zbieranie statystyk systemu
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            disk = psutil.disk_usage('/')
            disk_usage = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
            
            self.system_stats = {
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "disk": disk_usage,
                "uptime": time.time() - self.start_time,
                "timestamp": time.time()
            }
            
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                logger.info(f"Statystyki systemu: CPU: {cpu_percent}%, RAM: {memory.percent}%, Disk: {disk.percent}%")
                self.last_log_time = current_time
            
            return self.system_stats
        except Exception as e:
            logger.error(f"Błąd podczas monitorowania systemu: {e}")
            return {"error": str(e)}

    def monitor_strategy(self, strategy_name: str, performance_data: Dict[str, Any]) -> None:
        """
        Monitoruje wydajność strategii tradingowej.

        Parameters:
            strategy_name (str): Nazwa strategii.
            performance_data (Dict[str, Any]): Dane wydajności.
        """
        try:
            # Aktualizacja danych wydajności
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = []
            
            # Dodanie aktualnego czasu
            performance_data["timestamp"] = time.time()
            
            self.strategy_performance[strategy_name].append(performance_data)
            
            # Ograniczenie liczby przechowywanych rekordów do 1000
            if len(self.strategy_performance[strategy_name]) > 1000:
                self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-1000:]
            
            logger.info(f"Zaaktualizowano wydajność strategii {strategy_name}: {performance_data}")
        except Exception as e:
            logger.error(f"Błąd podczas monitorowania strategii {strategy_name}: {e}")

    def get_strategy_performance(self, strategy_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Pobiera dane wydajności strategii.

        Parameters:
            strategy_name (str): Nazwa strategii.
            limit (int): Maksymalna liczba rekordów.

        Returns:
            List[Dict[str, Any]]: Lista danych wydajności.
        """
        if strategy_name not in self.strategy_performance:
            return []
        
        return self.strategy_performance[strategy_name][-limit:]

    def get_system_report(self) -> Dict[str, Any]:
        """
        Generuje raport wydajności systemu.

        Returns:
            Dict[str, Any]: Raport wydajności.
        """
        # Aktualizacja statystyk
        self.monitor_system()
        
        # Przygotowanie raportu
        report = {
            "system": self.system_stats,
            "strategies": {},
            "timestamp": time.time(),
            "datetime": datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Dodanie danych wydajności strategii
        for strategy_name, performance_data in self.strategy_performance.items():
            if performance_data:
                report["strategies"][strategy_name] = performance_data[-1]
        
        return report

    def log_execution_time(self, function_name: str, execution_time: float) -> None:
        """
        Loguje czas wydef log_execution_time(self, function_name: str, execution_time: float):
        """
        Zapisuje czas wykonania funkcji.

        Parameters:
            function_name (str): Nazwa funkcji.
            execution_time (float): Czas wykonania w sekundach.
        """
        logger.info(f"Czas wykonania {function_name}: {execution_time:.6f}s")
        
        # Dodanie do statystyk
        if "execution_times" not in self.system_stats:
            self.system_stats["execution_times"] = {}
        
        if function_name not in self.system_stats["execution_times"]:
            self.system_stats["execution_times"][function_name] = []
        
        self.system_stats["execution_times"][function_name].append(execution_time)
        
        # Ograniczenie liczby przechowywanych czasów do 1000
        if len(self.system_stats["execution_times"][function_name]) > 1000:
            self.system_stats["execution_times"][function_name] = self.system_stats["execution_times"][function_name][-1000:]
            
    def track_strategy_performance(self, strategy_name: str, metrics: Dict[str, Any]):
        """
        Zapisuje metryki wydajności strategii.
        
        Parameters:
            strategy_name (str): Nazwa strategii.
            metrics (Dict[str, Any]): Metryki wydajności.
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
            
        metrics['timestamp'] = time.time()
        self.strategy_performance[strategy_name].append(metrics)
        
        # Skróć historię do ostatnich 1000 wpisów
        if len(self.strategy_performance[strategy_name]) > 1000:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-1000:]
            
        logger.info(f"Wydajność strategii {strategy_name}: {metrics}")
        
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Zwraca statystyki systemowe.
        
        Returns:
            Dict[str, Any]: Statystyki systemowe.
        """
        return self.system_stats
        
    def get_strategy_performance(self, strategy_name: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Zwraca metryki wydajności strategii.
        
        Parameters:
            strategy_name (str, optional): Nazwa strategii. Jeśli None, zwraca wszystkie strategie.
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Metryki wydajności strategii.
        """
        if strategy_name:
            return {strategy_name: self.strategy_performance.get(strategy_name, [])}
        return self.strategy_performance


# Singleton instancja dla łatwego dostępu z różnych modułów
performance_monitor = PerformanceMonitor()


# Dekorator do mierzenia czasu wykonania funkcji
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            performance_monitor.log_execution_time(func.__name__, execution_time)
    return wrappero pomiaru czasu wykonania funkcji
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
