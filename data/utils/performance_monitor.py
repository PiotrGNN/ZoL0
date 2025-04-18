"""
performance_monitor.py
--------------------
Moduł do monitorowania wydajności systemu i strategii tradingowych.
"""

import logging
import os
import time
import psutil
import platform
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps

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
        self.running = False
        self.monitor_thread = None
        self.metrics = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "timestamp": []
        }
        self.max_history = 1000  # Maksymalna liczba zapisanych pomiarów
        logger.info(f"Zainicjalizowano monitor wydajności z interwałem logowania: {log_interval}s")

    def start(self):
        """Rozpoczyna monitorowanie w osobnym wątku."""
        if self.running:
            logger.warning("Monitor już działa")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Monitoring wydajności uruchomiony")

    def stop(self):
        """Zatrzymuje monitorowanie."""
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring wydajności zatrzymany")

    def _monitor_loop(self):
        """Główna pętla monitoringu."""
        try:
            while self.running:
                self._collect_metrics()
                time.sleep(self.log_interval)
        except Exception as e:
            logger.error(f"Błąd w pętli monitoringu: {e}")
            self.running = False

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
            return {"error": str(e)}

    def _collect_metrics(self):
        """Zbiera metryki systemu."""
        try:
            # Pobierz metryki
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Zapisz metryki
            self.metrics["cpu"].append(cpu_percent)
            self.metrics["memory"].append(memory.percent)
            self.metrics["disk"].append(disk.percent)
            self.metrics["timestamp"].append(time.time())

            # Ogranicz historię
            if len(self.metrics["cpu"]) > self.max_history:
                for key in self.metrics:
                    self.metrics[key] = self.metrics[key][-self.max_history:]

            # Log jeśli wykryto wysokie zużycie
            if cpu_percent > 80:
                logger.warning(f"Wysokie zużycie CPU: {cpu_percent}%")
            if memory.percent > 80:
                logger.warning(f"Wysokie zużycie pamięci: {memory.percent}%")
            if disk.percent > 80:
                logger.warning(f"Niski poziom wolnej przestrzeni dyskowej: {disk.percent}%")

        except Exception as e:
            logger.error(f"Błąd podczas zbierania metryk: {e}")

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

    def get_current_usage(self) -> Dict[str, float]:
        """
        Pobiera aktualne zużycie zasobów.

        Returns:
            Dict[str, float]: Słownik zawierający bieżące zużycie zasobów
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu": cpu_percent,
                "memory": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Błąd podczas pobierania bieżącego zużycia: {e}")
            return {
                "cpu": 0.0,
                "memory": 0.0,
                "memory_available_mb": 0.0,
                "disk": 0.0,
                "disk_free_gb": 0.0,
                "timestamp": time.time(),
                "error": str(e)
            }

    def get_metrics_history(self) -> Dict[str, List[float]]:
        """
        Pobiera historię metryk.

        Returns:
            Dict[str, List[float]]: Słownik zawierający historię metryk
        """
        return self.metrics

    def get_system_info(self) -> Dict[str, Any]:
        """
        Pobiera informacje o systemie.

        Returns:
            Dict[str, Any]: Informacje o systemie
        """
        try:
            boot_time = psutil.boot_time()
            boot_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(boot_time))

            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_memory_gb": psutil.virtual_memory().total / (1024 * 1024 * 1024),
                "total_disk_gb": psutil.disk_usage('/').total / (1024 * 1024 * 1024),
                "boot_time": boot_time_str,
                "uptime_seconds": time.time() - boot_time
            }
        except Exception as e:
            logger.error(f"Błąd podczas pobierania informacji o systemie: {e}")
            return {
                "error": str(e)
            }


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
    return wrapper


def get_system_stats() -> Dict[str, Any]:
    """
    Funkcja pomocnicza do pobierania statystyk systemu.

    Returns:
        Dict[str, Any]: Statystyki systemu.
    """
    return performance_monitor.monitor_system()


# Funkcje pomocnicze dla łatwiejszego dostępu do wspólnej instancji
def get_performance_monitor() -> PerformanceMonitor:
    """
    Pobiera globalną instancję monitora wydajności.

    Returns:
        PerformanceMonitor: Instancja monitora wydajności
    """
    return performance_monitor


def get_system_usage() -> Dict[str, float]:
    """
    Pobiera aktualne zużycie zasobów systemowych.

    Returns:
        Dict[str, float]: Słownik zawierający bieżące zużycie zasobów
    """
    return performance_monitor.get_current_usage()