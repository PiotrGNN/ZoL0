"""
performance_monitor.py
----------------------
Moduł monitorujący wydajność systemu.
Funkcjonalności:
- Mierzy czasy wykonania kluczowych operacji, zużycie pamięci oraz obciążenie CPU.
- Generuje raporty trendów wydajności w dłuższym okresie.
- Umożliwia wysyłanie alertów, gdy wydajność spada poniżej ustalonych progów.
- Integruje się z modułami logowania oraz testami automatycznymi.
"""

import logging
import time

import psutil

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def measure_cpu_usage(interval: float = 1.0) -> float:
    """
    Mierzy zużycie CPU w procentach w określonym interwale.

    Parameters:
        interval (float): Czas pomiaru w sekundach.

    Returns:
        float: Średnie zużycie CPU w procentach.
    """
    cpu_usage = psutil.cpu_percent(interval=interval)
    logging.info("Zużycie CPU: %.2f%%", cpu_usage)
    return cpu_usage


def measure_memory_usage() -> float:
    """
    Mierzy zużycie pamięci RAM.

    Returns:
        float: Procentowe zużycie pamięci.
    """
    mem = psutil.virtual_memory()
    logging.info("Zużycie pamięci: %.2f%%", mem.percent)
    return mem.percent


def monitor_performance(duration: int = 60, check_interval: int = 5) -> dict:
    """
    Monitoruje wydajność systemu przez określony czas, mierząc CPU i zużycie pamięci.

    Parameters:
        duration (int): Całkowity czas monitorowania w sekundach.
        check_interval (int): Interwał między kolejnymi pomiarami w sekundach.

    Returns:
        dict: Raport zawierający listy z pomiarami CPU i pamięci oraz ich średnie wartości.
    """
    cpu_readings = []
    memory_readings = []
    start_time = time.time()

    while time.time() - start_time < duration:
        cpu = measure_cpu_usage(interval=check_interval)
        mem = measure_memory_usage()
        cpu_readings.append(cpu)
        memory_readings.append(mem)

    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    avg_memory = sum(memory_readings) / len(memory_readings) if memory_readings else 0

    report = {
        "cpu_usage": cpu_readings,
        "memory_usage": memory_readings,
        "average_cpu": avg_cpu,
        "average_memory": avg_memory,
    }
    logging.info("Monitoring wydajności zakończony. Raport: %s", report)
    return report


if __name__ == "__main__":
    try:
        # Przykładowe monitorowanie wydajności przez 30 sekund z interwałem 5 sekund
        monitor_performance(duration=30, check_interval=5)
    except Exception as e:
        logging.error("Błąd w module performance_monitor.py: %s", e)
        raise
"""
performance_monitor.py
--------------------
Moduł monitorujący wydajność systemu tradingowego.
"""

import logging
import time
import os
from typing import Dict, Any, List, Optional

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
    Klasa monitorująca wydajność systemu tradingowego.
    """

    def __init__(self):
        """Inicjalizacja monitora wydajności."""
        self.metrics = {}
        self.start_time = time.time()
        self.execution_times = {}
        self.memory_usage = {}
        self.api_calls = {}
        logger.info("Inicjalizacja monitora wydajności")

    def start_timer(self, operation_name: str) -> None:
        """
        Rozpoczyna pomiar czasu dla danej operacji.

        Parameters:
            operation_name (str): Nazwa operacji
        """
        self.metrics[operation_name] = {"start_time": time.time()}
        logger.debug(f"Rozpoczęto pomiar czasu dla operacji: {operation_name}")

    def stop_timer(self, operation_name: str) -> float:
        """
        Kończy pomiar czasu dla danej operacji i zwraca czas wykonania.

        Parameters:
            operation_name (str): Nazwa operacji

        Returns:
            float: Czas wykonania operacji w sekundach
        """
        if operation_name in self.metrics and "start_time" in self.metrics[operation_name]:
            start_time = self.metrics[operation_name]["start_time"]
            execution_time = time.time() - start_time
            
            if operation_name not in self.execution_times:
                self.execution_times[operation_name] = []
                
            self.execution_times[operation_name].append(execution_time)
            self.metrics[operation_name]["last_execution_time"] = execution_time
            
            logger.debug(f"Zakończono pomiar czasu dla operacji: {operation_name}, czas: {execution_time:.4f}s")
            return execution_time
        else:
            logger.warning(f"Nie znaleziono rozpoczętego pomiaru czasu dla operacji: {operation_name}")
            return 0.0

    def record_api_call(self, endpoint: str, success: bool = True) -> None:
        """
        Rejestruje wywołanie API.

        Parameters:
            endpoint (str): Nazwa endpointu API
            success (bool): Czy wywołanie zakończyło się sukcesem
        """
        if endpoint not in self.api_calls:
            self.api_calls[endpoint] = {"success": 0, "failure": 0}
            
        if success:
            self.api_calls[endpoint]["success"] += 1
        else:
            self.api_calls[endpoint]["failure"] += 1
            
        logger.debug(f"Zarejestrowano wywołanie API: {endpoint}, status: {'sukces' if success else 'błąd'}")

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generuje raport wydajności.

        Returns:
            Dict[str, Any]: Raport wydajności
        """
        total_runtime = time.time() - self.start_time
        
        # Obliczanie średnich czasów wykonania
        avg_execution_times = {}
        for operation, times in self.execution_times.items():
            if times:
                avg_execution_times[operation] = sum(times) / len(times)
        
        # Obliczanie procentu błędów API
        api_error_rates = {}
        for endpoint, counts in self.api_calls.items():
            total = counts["success"] + counts["failure"]
            if total > 0:
                api_error_rates[endpoint] = (counts["failure"] / total) * 100
        
        report = {
            "total_runtime": total_runtime,
            "avg_execution_times": avg_execution_times,
            "api_calls": self.api_calls,
            "api_error_rates": api_error_rates,
            "memory_usage": self.memory_usage
        }
        
        logger.info("Wygenerowano raport wydajności")
        return report

    def reset(self) -> None:
        """Resetuje monitor wydajności."""
        self.metrics = {}
        self.start_time = time.time()
        self.execution_times = {}
        self.memory_usage = {}
        self.api_calls = {}
        logger.info("Zresetowano monitor wydajności")
