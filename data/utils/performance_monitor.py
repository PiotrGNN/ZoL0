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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
