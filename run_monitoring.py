"""
Główny skrypt uruchomieniowy systemu monitorowania
"""
import os
import argparse
import logging
from utils.monitoring_scheduler import MonitoringScheduler
from utils.api_analyzer import APIMetricsAnalyzer
from utils.log_manager import LogManager
from utils.api_alerts import AlertManager

def setup_logging():
    """Konfiguracja systemu logowania."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Konfiguracja głównego loggera
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'monitoring.log')),
            logging.StreamHandler()  # Dodatkowo wyświetlamy logi w konsoli
        ]
    )

def main():
    """Główna funkcja uruchomieniowa."""
    parser = argparse.ArgumentParser(description='System monitorowania API')
    parser.add_argument('--retention-days', type=int, default=7,
                      help='Liczba dni przechowywania logów (domyślnie: 7)')
    parser.add_argument('--max-log-size', type=float, default=100.0,
                      help='Maksymalny rozmiar pliku logu w MB przed kompresją (domyślnie: 100)')
    parser.add_argument('--report-interval', type=int, default=6,
                      help='Częstotliwość generowania raportów w godzinach (domyślnie: 6)')
    args = parser.parse_args()
    
    # Konfiguracja logowania
    setup_logging()
    logger = logging.getLogger('run_monitoring')
    
    try:
        # Inicjalizacja komponentów
        logger.info("Inicjalizacja systemu monitorowania...")
        
        # Utworzenie wymaganych katalogów
        for directory in ['logs', 'reports', 'archived_logs']:
            os.makedirs(directory, exist_ok=True)
        
        # Inicjalizacja menedżera logów
        log_manager = LogManager()
        log_manager.cleanup_old_files(days=args.retention_days)
        log_manager.compress_logs(max_size_mb=args.max_log_size)
        
        # Wygenerowanie początkowego raportu storage
        storage_report = log_manager.generate_storage_report()
        logger.info(f"Aktualne wykorzystanie przestrzeni: "
                   f"{storage_report['total_size'] / (1024*1024):.2f} MB")
        
        # Inicjalizacja systemu alertów
        alert_manager = AlertManager()
        
        # Inicjalizacja analizatora metryk
        analyzer = APIMetricsAnalyzer()
        
        # Uruchomienie schedulera
        scheduler = MonitoringScheduler()
        
        # Dodanie dodatkowych zadań do harmonogramu
        scheduler.add_custom_task(
            lambda: analyzer.generate_report(days=1),
            "00:00",  # Codziennie o północy
            "Dzienny raport wydajności"
        )
        
        # Uruchomienie schedulera
        scheduler.start()
        
        logger.info("System monitorowania został uruchomiony")
        logger.info("Naciśnij Ctrl+C aby zatrzymać...")
        
        # Utrzymujemy skrypt uruchomiony
        while True:
            next_runs = scheduler.get_next_runs()
            for task, time in next_runs.items():
                logger.info(f"Następne wykonanie {task}: {time}")
            import time
            time.sleep(3600)  # Sprawdzamy co godzinę
            
    except KeyboardInterrupt:
        logger.info("Otrzymano sygnał zatrzymania...")
        scheduler.stop()
        logger.info("System monitorowania został zatrzymany")
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()