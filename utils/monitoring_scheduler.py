"""
Scheduler do automatycznego zarządzania monitorowaniem
"""
import schedule
import time
import threading
from datetime import datetime
import logging
from typing import Callable, Dict, Any
from utils.api_analyzer import APIMetricsAnalyzer
from utils.log_manager import LogManager

class MonitoringScheduler:
    """Zarządza harmonogramem zadań monitorowania."""
    
    def __init__(self):
        self.logger = logging.getLogger('monitoring_scheduler')
        if not self.logger.handlers:
            handler = logging.FileHandler('logs/scheduler.log')
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        self.analyzer = APIMetricsAnalyzer()
        self.log_manager = LogManager()
        self.running = False
        self.scheduler_thread = None
        
    def start(self):
        """Uruchamia scheduler z domyślnym harmonogramem."""
        if self.running:
            self.logger.warning("Scheduler już jest uruchomiony")
            return
            
        # Konfiguracja harmonogramu zadań
        
        # Generowanie raportów wydajności co 6 godzin
        schedule.every(6).hours.do(self._run_task, 
            self.analyzer.generate_report,
            "Generowanie raportu wydajności"
        )
        
        # Czyszczenie starych plików raz dziennie
        schedule.every().day.at("01:00").do(self._run_task,
            lambda: self.log_manager.cleanup_old_files(days=7),
            "Czyszczenie starych plików"
        )
        
        # Kompresja logów co 12 godzin
        schedule.every(12).hours.do(self._run_task,
            lambda: self.log_manager.compress_logs(max_size_mb=100),
            "Kompresja dużych plików logów"
        )
        
        # Generowanie raportu o wykorzystaniu dysku raz dziennie
        schedule.every().day.at("02:00").do(self._run_task,
            self.log_manager.generate_storage_report,
            "Generowanie raportu storage"
        )
        
        # Uruchomienie schedulera w osobnym wątku
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.logger.info("Uruchomiono scheduler monitorowania")
        
    def stop(self):
        """Zatrzymuje scheduler."""
        if not self.running:
            return
            
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
            self.scheduler_thread = None
            
        schedule.clear()
        self.logger.info("Zatrzymano scheduler monitorowania")
        
    def _run_scheduler(self):
        """Główna pętla schedulera."""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Sprawdzamy co minutę
            except Exception as e:
                self.logger.error(f"Błąd w pętli schedulera: {e}")
                
    def _run_task(self, task: Callable, description: str) -> Any:
        """Uruchamia pojedyncze zadanie z obsługą błędów."""
        try:
            self.logger.info(f"Rozpoczęto: {description}")
            result = task()
            self.logger.info(f"Zakończono: {description}")
            return result
        except Exception as e:
            self.logger.error(f"Błąd podczas {description}: {e}")
            return None
            
    def add_custom_task(self, task: Callable, schedule_at: str, description: str):
        """
        Dodaje własne zadanie do harmonogramu.
        
        Przykład użycia:
        scheduler.add_custom_task(
            task=my_function,
            schedule_at="*/30 * * * *",  # Co 30 minut
            description="Moje zadanie"
        )
        """
        try:
            schedule.every().day.at(schedule_at).do(
                self._run_task,
                task,
                description
            )
            self.logger.info(f"Dodano nowe zadanie: {description} ({schedule_at})")
        except Exception as e:
            self.logger.error(f"Błąd podczas dodawania zadania {description}: {e}")
            
    def get_next_runs(self) -> Dict[str, datetime]:
        """Zwraca czasy następnego uruchomienia dla wszystkich zadań."""
        next_runs = {}
        for job in schedule.get_jobs():
            try:
                next_runs[job.tags[0]] = job.next_run
            except Exception as e:
                self.logger.error(f"Błąd podczas sprawdzania następnego uruchomienia: {e}")
                
        return next_runs