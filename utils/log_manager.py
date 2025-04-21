"""
Zarządzanie rotacją logów i czyszczeniem starych metryk
"""
import os
import glob
import shutil
from datetime import datetime, timedelta
import logging
from typing import List
import json

class LogManager:
    """Zarządza rotacją logów i czyszczeniem starych metryk."""
    
    def __init__(self, base_dir: str = '.'):
        self.base_dir = base_dir
        self.logger = logging.getLogger('log_manager')
        if not self.logger.handlers:
            handler = logging.FileHandler('logs/log_manager.log')
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def cleanup_old_files(self, days: int = 7):
        """Usuwa pliki starsze niż określona liczba dni."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Lista katalogów do sprawdzenia
        directories = [
            os.path.join(self.base_dir, 'logs'),
            os.path.join(self.base_dir, 'reports')
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                continue
                
            self._cleanup_directory(directory, cutoff_date)
            
    def _cleanup_directory(self, directory: str, cutoff_date: datetime):
        """Czyści stare pliki z pojedynczego katalogu."""
        for file_path in glob.glob(os.path.join(directory, '*')):
            try:
                # Pomijamy katalogi
                if os.path.isdir(file_path):
                    continue
                    
                # Sprawdzamy datę modyfikacji pliku
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if mtime < cutoff_date:
                    # Przed usunięciem archiwizujemy ważne pliki
                    if self._should_archive(file_path):
                        self._archive_file(file_path)
                    else:
                        os.remove(file_path)
                        self.logger.info(f"Usunięto stary plik: {file_path}")
                        
            except Exception as e:
                self.logger.error(f"Błąd podczas czyszczenia pliku {file_path}: {e}")
                
    def _should_archive(self, file_path: str) -> bool:
        """Sprawdza czy plik powinien być zarchiwizowany zamiast usunięty."""
        # Archiwizujemy raporty wydajności i ważne metryki
        important_patterns = [
            '_performance_report_',
            '_metrics_summary_',
            'alert_statistics_'
        ]
        return any(pattern in os.path.basename(file_path) for pattern in important_patterns)
        
    def _archive_file(self, file_path: str):
        """Archiwizuje ważne pliki przed usunięciem."""
        archive_dir = os.path.join(self.base_dir, 'archived_logs')
        os.makedirs(archive_dir, exist_ok=True)
        
        # Tworzymy strukturę katalogów według roku/miesiąca
        date = datetime.fromtimestamp(os.path.getmtime(file_path))
        year_month = date.strftime('%Y/%m')
        target_dir = os.path.join(archive_dir, year_month)
        os.makedirs(target_dir, exist_ok=True)
        
        # Przenosimy plik do archiwum
        target_path = os.path.join(target_dir, os.path.basename(file_path))
        shutil.move(file_path, target_path)
        self.logger.info(f"Zarchiwizowano plik: {file_path} -> {target_path}")
        
    def compress_logs(self, max_size_mb: float = 100.0):
        """Kompresuje duże pliki logów."""
        for root, _, files in os.walk(os.path.join(self.base_dir, 'logs')):
            for file in files:
                if not file.endswith('.log'):
                    continue
                    
                file_path = os.path.join(root, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                if size_mb > max_size_mb:
                    self._compress_log_file(file_path)
                    
    def _compress_log_file(self, file_path: str):
        """Kompresuje pojedynczy plik logu."""
        try:
            # Tworzymy skompresowaną kopię
            import gzip
            compressed_path = file_path + '.gz'
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            # Usuwamy oryginalny plik po kompresji
            os.remove(file_path)
            self.logger.info(f"Skompresowano plik: {file_path} -> {compressed_path}")
            
        except Exception as e:
            self.logger.error(f"Błąd podczas kompresji pliku {file_path}: {e}")
            
    def generate_storage_report(self) -> dict:
        """Generuje raport o wykorzystaniu przestrzeni dyskowej."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'directories': {},
            'largest_files': [],
            'total_size': 0
        }
        
        # Zbieramy statystyki dla każdego katalogu
        monitored_dirs = ['logs', 'reports', 'archived_logs']
        for dir_name in monitored_dirs:
            dir_path = os.path.join(self.base_dir, dir_name)
            if os.path.exists(dir_path):
                stats = self._get_directory_stats(dir_path)
                report['directories'][dir_name] = stats
                report['total_size'] += stats['total_size']
                report['largest_files'].extend(stats['large_files'])
                
        # Sortujemy największe pliki
        report['largest_files'] = sorted(
            report['largest_files'],
            key=lambda x: x['size'],
            reverse=True
        )[:10]  # Top 10 największych plików
        
        # Zapisujemy raport
        report_path = os.path.join(
            self.base_dir,
            'reports',
            f'storage_report_{datetime.now().strftime("%Y%m%d")}.json'
        )
        
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            self.logger.error(f"Błąd podczas zapisywania raportu storage: {e}")
            
        return report
        
    def _get_directory_stats(self, directory: str) -> dict:
        """Zbiera statystyki dla pojedynczego katalogu."""
        stats = {
            'total_size': 0,
            'file_count': 0,
            'large_files': []
        }
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    stats['total_size'] += size
                    stats['file_count'] += 1
                    
                    # Zapisujemy informacje o dużych plikach (>10MB)
                    if size > 10 * 1024 * 1024:
                        stats['large_files'].append({
                            'path': os.path.relpath(file_path, self.base_dir),
                            'size': size,
                            'modified': datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            ).isoformat()
                        })
                except Exception as e:
                    self.logger.error(f"Błąd podczas analizy pliku {file_path}: {e}")
                    
        return stats