"""
Kompleksowe testy wydajnościowe systemu obejmujące:
- Pomiary wydajności API
- Testy obciążeniowe
- Profilowanie pamięci
- Testy wycieków pamięci
- Testy równoległego przetwarzania
"""

import os
import sys
import time
import logging
import psutil
import unittest
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Dodaj ścieżkę główną projektu do PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.model_training import ModelTrainer
from ai_models.sentiment_ai import SentimentAnalyzer
from ai_models.anomaly_detection import AnomalyDetector
from data.utils.api_handler import APIHandler

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Inicjalizacja zasobów dla wszystkich testów."""
        cls.model_trainer = ModelTrainer()
        cls.sentiment_analyzer = SentimentAnalyzer()
        cls.anomaly_detector = AnomalyDetector()
        cls.api_handler = APIHandler()
        
        # Przygotuj przykładowe dane testowe
        cls.sample_data = cls._generate_sample_data()
        
        # Zapisz początkowe zużycie zasobów
        cls.initial_memory = psutil.Process().memory_info().rss
        cls.start_time = time.time()
        
    @classmethod
    def tearDownClass(cls):
        """Sprzątanie po wszystkich testach."""
        # Zapisz raport wydajności
        cls._save_performance_report()
        
    def setUp(self):
        """Przygotowanie przed każdym testem."""
        self.test_start_time = time.time()
        self.test_start_memory = psutil.Process().memory_info().rss
        
    def tearDown(self):
        """Sprzątanie po każdym teście."""
        # Sprawdź wycieki pamięci
        current_memory = psutil.Process().memory_info().rss
        memory_diff = current_memory - self.test_start_memory
        
        if memory_diff > 10 * 1024 * 1024:  # 10MB
            logger.warning(
                f"Możliwy wyciek pamięci w teście {self._testMethodName}: "
                f"{memory_diff / 1024 / 1024:.2f} MB"
            )
            
        # Zapisz metryki wydajności testu
        self._save_test_metrics(
            test_name=self._testMethodName,
            execution_time=time.time() - self.test_start_time,
            memory_usage=memory_diff
        )
        
    @staticmethod
    def _generate_sample_data() -> Dict[str, np.ndarray]:
        """Generuje przykładowe dane do testów."""
        np.random.seed(42)
        sample_size = 1000
        
        return {
            'features': np.random.randn(sample_size, 10),
            'labels': np.random.randint(0, 2, size=sample_size),
            'time_series': np.random.randn(sample_size, 5),
            'text_data': [
                f"Sample text {i} with random sentiment" for i in range(sample_size)
            ]
        }
        
    @classmethod
    def _save_performance_report(cls):
        """Zapisuje szczegółowy raport wydajności."""
        total_time = time.time() - cls.start_time
        total_memory = psutil.Process().memory_info().rss - cls.initial_memory
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'total_memory_usage': total_memory,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'test_metrics': cls.test_metrics if hasattr(cls, 'test_metrics') else []
        }
        
        # Zapisz raport do pliku
        report_path = os.path.join(
            'reports',
            f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        os.makedirs('reports', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Zapisano raport wydajności: {report_path}")
        
    def _save_test_metrics(
        self,
        test_name: str,
        execution_time: float,
        memory_usage: int
    ):
        """Zapisuje metryki pojedynczego testu."""
        if not hasattr(self.__class__, 'test_metrics'):
            self.__class__.test_metrics = []
            
        self.__class__.test_metrics.append({
            'test_name': test_name,
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'timestamp': datetime.now().isoformat()
        })
        
    @profile
    def test_model_training_performance(self):
        """Test wydajności treningu modeli."""
        X = self.sample_data['features']
        y = self.sample_data['labels']
        
        start_time = time.time()
        models = {
            'lstm': self.model_trainer.create_model('lstm', {
                'input_size': X.shape[1],
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2
            })
        }
        
        results = self.model_trainer.train_multiple_models(models, X, y)
        execution_time = time.time() - start_time
        
        self.assertLess(execution_time, 300)  # Max 5 minut
        self.assertGreater(
            results['lstm']['metrics']['accuracy'],
            0.6  # Minimalna oczekiwana dokładność
        )
        
    def test_parallel_processing_performance(self):
        """Test wydajności przetwarzania równoległego."""
        def process_chunk(chunk: np.ndarray) -> float:
            return np.mean(chunk)
        
        data = self.sample_data['time_series']
        chunks = np.array_split(data, 4)
        
        # Test z ThreadPoolExecutor
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            thread_results = list(executor.map(process_chunk, chunks))
        thread_time = time.time() - start_time
        
        # Test z ProcessPoolExecutor
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            process_results = list(executor.map(process_chunk, chunks))
        process_time = time.time() - start_time
        
        self.assertLess(thread_time, 1.0)  # Max 1 sekunda
        self.assertLess(process_time, 2.0)  # Max 2 sekundy
        
    def test_api_response_time(self):
        """Test czasu odpowiedzi API."""
        endpoints = [
            '/api/portfolio',
            '/api/sentiment',
            '/api/market/analyze'
        ]
        
        response_times = []
        for endpoint in endpoints:
            start_time = time.time()
            try:
                self.api_handler.get(endpoint)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Sprawdź czy czas odpowiedzi jest akceptowalny
                self.assertLess(response_time, 1.0)  # Max 1 sekunda
            except Exception as e:
                logger.error(f"Błąd podczas testowania endpointu {endpoint}: {e}")
                
        avg_response_time = np.mean(response_times)
        logger.info(f"Średni czas odpowiedzi API: {avg_response_time:.3f}s")
        
    def test_memory_usage_under_load(self):
        """Test zużycia pamięci pod obciążeniem."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Generuj obciążenie
        data = []
        for _ in range(100):
            data.append(np.random.randn(1000, 1000))
            time.sleep(0.01)  # Krótka przerwa
            
            current_memory = psutil.Process().memory_info().rss
            memory_usage = current_memory - initial_memory
            
            # Sprawdź czy zużycie pamięci jest w akceptowalnym zakresie
            self.assertLess(
                memory_usage,
                1024 * 1024 * 1024  # Max 1GB
            )
            
        # Wyczyść dane
        del data
        
    def test_sentiment_analysis_performance(self):
        """Test wydajności analizy sentymentu."""
        texts = self.sample_data['text_data']
        
        start_time = time.time()
        for text in texts:
            sentiment = self.sentiment_analyzer.analyze_text(text)
            self.assertIn('sentiment', sentiment)
            
        execution_time = time.time() - start_time
        avg_time_per_text = execution_time / len(texts)
        
        self.assertLess(avg_time_per_text, 0.1)  # Max 100ms na tekst
        
    def test_anomaly_detection_performance(self):
        """Test wydajności wykrywania anomalii."""
        data = self.sample_data['time_series']
        
        start_time = time.time()
        anomalies = self.anomaly_detector.detect(data)
        execution_time = time.time() - start_time
        
        self.assertLess(execution_time, 5.0)  # Max 5 sekund
        self.assertIsInstance(anomalies, np.ndarray)
        
    def test_concurrent_api_requests(self):
        """Test wydajności równoległych zapytań API."""
        num_requests = 50
        endpoint = '/api/portfolio'
        
        def make_request():
            start_time = time.time()
            try:
                self.api_handler.get(endpoint)
                return time.time() - start_time
            except Exception as e:
                logger.error(f"Błąd podczas równoległego zapytania: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            response_times = list(executor.map(
                lambda _: make_request(),
                range(num_requests)
            ))
            
        response_times = [t for t in response_times if t is not None]
        avg_response_time = np.mean(response_times)
        max_response_time = max(response_times)
        
        self.assertLess(avg_response_time, 0.5)  # Średnio max 500ms
        self.assertLess(max_response_time, 2.0)  # Max 2 sekundy
        
    def test_data_processing_pipeline(self):
        """Test wydajności przetwarzania danych."""
        data = self.sample_data['features']
        
        # Symulacja przetwarzania danych
        start_time = time.time()
        
        # 1. Preprocessing
        normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        # 2. Feature extraction
        features = np.column_stack([
            np.mean(normalized_data, axis=1),
            np.std(normalized_data, axis=1),
            np.max(normalized_data, axis=1),
            np.min(normalized_data, axis=1)
        ])
        
        # 3. Model prediction
        model = self.model_trainer.create_model('lstm', {
            'input_size': features.shape[1],
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.1
        })
        
        predictions = model(torch.FloatTensor(features))
        
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 10.0)  # Max 10 sekund
        
    def test_model_checkpoint_io(self):
        """Test wydajności operacji I/O z checkpointami."""
        model = self.model_trainer.create_model('lstm', {
            'input_size': 10,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        })
        
        # Test zapisu
        start_time = time.time()
        self.model_trainer._save_checkpoint(
            model,
            torch.optim.Adam(model.parameters()),
            epoch=0,
            valid_loss=1.0
        )
        save_time = time.time() - start_time
        
        self.assertLess(save_time, 1.0)  # Max 1 sekunda na zapis
        
        # Test odczytu
        checkpoint_files = [
            f for f in os.listdir('saved_models')
            if f.startswith('checkpoint_epoch_0_')
        ]
        if checkpoint_files:
            start_time = time.time()
            self.model_trainer.load_checkpoint(
                os.path.join('saved_models', checkpoint_files[0])
            )
            load_time = time.time() - start_time
            
            self.assertLess(load_time, 1.0)  # Max 1 sekunda na odczyt
            
    @classmethod
    def generate_performance_report(cls):
        """Generuje szczegółowy raport wydajności."""
        if not hasattr(cls, 'test_metrics'):
            return
            
        report = pd.DataFrame(cls.test_metrics)
        
        # Agregacja metryk
        summary = {
            'total_tests': len(report),
            'total_time': report['execution_time'].sum(),
            'avg_time': report['execution_time'].mean(),
            'max_time': report['execution_time'].max(),
            'total_memory': report['memory_usage'].sum(),
            'avg_memory': report['memory_usage'].mean(),
            'max_memory': report['memory_usage'].max()
        }
        
        # Zapisz szczegółowy raport
        report_path = os.path.join(
            'reports',
            f'detailed_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
        report.to_csv(report_path, index=False)
        
        # Zapisz podsumowanie
        summary_path = os.path.join(
            'reports',
            f'performance_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Zapisano szczegółowy raport: {report_path}")
        logger.info(f"Zapisano podsumowanie: {summary_path}")

if __name__ == '__main__':
    unittest.main()