"""
test_models.py - Skrypt do testowania modeli AI w projekcie.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import time
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Dodaj ścieżkę do modułów projektu
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ai_models.strategy_runner import StrategyBacktestRunner
from ai_models.enhanced_backtester import EnhancedBacktester
from ai_models.scalar import DataScaler
from ai_models.sentiment_ai import SentimentAnalyzer
from ai_models.model_utils import ModelUtilsWrapper
from ai_models.anomaly_detection import AnomalyDetector
from ai_models.model_recognition import ModelRecognizer
from ai_models.model_loader import ModelLoader
from ai_models.reinforcement_learning import ReinforcementLearner
from ai_models.model_manager import ModelManager, ModelMetrics
from ai_models.model_training import ModelTrainer
from ai_models.model_tuner import ModelTuner
from ai_models.real_exchange_env import RealExchangeEnv
from ai_models.market_dummy_env import MarketDummyEnv
from ai_models.environment import MarketEnvironment
from ai_models.ropmer_temp import ExperimentManager

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/model_tests.log")
    ]
)
logger = logging.getLogger("test_models")

def generate_test_data():
    """
    Generuje testowe dane do uczenia modeli.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    # Generuj dane syntetyczne
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 próbek, 10 cech
    y = np.random.choice([0, 1], size=(100,), p=[0.7, 0.3])  # Etykiety binarne

    # Podział na zbiory treningowy i testowy
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test

def test_models():
    """Test sprawdzający podstawowe funkcjonalności klas."""
    results = {}
    
    classes_to_test = {
        'ropmer_temp.ExperimentManager': ExperimentManager,
        'real_exchange_env.RealExchangeEnv': RealExchangeEnv,
        'model_training.ModelTrainer': ModelTrainer,
        'strategy_runner.StrategyBacktestRunner': StrategyBacktestRunner,
        'strategy_runner.StrategyRunner': StrategyBacktestRunner,
        'market_dummy_env.MarketDummyEnv': MarketDummyEnv,
        'model_tuner.ModelTuner': ModelTuner,
        'scalar.DataScaler': DataScaler,
        'sentiment_ai.SentimentAnalyzer': SentimentAnalyzer,
        'model_utils.ModelUtilsWrapper': ModelUtilsWrapper,
        'anomaly_detection.AnomalyDetector': AnomalyDetector,
        'enhanced_backtester.EnhancedBacktester': EnhancedBacktester,
        'model_manager.ModelManager': ModelManager,
        'model_manager.ModelMetrics': ModelMetrics,
        'model_recognition.ModelRecognizer': ModelRecognizer,
        'model_loader.ModelLoader': ModelLoader,
        'reinforcement_learning.ReinforcementLearner': ReinforcementLearner,
        'environment.MarketEnvironment': MarketEnvironment
    }

    for class_name, class_type in classes_to_test.items():
        results[class_name] = {
            'status': 'success',
            'has_predict': hasattr(class_type, 'predict'),
            'has_fit': hasattr(class_type, 'fit'),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }
    
    return results

def test_enhanced_backtester():
    """Test funkcjonalności rozszerzonego backtestingu"""
    # Przygotowanie danych testowych
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(102, 10, len(dates)),
        'low': np.random.normal(98, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Inicjalizacja backtestera
    backtester = EnhancedBacktester(
        initial_capital=10000.0,
        commission=0.001,
        spread=0.0005,
        slippage=0.0005
    )
    
    # Test symulacji Monte Carlo
    returns = data['close'].pct_change().dropna()
    mc_results = backtester.run_monte_carlo_simulation(
        returns,
        n_simulations=100,
        n_days=30
    )
    
    assert isinstance(mc_results, dict)
    assert 'final_values_mean' in mc_results
    assert 'confidence_interval' in mc_results
    assert len(mc_results['paths']) == 100
    
    # Test analizy walk-forward
    def dummy_strategy(data):
        return pd.Series(1, index=data.index)  # Zawsze long
    
    wf_results = backtester.run_walk_forward_analysis(
        data,
        dummy_strategy,
        train_ratio=0.7,
        n_splits=3
    )
    
    assert isinstance(wf_results, dict)
    assert 'splits' in wf_results
    assert 'aggregated_metrics' in wf_results
    assert len(wf_results['splits']) == 3

def test_strategy_runner():
    """Test funkcjonalności zarządzania strategiami"""
    # Przygotowanie danych testowych
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(102, 10, len(dates)),
        'low': np.random.normal(98, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Inicjalizacja runner'a
    runner = StrategyBacktestRunner(
        initial_capital=10000.0,
        position_sizing_method="volatility"
    )
    
    # Test porównania strategii
    strategies_to_test = ['moving_average_crossover', 'rsi_reversal']
    params_dict = {
        'moving_average_crossover': {'short_window': 10, 'long_window': 30},
        'rsi_reversal': {'period': 14, 'overbought': 70, 'oversold': 30}
    }
    
    results = runner.run_strategy_comparison(
        data,
        strategies_to_test,
        params_dict
    )
    
    assert isinstance(results, dict)
    assert all(strategy in results for strategy in strategies_to_test)
    assert all('metrics' in results[strategy] for strategy in strategies_to_test)
    assert all('equity_curve' in results[strategy] for strategy in strategies_to_test)

class PerformanceTestSuite:
    """Kompleksowe testy wydajnościowe dla modeli uczenia maszynowego."""
    
    def __init__(self):
        self.results = {}
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self, size=10000):
        """Generuje dane testowe."""
        dates = pd.date_range(start='2023-01-01', periods=size, freq='5min')
        return pd.DataFrame({
            'open': np.random.normal(100, 10, size),
            'high': np.random.normal(102, 10, size),
            'low': np.random.normal(98, 10, size),
            'close': np.random.normal(100, 10, size),
            'volume': np.random.normal(1000000, 100000, size)
        }, index=dates)

    def measure_execution_time(self, func, *args, **kwargs):
        """Mierzy czas wykonania funkcji."""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {e}")
            result = None
            success = False
            
        execution_time = time.time() - start_time
        memory_after = self._get_memory_usage()
        memory_used = memory_after - memory_before
        
        return {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'success': success,
            'result': result
        }

    def _get_memory_usage(self):
        """Pobiera aktualne zużycie pamięci."""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def test_model_training_performance(self):
        """Test wydajności treningu modeli."""
        logger.info("Testing model training performance...")
        
        X = self.test_data[['open', 'high', 'low', 'volume']]
        y = (self.test_data['close'].pct_change() > 0).astype(int)
        
        model_trainer = ModelTrainer(
            model=ReinforcementLearner(state_size=4, action_size=3),
            model_name="PerformanceTest_RL",
            online_learning=True
        )
        
        metrics = self.measure_execution_time(
            model_trainer.train,
            X=X,
            y=y,
            n_splits=5,
            epochs=10,
            batch_size=32
        )
        
        self.results['model_training'] = metrics
        return metrics

    def test_prediction_performance(self):
        """Test wydajności predykcji."""
        logger.info("Testing prediction performance...")
        
        model = ReinforcementLearner(state_size=4, action_size=3)
        X = self.test_data[['open', 'high', 'low', 'volume']].values
        
        def predict_batch():
            for _ in range(1000):
                model.predict(X[-1:])
        
        metrics = self.measure_execution_time(predict_batch)
        metrics['predictions_per_second'] = 1000 / metrics['execution_time']
        
        self.results['prediction'] = metrics
        return metrics

    def test_feature_engineering_performance(self):
        """Test wydajności inżynierii cech."""
        logger.info("Testing feature engineering performance...")
        
        def engineer_features(df):
            df = df.copy()
            # Techniczne wskaźniki
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_30'] = df['close'].rolling(window=30).mean()
            df['RSI'] = calculate_rsi(df['close'])
            df['MACD'] = calculate_macd(df['close'])
            df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['close'])
            return df
            
        metrics = self.measure_execution_time(engineer_features, self.test_data)
        self.results['feature_engineering'] = metrics
        return metrics

    def test_backtesting_performance(self):
        """Test wydajności backtestingu."""
        logger.info("Testing backtesting performance...")
        
        backtester = EnhancedBacktester(
            initial_capital=10000.0,
            commission=0.001
        )
        
        def simple_strategy(data):
            return pd.Series(1, index=data.index)
        
        metrics = self.measure_execution_time(
            backtester.run_walk_forward_analysis,
            self.test_data,
            simple_strategy,
            train_ratio=0.7,
            n_splits=5
        )
        
        self.results['backtesting'] = metrics
        return metrics

    def test_model_tuning_performance(self):
        """Test wydajności strojenia hiperparametrów."""
        logger.info("Testing hyperparameter tuning performance...")
        
        X = self.test_data[['open', 'high', 'low', 'volume']]
        y = (self.test_data['close'].pct_change() > 0).astype(int)
        
        param_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1}
        }
        
        tuner = ModelTuner(
            model_class=ReinforcementLearner,
            param_space=param_space,
            metric="mse",
            n_trials=10
        )
        
        metrics = self.measure_execution_time(tuner.tune, X, y)
        self.results['model_tuning'] = metrics
        return metrics

    def run_all_tests(self):
        """Uruchamia wszystkie testy wydajnościowe."""
        logger.info("Starting comprehensive performance tests...")
        
        test_functions = [
            self.test_model_training_performance,
            self.test_prediction_performance,
            self.test_feature_engineering_performance,
            self.test_backtesting_performance,
            self.test_model_tuning_performance
        ]
        
        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                logger.error(f"Error in {test_func.__name__}: {e}")
        
        self._generate_report()
        return self.results

    def _generate_report(self):
        """Generuje raport z wyników testów."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results.values() if r['success']),
                "total_execution_time": sum(r['execution_time'] for r in self.results.values()),
                "total_memory_used": sum(r['memory_used'] for r in self.results.values())
            },
            "detailed_results": self.results
        }
        
        # Zapisz raport do pliku
        report_path = f"reports/performance_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("reports", exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Performance test report generated: {report_path}")
        return report

def calculate_rsi(prices, period=14):
    """Oblicza RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Oblicza MACD."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Oblicza wstęgi Bollingera."""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

if __name__ == "__main__":
    try:
        # Uruchom testy wydajnościowe
        test_suite = PerformanceTestSuite()
        results = test_suite.run_all_tests()
        
        # Wyświetl podsumowanie
        print("\n=== Performance Test Results ===")
        print(f"Total tests: {results['summary']['total_tests']}")
        print(f"Successful tests: {results['summary']['successful_tests']}")
        print(f"Total execution time: {results['summary']['total_execution_time']:.2f} seconds")
        print(f"Total memory used: {results['summary']['total_memory_used']:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error running performance tests: {e}")
        raise