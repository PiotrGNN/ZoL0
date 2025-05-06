#!/usr/bin/env python3
"""
test_models.py - Script for testing AI models in the project with enhanced validation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List, Optional
import time
import pickle
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/model_tests.log")
    ]
)
logger = logging.getLogger("test_models")

# Try importing tensorflow and keras but make them optional
try:
    import tensorflow as tf
    import keras
    from keras.optimizers import Adam
    from keras.models import Sequential
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    keras = None
    HAS_TENSORFLOW = False
    logger.info("Tensorflow/Keras not installed - some functionality will be limited")

# Upewnij się, że mamy dostęp do modułów projektu
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importuj tester modeli
try:
    from python_libs.model_tester import ModelTester
except ImportError:
    logger.error("❌ Nie znaleziono modułu model_tester w python_libs")
    logger.info("Utworzę własną implementację testową ModelTester")

    class ModelTester:
        """Enhanced implementation of ModelTester with parallel testing and validation."""

        def __init__(self, models_path='ai_models', log_path='logs/model_tests.log'):
            self.models_path = models_path
            self.log_path = log_path
            self.loaded_models = []
            self.logger = logging.getLogger("ModelTester")
            self.logger.info(f"ModelTester initialized. Models folder: {models_path}, Log: {log_path}")
            self._test_lock = threading.Lock()
            self.test_data = self._generate_test_data()

        def _generate_test_data(self) -> pd.DataFrame:
            """Generate synthetic test data for model validation."""
            periods = 1000
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='1h')
            base_price = 100
            
            # Generate OHLCV data
            data = pd.DataFrame(index=dates)
            data['close'] = base_price * (1 + np.random.randn(periods).cumsum() * 0.02)
            data['volume'] = np.random.lognormal(10, 1, periods)
            data['high'] = data['close'] * (1 + np.random.uniform(0, 0.03, periods))
            data['low'] = data['close'] * (1 - np.random.uniform(0, 0.03, periods))
            data['open'] = data['close'].shift(1)
            
            # Add technical indicators
            data['rsi_14'] = self._calculate_rsi(data['close'])
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['volatility'] = data['close'].pct_change().rolling(window=20).std()
            
            return data.dropna()

        def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
            """Calculate RSI technical indicator."""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        def run_tests(self) -> Dict[str, Dict[str, Any]]:
            """Run tests on all models in parallel with enhanced validation."""
            test_results = {}
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_model = {
                    executor.submit(self._test_model, model_info): model_info['name']
                    for model_info in self.loaded_models
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        test_results[model_name] = result
                    except Exception as e:
                        logger.error(f"Error testing {model_name}: {e}")
                        test_results[model_name] = {
                            "success": False,
                            "error": str(e)
                        }
            
            self._save_test_results(test_results)
            return test_results

        def _test_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
            """Test individual model with comprehensive validation."""
            model = model_info['instance']
            model_name = model_info['name']
            
            try:
                with self._test_lock:  # Prevent concurrent logging
                    self.logger.info(f"Testing {model_name}...")
                
                result = {
                    "success": True,
                    "model_type": model.__class__.__name__,
                    "methods": {
                        "predict": hasattr(model, "predict"),
                        "fit": hasattr(model, "fit")
                    }
                }
                
                # Test prediction if available
                if hasattr(model, "predict"):
                    try:
                        start_time = time.time()
                        predictions = model.predict(self.test_data)
                        execution_time = time.time() - start_time
                        
                        result.update({
                            "predict_successful": True,
                            "prediction_time": execution_time,
                            "prediction_shape": str(np.asarray(predictions).shape)
                        })
                    except Exception as e:
                        result.update({
                            "predict_successful": False,
                            "predict_error": str(e)
                        })
                
                # Test training if available
                if hasattr(model, "fit"):
                    try:
                        start_time = time.time()
                        model.fit(self.test_data)
                        execution_time = time.time() - start_time
                        
                        result.update({
                            "fit_successful": True,
                            "training_time": execution_time
                        })
                    except Exception as e:
                        result.update({
                            "fit_successful": False,
                            "fit_error": str(e)
                        })
                
                # Validate model attributes and parameters
                result["model_info"] = self._get_model_info(model)
                
                return result
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }

        def _get_model_info(self, model: Any) -> Dict[str, Any]:
            """Extract model parameters and attributes."""
            info = {
                "attributes": [],
                "parameters": {}
            }
            
            # Get public attributes
            for attr in dir(model):
                if not attr.startswith('_'):
                    try:
                        value = getattr(model, attr)
                        if not callable(value):
                            info["attributes"].append(attr)
                            if isinstance(value, (int, float, str, bool)):
                                info["parameters"][attr] = value
                    except:
                        continue
            
            return info

        def _save_test_results(self, results: Dict[str, Any]):
            """Save test results to file."""
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"logs/model_test_results_{timestamp}.json"
                
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                self.logger.info(f"Test results saved to {filepath}")
            except Exception as e:
                self.logger.error(f"Error saving test results: {e}")

        def load_models(self, force_retrain: bool = False) -> List[Dict[str, Any]]:
            """Load all available AI models with validation."""
            models = []
            
            # Load Anomaly Detector
            try:
                from ai_models.anomaly_detection import AnomalyDetector
                anomaly_detector = AnomalyDetector()
                models.append({
                    "name": "AnomalyDetector",
                    "instance": anomaly_detector,
                    "has_predict": hasattr(anomaly_detector, "predict"),
                    "has_fit": hasattr(anomaly_detector, "fit")
                })
                self.logger.info("Loaded AnomalyDetector successfully")
            except Exception as e:
                self.logger.error(f"Error loading AnomalyDetector: {e}")

            # Load Model Recognizer
            try:
                from ai_models.model_recognition import ModelRecognizer
                model_recognizer = ModelRecognizer()
                models.append({
                    "name": "ModelRecognizer",
                    "instance": model_recognizer,
                    "has_predict": hasattr(model_recognizer, "predict"),
                    "has_fit": hasattr(model_recognizer, "fit")
                })
                self.logger.info("Loaded ModelRecognizer successfully")
            except Exception as e:
                self.logger.error(f"Error loading ModelRecognizer: {e}")

            # Load Sentiment Analyzer
            try:
                from ai_models.sentiment_ai import SentimentAnalyzer
                sentiment_analyzer = SentimentAnalyzer()
                models.append({
                    "name": "SentimentAnalyzer",
                    "instance": sentiment_analyzer,
                    "has_predict": hasattr(sentiment_analyzer, "predict"),
                    "has_fit": hasattr(sentiment_analyzer, "fit")
                })
                self.logger.info("Loaded SentimentAnalyzer successfully")
            except Exception as e:
                self.logger.error(f"Error loading SentimentAnalyzer: {e}")

            self.loaded_models = models
            return models

        def get_loaded_models(self) -> List[Dict[str, Any]]:
            """Return loaded models with validation info."""
            return self.loaded_models


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

def test_models(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Testuje modele AI w projekcie.

    Args:
        force_retrain: Czy wymuszać ponowne trenowanie modeli

    Returns:
        Dict[str, Any]: Wyniki testów
    """
    import os
    import pickle
    import joblib
    import datetime

    # Upewnij się, że katalog models istnieje
    os.makedirs('models', exist_ok=True)

    print("🔍 Rozpoczęcie testowania modeli AI...")

    # Inicjalizuj tester modeli
    model_tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')

    # Uruchom testy
    results = model_tester.run_tests()

    # Pobierz załadowane modele
    models = model_tester.get_loaded_models()

    # Wygeneruj dane testowe
    X_train, X_test, y_train, y_test = generate_test_data()

    # Testuj modele typu ML z metodami fit/predict
    ml_results = {}
    for model_info in models:
        model_name = model_info['name']
        instance = model_info.get('instance')

        # Sprawdź czy model ma metody fit i predict
        if instance and model_info.get('has_fit') and model_info.get('has_predict'):
            # Sprawdź czy to RandomForestRegressor
            if model_name == "RandomForestRegressor":
                model_path = f"models/randomforest_model.pkl"

                # Jeśli wymuszamy retrenowanie lub plik nie istnieje
                if force_retrain or not os.path.exists(model_path):
                    print(f"📊 Trenowanie modelu {model_name} od zera...")
                    instance.fit(X_train, y_train)

                    # Zapisz wytrenowany model
                    model_data = {
                        "model": instance,
                        "metadata": {
                            "train_date": datetime.datetime.now().isoformat(),
                            "features_shape": X_train.shape,
                            "accuracy": instance.score(X_test, y_test),
                            "trained_from_scratch": True,
                            "training_samples": len(X_train)
                        }
                    }

                    try:
                        # Utwórz katalog models jeśli nie istnieje
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)

                        # Zapisz do pliku
                        with open(model_path, 'wb') as f:
                            pickle.dump(model_data, f)
                        print(f"💾 Model {model_name} zapisany do {model_path}")
                    except Exception as e:
                        print(f"❌ Błąd podczas zapisywania modelu {model_name}: {e}")
                else:
                    # Ładuj model z pliku
                    try:
                        with open(model_path, 'rb') as f:
                            model_data = pickle.load(f)
                            if "model" in model_data:
                                instance = model_data["model"]
                                meta = model_data.get("metadata", {})
                                train_date = meta.get("train_date", "nieznana data")
                                accuracy = meta.get("accuracy", "nieznana")
                                print(f"📂 Załadowano model {model_name} z pliku {model_path}")
                                print(f"   📊 Data treningu: {train_date}, Dokładność: {accuracy}")
                    except Exception as e:
                        print(f"❌ Błąd podczas ładowania modelu {model_name}: {e}")
                        print(f"   🔄 Trenuję model od nowa...")
                        instance.fit(X_train, y_train)

                        # Zapisz nowo wytrenowany model po błędzie ładowania
                        try:
                            model_data = {
                                "model": instance,
                                "metadata": {
                                    "train_date": datetime.datetime.now().isoformat(),
                                    "features_shape": X_train.shape,
                                    "accuracy": instance.score(X_test, y_test),
                                    "retrained_after_error": True
                                }
                            }
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            with open(model_path, 'wb') as f:
                                pickle.dump(model_data, f)
                            print(f"💾 Model {model_name} zapisany do {model_path} po ponownym treningu")
                        except Exception as e:
                            print(f"❌ Błąd podczas zapisywania modelu po retreningu: {e}")

            print(f"⏳ Testowanie modelu ML: {model_name}...")

            # Handle tensorflow imports safely
            if HAS_TENSORFLOW and isinstance(instance, Sequential):
                if len(instance.layers) == 0:
                    print(f"⚠️ Model {model_name} (Sequential) nie ma warstw, pomijam test")
                    continue
                # Kompilacja modelu Sequential jeśli nie został skompilowany
                if not hasattr(instance, 'optimizer'):
                    print(f"🔧 Kompiluję model {model_name} (Sequential)")
                    try:
                        instance.compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss="mse",
                            metrics=["mae", "mse"]
                        )
                    except Exception as e:
                        logger.warning(f"Nie można skompilować modelu Sequential: {e}")
                        continue

            try:
                # Trenuj model
                instance.fit(X_train, y_train)

                # Oceń model
                evaluation = model_tester.evaluate_model(model_name, X_test, y_test)
                ml_results[model_name] = evaluation

                print(f"✅ Model {model_name} przetestowany pomyślnie!")
            except Exception as e:
                print(f"❌ Błąd podczas testowania modelu {model_name}: {e}")
                ml_results[model_name] = {
                    "success": False, 
                    "error": str(e)
                }

    # Połącz wyniki
    final_results = {**results, **ml_results}

    # Podsumowanie
    success_count = sum(1 for result in final_results.values() if result.get('success', False))
    print(f"\n📊 Podsumowanie testów:")
    print(f"   - Przetestowano {len(final_results)} modeli")
    print(f"   - Pomyślnie: {success_count}")
    print(f"   - Niepowodzenia: {len(final_results) - success_count}")

    return final_results

def main():
    """
    Główna funkcja testu modeli z obsługą błędów i raportowaniem.
    """
    import argparse
    import json
    import time

    # Parsowanie argumentów linii poleceń
    parser = argparse.ArgumentParser(description="Test modeli AI")
    parser.add_argument('--force-retrain', action='store_true', help='Wymuś ponowne trenowanie modeli')
    parser.add_argument('--verbose', action='store_true', help='Zwiększona ilość logów')
    args = parser.parse_args()

    # Ustawienie poziomu logowania
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Inicjalizacja testera modeli
    tester = ModelTester(models_path='ai_models', log_path='logs/model_tests.log')

    # Statystyki testów
    test_stats = {
        "total_models": 0,
        "successful_tests": 0,
        "failed_tests": 0,
        "model_results": {}
    }

    # Ładowanie i testowanie modeli
    loaded_models = tester.load_models(force_retrain=args.force_retrain) # Assumed load_models method added to ModelTester
    test_stats["total_models"] = len(loaded_models)

    for model_info in loaded_models:
        model_name = model_info.get('name', 'Nieznany model')
        model_instance = model_info.get('instance')

        if model_instance:
            print(f"Testowanie modelu: {model_name}")

            # Sprawdzenie metod
            has_predict = hasattr(model_instance, 'predict')
            has_fit = hasattr(model_instance, 'fit')

            print(f"  - Metoda predict: {'Tak' if has_predict else 'Nie'}")
            print(f"  - Metoda fit: {'Tak' if has_fit else 'Nie'}")

            start_time = time.time()
            result = tester.test_model(model_instance, model_name)
            test_time = time.time() - start_time

            # Zapisz wyniki testu
            test_stats["model_results"][model_name] = {
                "success": result,
                "test_time": f"{test_time:.2f}s",
                "has_predict": has_predict,
                "has_fit": has_fit
            }

            if result:
                test_stats["successful_tests"] += 1
                print(f"  ✅ Wynik testu: Sukces ({test_time:.2f}s)")
            else:
                test_stats["failed_tests"] += 1
                print(f"  ❌ Wynik testu: Błąd ({test_time:.2f}s)")
        else:
            print(f"❌ Nie udało się załadować modelu: {model_name}")
            test_stats["model_results"][model_name] = {
                "success": False,
                "error": "Nie udało się załadować modelu"
            }
            test_stats["failed_tests"] += 1

    # Wyświetlenie podsumowania
    successful_models = [name for name, stats in test_stats["model_results"].items() if stats.get("success", False)]
    failed_models = [name for name, stats in test_stats["model_results"].items() if not stats.get("success", False)]

    print("\n=== PODSUMOWANIE TESTÓW ===")
    print(f"Załadowano {test_stats['total_models']} modeli")
    print(f"Pomyślnie przetestowano: {test_stats['successful_tests']} modeli")
    print(f"Niepowodzenia: {test_stats['failed_tests']} modeli")

    if successful_models:
        print("\nModele działające poprawnie:")
        for model_name in successful_models:
            print(f" - {model_name}")

    if failed_models:
        print("\nModele z błędami:")
        for model_name in failed_models:
            print(f" - {model_name}")

    try:
        os.makedirs('reports', exist_ok=True)
        report_path = f'reports/model_test_report_{time.strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(test_stats, f, indent=2)
        print(f"\nRaport testów zapisany do {report_path}")
    except Exception as e:
        print(f"Błąd podczas zapisywania raportu: {e}")

    return test_stats["failed_tests"] == 0  # Zwraca True, jeśli wszystkie testy się powiodły

if __name__ == "__main__":
    logger.info("Starting model tests...")
    
    # Create tester instance
    tester = ModelTester()
    
    # Load and test models
    loaded_models = tester.load_models()
    logger.info(f"Loaded {len(loaded_models)} models")
    
    # Run tests
    results = tester.run_tests()
    
    # Print results summary
    print("\nTest Results Summary:")
    print("-" * 50)
    for model_name, result in results.items():
        status = "✅ Success" if result.get("success", False) else "❌ Failed"
        error = f" ({result.get('error', '')})" if not result.get("success", False) else ""
        print(f"{model_name}: {status}{error}")
    print("-" * 50)