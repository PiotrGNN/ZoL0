#!/usr/bin/env python3
"""
model_tester.py - Module for testing ML models in the project.
"""

import os
import sys
import logging
import importlib
import numpy as np
import pandas as pd
import json
import time
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class ModelTester:
    """Implementation of ModelTester with model loading and validation functionality."""

    def __init__(self, models_path="ai_models", log_path="logs/model_tests.log"):
        """
        Initialize the ModelTester.

        Args:
            models_path: Path to the directory containing AI models
            log_path: Path to log file for test results
        """
        self.models_path = models_path
        self.log_path = log_path
        self.loaded_models = []
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"ModelTester initialized. Folder modeli: {models_path}, Log: {log_path}"
        )

        # Create a synthetic test dataset
        self.test_data = self._generate_test_data()

    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data for model validation."""
        data = pd.DataFrame()
        data["timestamp"] = pd.date_range(start="2024-01-01", periods=1000, freq="H")
        data["close"] = np.random.normal(100, 10, 1000).cumsum()
        data["volume"] = np.random.randint(1000, 10000, 1000)
        data["high"] = data["close"] * (1 + np.random.uniform(0, 0.02, 1000))
        data["low"] = data["close"] * (1 - np.random.uniform(0, 0.02, 1000))
        data["open"] = data["close"].shift(1)

        # Add some technical indicators
        data["sma_20"] = data["close"].rolling(window=20).mean()
        data["volatility"] = data["close"].pct_change().rolling(window=20).std()

        return data.dropna()

    def load_models(self) -> List[Dict[str, Any]]:
        """Load all available AI models from the specified directory."""
        models = []
        self.logger.info(f"Rozpoczęcie ładowania modeli...")

        try:
            # Get all Python modules in the directory
            if not os.path.exists(self.models_path):
                self.logger.error(f"Models path {self.models_path} does not exist")
                return models

            for filename in os.listdir(self.models_path):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = os.path.splitext(filename)[0]

                    try:
                        # Import the module
                        module = importlib.import_module(
                            f"{os.path.basename(self.models_path)}.{module_name}"
                        )

                        # Look for model classes in the module
                        for attr_name in dir(module):
                            if attr_name.startswith("__"):
                                continue

                            try:
                                attr = getattr(module, attr_name)

                                # Check if it's a class with predict/fit methods
                                if hasattr(attr, "__class__") and (
                                    hasattr(attr, "predict") or hasattr(attr, "fit")
                                ):

                                    models.append(
                                        {
                                            "name": attr_name,
                                            "instance": attr,
                                            "module": module_name,
                                            "has_predict": hasattr(attr, "predict"),
                                            "has_fit": hasattr(attr, "fit"),
                                        }
                                    )
                                    self.logger.info(
                                        f"Załadowano model {attr_name} z modułu {module_name}"
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Error inspecting {attr_name} in {module_name}: {e}"
                                )

                    except Exception as e:
                        self.logger.error(f"Błąd podczas wyszukiwania modeli: {e}")

            self.loaded_models = models
            return models

        except Exception as e:
            self.logger.error(f"Błąd podczas ładowania modeli: {e}")
            traceback.print_exc()
            return models

    def run_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run tests on all loaded models."""
        if not self.loaded_models:
            self.load_models()

        results = {}
        for model_info in self.loaded_models:
            try:
                result = self._test_model(model_info)
                results[model_info["name"]] = result
            except Exception as e:
                self.logger.error(f"Error testing {model_info['name']}: {e}")
                results[model_info["name"]] = {"success": False, "error": str(e)}

        self._save_test_results(results)
        return results

    def _test_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual model with validation."""
        result = {
            "success": False,
            "name": model_info["name"],
            "has_predict": model_info["has_predict"],
            "has_fit": model_info["has_fit"],
        }

        model = model_info.get("instance")
        if not model:
            return {**result, "error": "No model instance provided"}

        try:
            # Test prediction if available
            if hasattr(model, "predict"):
                try:
                    start_time = time.time()
                    _ = model.predict(self.test_data)
                    execution_time = time.time() - start_time

                    result.update(
                        {
                            "success": True,
                            "predict_successful": True,
                            "execution_time": execution_time,
                        }
                    )
                except Exception as e:
                    result.update(
                        {"predict_successful": False, "predict_error": str(e)}
                    )

            # Test training if available
            if hasattr(model, "fit"):
                try:
                    start_time = time.time()
                    model.fit(self.test_data)
                    execution_time = time.time() - start_time

                    result.update(
                        {"fit_successful": True, "training_time": execution_time}
                    )
                except Exception as e:
                    result.update({"fit_successful": False, "fit_error": str(e)})

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def evaluate_model(
        self, model_name: str, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate a specific model with test data."""
        for model_info in self.loaded_models:
            if model_info["name"] == model_name:
                model = model_info.get("instance")
                if model and hasattr(model, "predict"):
                    try:
                        predictions = model.predict(X_test)

                        # Calculate metrics based on type of problem
                        if len(np.unique(y_test)) <= 5:  # Classification
                            from sklearn.metrics import accuracy_score, f1_score

                            accuracy = accuracy_score(y_test, predictions)
                            f1 = f1_score(y_test, predictions, average="weighted")
                            return {
                                "success": True,
                                "accuracy": float(accuracy),
                                "f1_score": float(f1),
                                "samples_tested": len(X_test),
                            }
                        else:  # Regression
                            from sklearn.metrics import mean_squared_error, r2_score

                            mse = mean_squared_error(y_test, predictions)
                            r2 = r2_score(y_test, predictions)
                            return {
                                "success": True,
                                "mse": float(mse),
                                "r2": float(r2),
                                "samples_tested": len(X_test),
                            }
                    except Exception as e:
                        return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Model {model_name} not found"}

    def save_model_metadata(self, models_metadata: Dict[str, Any]) -> bool:
        """Save metadata about trained models."""
        try:
            metadata_file = "models/models_metadata.json"
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

            # Update existing data if file exists
            existing_data = {}
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        existing_data = json.load(f)
                except:
                    pass

            # Update with new metadata
            existing_data.update(models_metadata)

            # Save back to file
            with open(metadata_file, "w") as f:
                json.dump(existing_data, f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Error saving model metadata: {e}")
            return False

    def _save_test_results(self, results: Dict[str, Any]) -> None:
        """Save test results to file."""
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{os.path.splitext(self.log_path)[0]}_{timestamp}.json"

            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)

            self.logger.info(f"Test results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Error saving test results: {e}")
