"""
test_ai_models.py
----------------
Test suite for AI model functionality including:
- Model loading and initialization
- Feature engineering and selection
- Training and prediction
- Error handling and edge cases
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from data.tests import BaseTestCase

from ai_models.model_loader import ModelLoader
from ai_models.feature_engineering import FeatureEngineer
from ai_models.model_training import ModelTrainer
from ai_models.model_evaluation import ModelEvaluator

import pytest
from ai_models.model_recognition import ModelRecognizer
from ai_models.anomaly_detection import AnomalyDetector

class TestAIModels(BaseTestCase):
    """Test AI model functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test-wide resources."""
        super().setUpClass()
        cls.sample_features = [
            "rsi_14", "macd", "bb_upper", "bb_lower",
            "atr_14", "adx_14", "cci_20", "mfi_14"
        ]

    def setUp(self):
        """Initialize test instance resources."""
        super().setUp()
        self.df = self.generate_test_data(periods=200)
        # Add technical indicators as features
        for feature in self.sample_features:
            self.df[feature] = np.random.normal(0, 1, len(self.df))
        self.df["target"] = self.df["close"].shift(-1) / self.df["close"] - 1
        self.df = self.df.dropna()
        
        # Create train/test split for model evaluation
        self.train_idx = int(len(self.df) * 0.8)
        self.train_data = self.df[:self.train_idx]
        self.test_data = self.df[self.train_idx:]

    @pytest.mark.critical
    def test_model_initialization(self):
        """Test model initialization and config loading."""
        loader = ModelLoader()
        models = loader.load_available_models()
        
        self.assertIsInstance(models, dict)
        self.assertGreater(len(models), 0, "Should load at least one model")
        
        for name, model in models.items():
            with self.subTest(model_name=name):
                self.assertTrue(hasattr(model, "predict"), f"{name} missing predict method")
                if hasattr(model, "fit"):
                    # Test model can be fit with sample data
                    X = self.df[self.sample_features]
                    y = self.df["target"]
                    try:
                        model.fit(X, y)
                        # Test prediction after fitting
                        preds = model.predict(X[:5])
                        self.assertEqual(len(preds), 5)
                        self.assertTrue(np.isfinite(preds).all())
                    except Exception as e:
                        self.fail(f"Model {name} failed to fit/predict: {e}")

        # Test error handling
        with self.assertRaises(ValueError):
            loader.load_model("nonexistent_model")

    @pytest.mark.critical
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        detector = AnomalyDetector()
        
        # Test with normal data
        normal_data = pd.DataFrame({
            'value': np.random.normal(0, 1, 100)
        })
        detector.fit(normal_data)
        result = detector.detect(normal_data)
        
        self.assertIn("anomaly_indices", result)
        self.assertIn("anomaly_scores", result)
        self.assertLess(
            len(result["anomaly_indices"]) / len(normal_data),
            0.1,  # Less than 10% should be anomalies in normal data
            "Too many anomalies detected in normal data"
        )

        # Test with injected anomalies
        anomaly_data = normal_data.copy()
        anomaly_indices = [20, 40, 60, 80]
        anomaly_data.iloc[anomaly_indices] = 100  # Obvious anomalies
        
        result = detector.detect(anomaly_data)
        detected_indices = set(result["anomaly_indices"])
        
        for idx in anomaly_indices:
            self.assertIn(
                idx, detected_indices,
                f"Failed to detect injected anomaly at index {idx}"
            )

    @pytest.mark.critical
    def test_model_recognition(self):
        """Test pattern recognition functionality."""
        recognizer = ModelRecognizer()
        
        # Create a clear bull flag pattern
        bull_flag_data = self.generate_test_data(periods=30)
        # Simulate uptrend
        bull_flag_data.loc[:9, "close"] *= np.linspace(1, 1.2, 10)
        # Simulate consolidation
        bull_flag_data.loc[10:, "close"] *= np.linspace(1.2, 1.18, 20)
        
        result = recognizer.identify_model_type(bull_flag_data)
        
        self.assertIn("pattern_type", result)
        self.assertIn("confidence", result)
        self.assertIn("bull_flag", result["pattern_type"].lower())
        self.assertGreater(result["confidence"], 0.7)  # High confidence threshold

    @pytest.mark.critical
    def test_model_training(self):
        """Test model training workflow."""
        trainer = ModelTrainer()
        
        # Test with different configs
        configs = [
            {"model_type": "random_forest", "n_estimators": 100},
            {"model_type": "gradient_boost", "n_estimators": 50}
        ]
        
        for config in configs:
            with self.subTest(config=config):
                model = trainer.train_model(
                    features=self.train_data[self.sample_features],
                    target=self.train_data["target"],
                    config=config
                )
                self.assertIsNotNone(model)
                
                # Verify model performance
                predictions = model.predict(self.test_data[self.sample_features])
                self.assertEqual(len(predictions), len(self.test_data))
                self.assertTrue(np.isfinite(predictions).all())
                
                # Test model persistence
                model_path = self.create_temp_file(suffix=".pkl")
                trainer.save_model(model, model_path)
                loaded_model = trainer.load_model(model_path)
                self.assertIsNotNone(loaded_model)
                
                # Verify loaded model predictions match
                new_preds = loaded_model.predict(self.test_data[self.sample_features])
                np.testing.assert_array_almost_equal(predictions, new_preds)

    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        evaluator = ModelEvaluator()
        
        # Train a simple model for testing
        model = GradientBoostingRegressor(n_estimators=50)
        X_train = self.train_data[self.sample_features]
        y_train = self.train_data["target"]
        model.fit(X_train, y_train)
        
        # Test different evaluation scenarios
        evaluation_scenarios = [
            {
                "name": "basic",
                "X": self.test_data[self.sample_features],
                "y": self.test_data["target"]
            },
            {
                "name": "train",
                "X": X_train,
                "y": y_train,
                "expected_better_mae": True
            }
        ]
        
        for scenario in evaluation_scenarios:
            with self.subTest(scenario=scenario["name"]):
                metrics = evaluator.evaluate_model(
                    model, 
                    scenario["X"],
                    scenario["y"]
                )
                
                # Verify metric structure
                required_metrics = ["mae", "mse", "rmse", "r2", "max_error"]
                for metric in required_metrics:
                    self.assertIn(metric, metrics)
                    self.assertIsInstance(metrics[metric], float)
                    self.assertTrue(np.isfinite(metrics[metric]))
                
                # Check relative performance expectations
                if scenario.get("expected_better_mae"):
                    self.assertLess(metrics["mae"], 0.1)
                
                # Test prediction intervals if supported
                try:
                    intervals = evaluator.get_prediction_intervals(
                        model,
                        scenario["X"],
                        confidence_level=0.95
                    )
                    self.assertEqual(len(intervals), len(scenario["X"]))
                    self.assertTrue(all(low <= high for low, high in intervals))
                except NotImplementedError:
                    pass  # Prediction intervals not supported for all models

    def test_error_handling(self):
        """Test error handling and validation."""
        trainer = ModelTrainer()
        
        # Test invalid feature names
        with self.assertRaises(ValueError):
            trainer.train_model(
                features=self.df[["invalid_feature"]],
                target=self.df["target"]
            )
        
        # Test invalid target values
        with self.assertRaises(ValueError):
            trainer.train_model(
                features=self.df[self.sample_features],
                target=pd.Series([np.nan] * len(self.df))
            )
        
        # Test mismatched dimensions
        with self.assertRaises(ValueError):
            trainer.train_model(
                features=self.df[self.sample_features],
                target=pd.Series([0, 1])  # Wrong length
            )
        
        # Test model prediction without training
        model = GradientBoostingRegressor()
        with self.assertRaises(NotFittedError):
            model.predict(self.df[self.sample_features])
        
        # Test invalid hyperparameters
        with self.assertRaises(ValueError):
            trainer.train_model(
                features=self.df[self.sample_features],
                target=self.df["target"],
                config={"n_estimators": -1}  # Invalid value
            )

    def test_model_updates(self):
        """Test model update functionality."""
        trainer = ModelTrainer()
        
        # Initial training
        model = trainer.train_model(
            features=self.train_data[self.sample_features],
            target=self.train_data["target"]
        )
        
        # Test online update with new data
        new_data = self.generate_test_data(periods=50)
        for feature in self.sample_features:
            new_data[feature] = np.random.normal(0, 1, len(new_data))
        new_data["target"] = new_data["close"].shift(-1) / new_data["close"] - 1
        new_data = new_data.dropna()
        
        update_result = trainer.update_online(
            X_new=new_data[self.sample_features],
            y_new=new_data["target"]
        )
        
        self.assertTrue(update_result["success"])
        self.assertIn("metrics", update_result)
        self.assertGreater(update_result["metrics"]["test_score"], 0)

    def _verify_feature_importance(self, importances: Dict[str, float]) -> bool:
        """Helper to verify feature importance values."""
        return all(isinstance(v, float) and 0 <= v <= 1 for v in importances.values())
