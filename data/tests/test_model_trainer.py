"""
test_model_trainer.py
-------------------
Tests for ModelTrainer functionality including:
- Model initialization and validation
- Training and evaluation
- Model saving and loading
- Online learning capabilities
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

try:
    from data.ai_models.model_training import ModelTrainer, prepare_data_for_model
except ImportError:
    from ..ai_models.model_training import ModelTrainer, prepare_data_for_model

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""

    def setUp(self) -> None:
        """Set up test environment before each test."""
        np.random.seed(42)
        self.X = np.random.rand(100, 3)
        self.y = np.random.rand(100)
        self.model = RandomForestRegressor(n_estimators=10)
        self.trainer = ModelTrainer(
            model=self.model,
            model_name="test_model",
            saved_model_dir="test_models",
            online_learning=True
        )

    def tearDown(self) -> None:
        """Clean up test environment after each test."""
        # Remove test model files
        if os.path.exists("test_models"):
            for file in os.listdir("test_models"):
                os.remove(os.path.join("test_models", file))
            os.rmdir("test_models")

    def test_initialization(self) -> None:
        """Test ModelTrainer initialization."""
        self.assertIsInstance(self.trainer.model, RandomForestRegressor)
        self.assertEqual(self.trainer.model_name, "test_model")
        self.assertTrue(self.trainer.online_learning)
        self.assertFalse(self.trainer.use_gpu)  # Should be False by default

        # Test invalid model initialization
        class InvalidModel:
            pass

        with self.assertRaises(ValueError):
            ModelTrainer(InvalidModel(), "invalid_model")

    def test_data_validation(self) -> None:
        """Test input data validation."""
        # Test valid data
        self.trainer.validate_data(self.X, self.y)

        # Test invalid data
        with self.assertRaises(ValueError):
            self.trainer.validate_data(None, self.y)

        with self.assertRaises(ValueError):
            self.trainer.validate_data(self.X, self.y[:50])  # Length mismatch

        with self.assertRaises(ValueError):
            self.trainer.validate_data([], self.y)  # Empty input

    def test_training(self) -> None:
        """Test model training functionality."""
        # Test basic training
        result = self.trainer.train(self.X, self.y)
        self.assertTrue(result["success"])
        self.assertFalse(result["skipped"])
        self.assertIn("metrics", result)

        # Test training with same data (should skip)
        result = self.trainer.train(self.X, self.y, force_train=False)
        self.assertTrue(result["success"])
        self.assertTrue(result["skipped"])

        # Force retrain
        result = self.trainer.train(self.X, self.y, force_train=True)
        self.assertTrue(result["success"])
        self.assertFalse(result["skipped"])

    def test_evaluation(self) -> None:
        """Test model evaluation."""
        # Train model first
        self.trainer.train(self.X, self.y)

        # Test evaluation
        metrics = self.trainer.evaluate(self.X, self.y)
        required_metrics = ["test_score", "mae", "mse", "rmse"]
        self.assertTrue(all(metric in metrics for metric in required_metrics))
        self.assertGreaterEqual(metrics["test_score"], 0.0)
        self.assertGreaterEqual(metrics["rmse"], 0.0)

        # Test evaluation with untrained model
        trainer = ModelTrainer(LinearRegression(), "untrained_model")
        with self.assertRaises(NotFittedError):
            trainer.evaluate(self.X, self.y)

    def test_online_learning(self) -> None:
        """Test online learning capabilities."""
        # Initial training
        self.trainer.train(self.X[:50], self.y[:50])

        # Online update
        result = self.trainer.update_online(self.X[50:], self.y[50:])
        self.assertTrue(result["success"])
        self.assertIn("metrics", result)

        # Test with online learning disabled
        trainer = ModelTrainer(
            self.model,
            "no_online_model",
            online_learning=False
        )
        with self.assertRaises(ValueError):
            trainer.update_online(self.X, self.y)

    def test_model_saving(self) -> None:
        """Test model saving functionality."""
        # Train and save model
        self.trainer.train(self.X, self.y)
        save_path = self.trainer.save_model()
        
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(save_path + '.meta'))

        # Test saving untrained model
        untrained_trainer = ModelTrainer(LinearRegression(), "untrained")
        with self.assertRaises(ValueError):
            untrained_trainer.save_model()

    def test_feature_importance(self) -> None:
        """Test feature importance extraction."""
        # Test with RandomForestRegressor (has feature_importances_)
        self.trainer.train(self.X, self.y)
        importance = self.trainer.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), self.X.shape[1])

        # Test with LinearRegression (has coef_)
        linear_trainer = ModelTrainer(LinearRegression(), "linear_model")
        linear_trainer.train(self.X, self.y)
        importance = linear_trainer.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), self.X.shape[1])

    def test_data_preparation(self) -> None:
        """Test data preparation utility."""
        # Test with different input types
        df = pd.DataFrame(self.X)
        data_list = self.X.tolist()

        # DataFrame input
        prepared = prepare_data_for_model(df)
        self.assertIsInstance(prepared, np.ndarray)
        self.assertEqual(prepared.shape, self.X.shape)

        # List input
        prepared = prepare_data_for_model(data_list)
        self.assertIsInstance(prepared, np.ndarray)
        self.assertEqual(prepared.shape, self.X.shape)

        # Already numpy array
        prepared = prepare_data_for_model(self.X)
        self.assertIsInstance(prepared, np.ndarray)
        self.assertEqual(prepared.shape, self.X.shape)

        # Test validation
        with self.assertRaises(ValueError):
            prepare_data_for_model(self.X, features_count=10)  # Wrong feature count

if __name__ == "__main__":
    unittest.main()