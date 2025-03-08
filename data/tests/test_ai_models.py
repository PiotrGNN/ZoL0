"""
test_ai_models.py
-----------------
Testy jednostkowe dla modułu AIStrategyGenerator oraz powiązanych strategii.
Weryfikujemy poprawność wyboru cech, tuningu hiperparametrów, budowy ensemblu,
oceny strategii oraz generowania strategii.
"""

import logging
import unittest
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from data.strategies.AI_strategy_generator import AIStrategyGenerator
from data.data.historical_data import HistoricalDataManager

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TestAIStrategyGenerator(unittest.TestCase):
    """Testy jednostkowe dla klasy AIStrategyGenerator."""

    def setUp(self) -> None:
        """Przygotowanie danych testowych przed każdym testem."""
        try:
            np.random.seed(42)
            dates = pd.date_range(start="2022-01-01", periods=100, freq="B")
            data = {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(5, 2, 100),
                "feature3": np.random.normal(10, 3, 100),
                "target": np.random.normal(0, 1, 100),
            }
            self.df = pd.DataFrame(data, index=dates)
            self.ai_gen = AIStrategyGenerator(self.df, target="target")
        except Exception as e:
            self.fail(f"Nie udało się poprawnie przygotować danych testowych: {e}")

    def test_feature_selection(self) -> None:
        """Test wyboru cech."""
        try:
            selected_features = self.ai_gen.feature_selection(k=2)
            expected_features = min(2, self.df.shape[1] - 1)
            self.assertEqual(
                selected_features.shape[1],
                expected_features,
                f"Liczba wybranych cech powinna być równa {expected_features}.",
            )
        except Exception as e:
            self.fail(f"Niepowodzenie testu selekcji cech: {e}")

    def test_hyperparameter_tuning(self) -> None:
        """Test tuningu hiperparametrów modelu GradientBoostingRegressor."""
        param_grid = {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 4],
        }
        try:
            model = self.ai_gen.hyperparameter_tuning(GradientBoostingRegressor(), param_grid, cv=3)
            self.assertIsNotNone(model, "Model po tuningu nie powinien być None.")
            self.assertTrue(hasattr(model, "predict"), "Model powinien mieć metodę predict.")
            self.assertIsInstance(self.ai_gen.best_params, dict, "Najlepsze hiperparametry powinny być słownikiem.")
            self.assertGreater(len(self.ai_gen.best_params), 0, "Hiperparametry powinny zawierać co najmniej jeden wpis.")
        except Exception as e:
            self.fail(f"Niepowodzenie testu tuningu hiperparametrów: {e}")

    def test_build_ensemble(self) -> None:
        """Test budowy ensemblu modeli."""
        try:
            ensemble = self.ai_gen.build_ensemble()
            self.assertIsNotNone(ensemble, "Ensemble modeli nie powinien być None.")
            self.assertTrue(hasattr(ensemble, "predict"), "Ensemble powinien mieć metodę predict.")
        except Exception as e:
            self.fail(f"Niepowodzenie testu budowy ensemble: {e}")

    def test_evaluate_strategy(self) -> None:
        """Test oceny strategii."""
        try:
            self.ai_gen.build_ensemble()
            evaluation_report = self.ai_gen.evaluate_strategy()
            self.assertIn("MSE", evaluation_report, "Raport oceny strategii powinien zawierać MSE.")
            self.assertGreaterEqual(evaluation_report["MSE"], 0, "MSE powinno być nieujemne.")
        except Exception as e:
            self.fail(f"Niepowodzenie testu oceny strategii: {e}")

    def test_generate_strategy(self) -> None:
        """Test pełnego pipeline'u generowania strategii."""
        try:
            report = self.ai_gen.generate_strategy()
            self.assertIsInstance(report, dict, "Raport strategii powinien być słownikiem.")
            self.assertIn("best_hyperparameters", report, "Raport strategii powinien zawierać best_hyperparameters.")
            self.assertIn("evaluation", report, "Raport strategii powinien zawierać evaluation.")
            self.assertGreaterEqual(report["evaluation"]["MSE"], 0, "MSE w ocenie strategii powinno być nieujemne.")
        except Exception as e:
            self.fail(f"Niepowodzenie testu generowania strategii: {e}")


if __name__ == "__main__":
    unittest.main()
