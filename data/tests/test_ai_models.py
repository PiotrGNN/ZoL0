"""
test_ai_models.py
-----------------
Testy jednostkowe dla modułu AI_strategy_generator.py oraz powiązanych strategii.
Testy weryfikują poprawność generowania strategii, spójność danych treningowych oraz działanie metod predykcyjnych.
"""

import logging
import unittest

import numpy as np
import pandas as pd

# Zakładamy, że moduł AI_strategy_generator.py znajduje się w folderze data/strategies
from data.strategies.AI_strategy_generator import AIStrategyGenerator


class TestAIStrategyGenerator(unittest.TestCase):
    def setUp(self):
        # Generujemy przykładowe dane historyczne
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=100, freq="B")
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(5, 2, 100),
                "feature3": np.random.normal(10, 3, 100),
                "target": np.random.normal(0, 1, 100),
            },
            index=dates,
        )
        self.data = data
        self.ai_gen = AIStrategyGenerator(data, target="target")

    def test_feature_selection(self):
        # Testujemy, czy funkcja feature_selection poprawnie wybiera cechy
        selected_features = self.ai_gen.feature_selection(k=2)
        self.assertEqual(selected_features.shape[1], 2, "Liczba wybranych cech powinna być równa 2.")

    def test_hyperparameter_tuning(self):
        # Testujemy tuning hiperparametrów dla przykładowego modelu
        from sklearn.ensemble import GradientBoostingRegressor

        param_grid = {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 4],
        }
        model = self.ai_gen.hyperparameter_tuning(GradientBoostingRegressor, param_grid, cv=3)
        self.assertIsNotNone(model, "Model po tuningu nie powinien być None.")

    def test_build_ensemble(self):
        # Testujemy budowę ensemblu
        ensemble = self.ai_gen.build_ensemble()
        self.assertTrue(hasattr(ensemble, "predict"), "Ensemble powinien mieć metodę predict.")

    def test_evaluate_strategy(self):
        # Testujemy ocenę strategii
        evaluation_report = self.ai_gen.evaluate_strategy()
        self.assertIn("MSE", evaluation_report, "Raport oceny strategii powinien zawierać MSE.")

    def test_generate_strategy(self):
        # Testujemy kompletny pipeline generowania strategii
        report = self.ai_gen.generate_strategy()
        self.assertIn(
            "best_hyperparameters",
            report,
            "Raport strategii powinien zawierać best_hyperparameters.",
        )
        self.assertIn("evaluation", report, "Raport strategii powinien zawierać evaluation.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
