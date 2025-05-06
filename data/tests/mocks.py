"""Mock implementations for testing."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

class MockModel:
    """Mock AI model for testing."""
    def __init__(self):
        self.trained = False
        self.predictions = []
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate mock predictions."""
        n_samples = len(data)
        predictions = np.random.uniform(0, 1, n_samples)
        self.predictions.append(predictions)
        return predictions
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Mock training."""
        self.trained = True

class MockAnomalyDetector:
    """Mock anomaly detector for testing."""
    def detect(self, data: pd.DataFrame) -> List[bool]:
        """Return mock anomaly flags."""
        return [False] * len(data)

class MockLeverageOptimizer:
    """Mock leverage optimizer for testing."""
    def optimize(self, data: pd.DataFrame, risk_tolerance: float = 0.02) -> float:
        """Return mock optimized leverage."""
        return min(3.0, 1.0 / risk_tolerance)

class MockStrategyGenerator:
    """Mock strategy generator for testing."""
    def generate(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock trading strategy."""
        return {
            "entry_rules": ["price > ma_200", "rsi < 30"],
            "exit_rules": ["price < ma_200", "rsi > 70"],
            "risk_params": {
                "stop_loss": 0.02,
                "take_profit": 0.04
            }
        }

class MockHyperparameterTuner:
    """Mock hyperparameter tuner for testing."""
    def tune(self, model, param_grid: Dict[str, List[Any]], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Return mock optimal parameters."""
        return {k: v[0] for k, v in param_grid.items()}