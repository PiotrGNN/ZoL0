"""
AI_strategy_generator.py
----------------------
Module for generating trading strategies using artificial intelligence.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class AIStrategyGenerator:
    """Class for generating AI-based trading strategies."""

    def __init__(self, data: pd.DataFrame, target: str):
        """
        Initialize the AI Strategy Generator.

        Parameters:
            data (pd.DataFrame): Input data with features and target
            target (str): Name of target column
        """
        if target not in data.columns:
            raise KeyError(f"Target column '{target}' not found in data")
            
        self.data = data
        self.target = target
        self.features = [col for col in data.columns if col != target]
        self.models = []
        self.best_model = None

    def select_features(self, method: str = "correlation", threshold: float = 0.1) -> List[str]:
        """
        Select features using specified method.

        Parameters:
            method (str): Feature selection method ("correlation", "mutual_info", "recursive")
            threshold (float): Selection threshold

        Returns:
            List[str]: Selected feature names
        """
        if method not in ["correlation", "mutual_info", "recursive"]:
            raise ValueError(f"Invalid feature selection method: {method}")

        X = self.data[self.features]
        y = self.data[self.target]

        if method == "correlation":
            correlations = abs(X.corrwith(y))
            selected = correlations[correlations > threshold].index.tolist()
        
        elif method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_regression, k='all')
            selector.fit(X, y)
            scores = selector.scores_
            selected = [self.features[i] for i in range(len(scores)) if scores[i] > threshold]
        
        elif method == "recursive":
            if self.best_model is None:
                from sklearn.linear_model import LinearRegression
                self.best_model = LinearRegression()
            rfe = RFE(estimator=self.best_model, n_features_to_select=int(len(self.features) * 0.5))
            rfe.fit(X, y)
            selected = [self.features[i] for i in range(len(rfe.support_)) if rfe.support_[i]]

        return selected

    def tune_hyperparameters(self, model_class: Any, param_grid: Dict[str, List[Any]], cv: Any = 5) -> Dict[str, Any]:
        """
        Tune model hyperparameters using grid search.

        Parameters:
            model_class: Scikit-learn model class
            param_grid (Dict[str, List[Any]]): Grid of parameters to search
            cv: Cross-validation splitting strategy

        Returns:
            Dict[str, Any]: Best parameters found
        """
        if not param_grid or not all(isinstance(v, list) for v in param_grid.values()):
            raise ValueError("Invalid parameter grid format")

        X = self.data[self.features]
        y = self.data[self.target]

        grid_search = GridSearchCV(
            model_class(),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        return grid_search.best_params_

    def build_ensemble(self, n_models: int = 3, features: Optional[List[str]] = None) -> List[Any]:
        """
        Build an ensemble of models.

        Parameters:
            n_models (int): Number of models in ensemble
            features (List[str], optional): Features to use for training

        Returns:
            List[Any]: List of trained models
        """
        if n_models < 1:
            raise ValueError("Number of models must be positive")

        features = features or self.features
        X = self.data[features]
        y = self.data[self.target]

        from sklearn.ensemble import GradientBoostingRegressor
        models = []
        for _ in range(n_models):
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=np.random.randint(1000)
            )
            model.fit(X, y)
            models.append(model)

        return models

    def evaluate_strategy(self, predictions: np.ndarray, actual_values: np.ndarray) -> Dict[str, float]:
        """
        Evaluate strategy performance.

        Parameters:
            predictions (np.ndarray): Model predictions
            actual_values (np.ndarray): Actual values

        Returns:
            Dict[str, float]: Performance metrics
        """
        # Calculate basic regression metrics
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)

        # Calculate Sharpe ratio using predictions as returns
        returns = pd.Series(predictions).pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "sharpe_ratio": sharpe_ratio
        }

    def generate_strategy(self, features: List[str], ensemble_size: int = 3, cv_splits: int = 3) -> Dict[str, Any]:
        """
        Generate complete trading strategy.

        Parameters:
            features (List[str]): Features to use
            ensemble_size (int): Number of models in ensemble
            cv_splits (int): Number of cross-validation splits

        Returns:
            Dict[str, Any]: Generated strategy configuration
        """
        # Build model ensemble
        models = self.build_ensemble(n_models=ensemble_size, features=features)
        self.models = models

        # Generate predictions
        X = self.data[features]
        y = self.data[self.target]
        
        predictions = np.mean([model.predict(X) for model in models], axis=0)
        
        # Evaluate performance
        metrics = self.evaluate_strategy(predictions, y)

        return {
            "models": models,
            "selected_features": features,
            "performance_metrics": metrics,
            "parameters": {
                "ensemble_size": ensemble_size,
                "cv_splits": cv_splits
            }
        }

# -------------------- Example usage --------------------
if __name__ == "__main__":
    try:
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=500, freq="B")
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 500),
                "feature2": np.random.normal(5, 2, 500),
                "feature3": np.random.normal(10, 3, 500),
                "target": np.random.normal(0, 1, 500),
            },
            index=dates,
        )

        ai_generator = AIStrategyGenerator(data=data, target="target")
        selected_features = ai_generator.select_features(method="correlation", threshold=0.1)
        print(f"Selected features: {selected_features}")

        best_params = ai_generator.tune_hyperparameters(
            model_class=GradientBoostingRegressor,
            param_grid={"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
            cv=3
        )
        print(f"Best hyperparameters: {best_params}")

        strategy = ai_generator.generate_strategy(features=selected_features, ensemble_size=3, cv_splits=3)
        print(f"Generated strategy: {strategy}")

    except Exception as e:
        logger.error("Error in AI_strategy_generator.py: %s", e)
        raise