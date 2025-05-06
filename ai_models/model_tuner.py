"""
Advanced model hyperparameter tuning with Optuna and PyTorch support.
"""

import logging
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
from datetime import datetime
from pathlib import Path

class ModelTuner:
    """Advanced model hyperparameter tuning with multiple backend support"""
    
    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, Dict[str, Any]],
        metric: str = "mse",
        n_trials: int = 50,
        cv_splits: int = 5,
        random_state: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        self.model_class = model_class
        self.param_space = param_space
        self.metric = metric
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.device = device
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        
        self.study = None
        self.best_params = None
        self.best_score = None
        self.best_model = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Store history
        self.history = []
        self.trial_info = {}

    def _create_model(self, trial: optuna.Trial) -> Any:
        """Create model with trial parameters"""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config["type"]
            
            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_type == "float":
                if param_config.get("log", False):
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step")
                    )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        # Handle PyTorch models
        if issubclass(self.model_class, nn.Module):
            model = self.model_class(**params).to(self.device)
        else:
            model = self.model_class(**params)
            
        return model

    def _calculate_metric(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Calculate specified metric"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
            
        if self.metric == "mse":
            return mean_squared_error(y_true, y_pred)
        elif self.metric == "mae":
            return mean_absolute_error(y_true, y_pred)
        elif self.metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == "sharpe":
            returns = pd.Series(y_pred) / pd.Series(y_true) - 1
            return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def objective(
        self,
        trial: optuna.Trial,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Optimization objective function"""
        model = self._create_model(trial)
        cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        scores = []
        
        # Convert to torch tensors if using PyTorch
        if issubclass(self.model_class, nn.Module):
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X).to(self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.FloatTensor(y).to(self.device)
        
        for train_idx, val_idx in cv.split(X):
            try:
                if issubclass(self.model_class, nn.Module):
                    score = self._train_pytorch_model(
                        model, X, y, train_idx, val_idx, trial
                    )
                else:
                    score = self._train_sklearn_model(
                        model, X, y, train_idx, val_idx
                    )
                scores.append(score)
            except Exception as e:
                self.logger.error(f"Error in trial {trial.number}: {str(e)}")
                raise optuna.TrialPruned()
        
        mean_score = np.mean(scores)
        
        # Store trial information
        self.trial_info[trial.number] = {
            "params": trial.params,
            "value": mean_score,
            "scores": scores
        }
        
        return mean_score

    def _train_pytorch_model(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        trial: optuna.Trial
    ) -> float:
        """Train PyTorch model for one fold"""
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        n_epochs = 100
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
            # Report intermediate value for pruning
            trial.report(val_loss.item(), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return val_loss.item()

    def _train_sklearn_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray
    ) -> float:
        """Train scikit-learn model for one fold"""
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return self._calculate_metric(y_val, y_pred)

    def tune(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using Optuna
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional arguments passed to optuna.create_study
            
        Returns:
            Tuple of (best parameters, best score)
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            **kwargs
        )
        
        func = lambda trial: self.objective(trial, X, y)
        study.optimize(func, n_trials=self.n_trials)
        
        self.study = study
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Create and store best model
        self.best_model = self._create_model(study.best_trial)
        
        return self.best_params, self.best_score

    def get_best_model(self) -> Any:
        """Get the best performing model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        return self.best_model

    def save_study(self, path: str) -> None:
        """Save study results and best model"""
        if self.study is None:
            raise ValueError("No study to save")
            
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save study results
        study_info = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study.trials),
            "study_name": self.study_name,
            "metric": self.metric,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path / "study_info.json", "w") as f:
            import json
            json.dump(study_info, f, indent=4)
        
        # Save best model if it's a sklearn model
        if not issubclass(self.model_class, nn.Module):
            import joblib
            joblib.dump(self.best_model, path / "best_model.joblib")
        else:
            # Save PyTorch model
            torch.save(self.best_model.state_dict(), path / "best_model.pt")

    def load_study(self, path: str) -> None:
        """Load saved study results and best model"""
        path = Path(path)
        
        # Load study info
        with open(path / "study_info.json", "r") as f:
            import json
            study_info = json.load(f)
            
        self.best_params = study_info["best_params"]
        self.best_score = study_info["best_score"]
        
        # Load best model
        if issubclass(self.model_class, nn.Module):
            self.best_model = self._create_model(optuna.trial.FixedTrial(self.best_params))
            self.best_model.load_state_dict(torch.load(path / "best_model.pt"))
        else:
            import joblib
            self.best_model = joblib.load(path / "best_model.joblib")

    def plot_optimization_history(self) -> None:
        """Plot optimization history using Optuna's visualization"""
        if self.study is None:
            raise ValueError("No study to plot")
        
        try:
            from optuna.visualization import plot_optimization_history
            plot_optimization_history(self.study)
        except ImportError:
            self.logger.warning("optuna.visualization not available")

    def get_param_importances(self) -> Dict[str, float]:
        """Get parameter importances"""
        if self.study is None:
            raise ValueError("No study to analyze")
            
        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception as e:
            self.logger.error(f"Could not calculate parameter importances: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Example with PyTorch model
    class SimpleNet(nn.Module):
        def __init__(self, hidden_size: int = 64, num_layers: int = 2):
            super().__init__()
            layers = []
            input_size = 10  # Example input size
            
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                input_size = hidden_size
                
            layers.append(nn.Linear(hidden_size, 1))
            self.net = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.net(x)
    
    # Define parameter space
    param_space = {
        "hidden_size": {
            "type": "int",
            "low": 32,
            "high": 128,
            "step": 32
        },
        "num_layers": {
            "type": "int",
            "low": 1,
            "high": 3
        }
    }
    
    # Create tuner
    tuner = ModelTuner(
        model_class=SimpleNet,
        param_space=param_space,
        metric="mse",
        n_trials=20
    )
    
    # Generate example data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    
    # Tune model
    best_params, best_score = tuner.tune(X, y)
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
