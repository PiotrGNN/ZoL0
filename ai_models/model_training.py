"""
Enhanced model training module with composite models and GPU acceleration.
"""
from datetime import datetime, timedelta
import os
import logging
import hashlib
import time
from typing import Dict, Union, List, Any, Optional, Tuple, Protocol, Type
from typing_extensions import runtime_checkable
import functools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def prepare_data_for_model(data: Union[pd.DataFrame, Dict]) -> np.ndarray:
    """
    Prepare data for model training or prediction.
    
    Args:
        data: Input data as DataFrame or dict
        
    Returns:
        np.ndarray: Preprocessed data ready for model
        
    Raises:
        ValueError: If data format is invalid
    """
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be DataFrame or dict")
        
    if data.empty:
        raise ValueError("Empty dataset provided")
        
    # Handle missing values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Scale numeric features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data.values

@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining required model interface"""
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Any:
        """Fit the model"""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        ...
    
    def score(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Score the model"""
        ...

class CompositeModel:
    """Composite model that combines multiple models"""
    
    def __init__(
        self,
        models: List[Union[nn.Module, ModelProtocol]],
        weights: Optional[List[float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.device = device
        
        if len(self.models) != len(self.weights):
            raise ValueError("Number of models must match number of weights")
            
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.trained = False

    def to(self, device: str) -> 'CompositeModel':
        """Move PyTorch models to specified device"""
        self.device = device
        for model in self.models:
            if isinstance(model, nn.Module):
                model.to(device)
        return self

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> 'CompositeModel':
        """Fit all models in the ensemble"""
        X = self._prepare_input(X)
        if y is not None:
            y = self._prepare_input(y)
        
        for model in self.models:
            if isinstance(model, nn.Module):
                model.train()
                optimizer = torch.optim.Adam(model.parameters())
                criterion = nn.MSELoss()
                
                for epoch in range(100):  # Basic training loop
                    optimizer.zero_grad()
                    y_pred = model(X)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()
            else:
                model.fit(self._to_numpy(X), self._to_numpy(y))
        
        self.trained = True
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Make predictions using weighted ensemble"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self._prepare_input(X)
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    pred = model(X)
                predictions.append(weight * self._to_numpy(pred))
            else:
                predictions.append(weight * model.predict(self._to_numpy(X)))
        
        return np.sum(predictions, axis=0)

    def _prepare_input(
        self,
        X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Prepare input data for models"""
        if isinstance(X, np.ndarray) and any(isinstance(m, nn.Module) for m in self.models):
            return torch.FloatTensor(X).to(self.device)
        return X

    def _to_numpy(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array"""
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return X

class ModelTraining:
    """Enhanced model training with GPU support and advanced features"""
    
    def __init__(
        self,
        model_dir: str = "saved_models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.logger = logging.getLogger(__name__)
        self.model_dir = model_dir
        self.device = device
        self._cache = {}
        self._cache_timeout = timedelta(minutes=30)
        self._last_cache_cleanup = datetime.now()
        
        os.makedirs(model_dir, exist_ok=True)

    def _clean_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        remove_outliers: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Clean data with advanced options"""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        if remove_outliers:
            # Remove outliers using IQR method
            Q1 = data[numeric_cols].quantile(0.25)
            Q3 = data[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                    (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            data = data[mask]
        
        return data if isinstance(data, pd.DataFrame) else data.values

    def _normalize_features(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        scaler: Optional[StandardScaler] = None
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], StandardScaler]:
        """Normalize features with optional scaler"""
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if scaler is None:
                scaler = StandardScaler()
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            else:
                data[numeric_cols] = scaler.transform(data[numeric_cols])
        else:
            if scaler is None:
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
            else:
                data = scaler.transform(data)
        
        return data, scaler

    def train_model(
        self,
        model: Union[nn.Module, ModelProtocol, CompositeModel],
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        batch_size: Optional[int] = None,
        epochs: int = 100,
        optimizer_cls: Optional[Type[torch.optim.Optimizer]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train model with advanced options and monitoring"""
        start_time = datetime.now()
        
        try:
            # Move PyTorch models to device
            if isinstance(model, (nn.Module, CompositeModel)):
                model = model.to(self.device)
                if isinstance(X, torch.Tensor):
                    X = X.to(self.device)
                if isinstance(y, torch.Tensor):
                    y = y.to(self.device)
            
            # Split data
            if validation_split > 0:
                split_idx = int(len(X) * (1 - validation_split))
                if isinstance(X, torch.Tensor):
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train = y[:split_idx] if y is not None else None
                    y_val = y[split_idx:] if y is not None else None
                else:
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train = y[:split_idx] if y is not None else None
                    y_val = y[split_idx:] if y is not None else None
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None
            
            if isinstance(model, nn.Module):
                metrics = self._train_pytorch_model(
                    model, X_train, y_train, X_val, y_val,
                    early_stopping, patience, batch_size, epochs,
                    optimizer_cls, optimizer_kwargs,
                    scheduler_cls, scheduler_kwargs
                )
            else:
                metrics = self._train_sklearn_model(
                    model, X_train, y_train, X_val, y_val
                )
            
            training_time = (datetime.now() - start_time).total_seconds()
            metrics["training_time"] = training_time
            
            return {
                "success": True,
                **metrics
            }
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _train_pytorch_model(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        early_stopping: bool,
        patience: int,
        batch_size: Optional[int],
        epochs: int,
        optimizer_cls: Optional[Type[torch.optim.Optimizer]],
        optimizer_kwargs: Optional[Dict[str, Any]],
        scheduler_cls: Optional[Type[torch.optim.lr_scheduler._LRScheduler]],
        scheduler_kwargs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Train PyTorch model with advanced options"""
        # Setup optimizer
        optimizer_cls = optimizer_cls or torch.optim.Adam
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        
        # Setup scheduler
        scheduler = None
        if scheduler_cls is not None:
            scheduler_kwargs = scheduler_kwargs or {}
            scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
        
        # Setup criterion
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            if batch_size:
                train_loss = self._train_batch(
                    model, X_train, y_train, optimizer, criterion, batch_size
                )
            else:
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
            
            train_losses.append(train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_val)
                    val_loss = criterion(y_val_pred, y_val).item()
                val_losses.append(val_loss)
                
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_state = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            # Restore best model state
                            model.load_state_dict(best_state)
                            break
            
            if scheduler is not None:
                scheduler.step()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_train_pred = model(X_train)
            final_train_loss = criterion(final_train_pred, y_train).item()
            
            metrics = {
                "final_train_loss": final_train_loss,
                "train_losses": train_losses
            }
            
            if X_val is not None and y_val is not None:
                final_val_pred = model(X_val)
                final_val_loss = criterion(final_val_pred, y_val).item()
                metrics.update({
                    "final_val_loss": final_val_loss,
                    "val_losses": val_losses,
                    "best_val_loss": best_val_loss
                })
        
        return metrics

    def _train_batch(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        batch_size: int
    ) -> float:
        """Train model in batches"""
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches

    def _train_sklearn_model(
        self,
        model: ModelProtocol,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Train scikit-learn compatible model"""
        model.fit(X_train, y_train)
        
        metrics = {
            "train_score": model.score(X_train, y_train)
        }
        
        if X_val is not None and y_val is not None:
            metrics["val_score"] = model.score(X_val, y_val)
            
        return metrics

    def save_model(
        self,
        model: Union[nn.Module, ModelProtocol, CompositeModel],
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model with metadata"""
        save_path = os.path.join(self.model_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        try:
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
            elif isinstance(model, CompositeModel):
                os.makedirs(os.path.join(save_path, "components"), exist_ok=True)
                for i, component in enumerate(model.models):
                    if isinstance(component, nn.Module):
                        torch.save(
                            component.state_dict(),
                            os.path.join(save_path, f"components/model_{i}.pt")
                        )
                    else:
                        import joblib
                        joblib.dump(
                            component,
                            os.path.join(save_path, f"components/model_{i}.joblib")
                        )
                # Save weights
                np.save(
                    os.path.join(save_path, "weights.npy"),
                    np.array(model.weights)
                )
            else:
                import joblib
                joblib.dump(model, os.path.join(save_path, "model.joblib"))
            
            # Save metadata
            if metadata:
                with open(os.path.join(save_path, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(
        self,
        name: str,
        model_class: Optional[Union[Type[nn.Module], Type[ModelProtocol]]] = None
    ) -> Tuple[Union[nn.Module, ModelProtocol, CompositeModel], Dict[str, Any]]:
        """Load model and metadata"""
        load_path = os.path.join(self.model_dir, name)
        
        if not os.path.exists(load_path):
            raise ValueError(f"Model {name} not found")
        
        try:
            # Load metadata
            metadata = {}
            metadata_path = os.path.join(load_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            
            # Check for composite model
            if os.path.exists(os.path.join(load_path, "components")):
                weights = np.load(os.path.join(load_path, "weights.npy")).tolist()
                models = []
                
                for i, file in enumerate(sorted(os.listdir(os.path.join(load_path, "components")))):
                    if file.endswith(".pt"):
                        if model_class is None:
                            raise ValueError("model_class required for PyTorch models")
                        model = model_class()
                        model.load_state_dict(torch.load(
                            os.path.join(load_path, "components", file)
                        ))
                        models.append(model)
                    else:
                        import joblib
                        model = joblib.load(
                            os.path.join(load_path, "components", file)
                        )
                        models.append(model)
                
                return CompositeModel(models, weights), metadata
            
            # Load single model
            if os.path.exists(os.path.join(load_path, "model.pt")):
                if model_class is None:
                    raise ValueError("model_class required for PyTorch models")
                model = model_class()
                model.load_state_dict(torch.load(os.path.join(load_path, "model.pt")))
            else:
                import joblib
                model = joblib.load(os.path.join(load_path, "model.joblib"))
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create example composite model
    class SimpleNet(nn.Module):
        def __init__(self, input_size: int = 10, hidden_size: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            
        def forward(self, x):
            return self.net(x)
    
    from sklearn.ensemble import RandomForestRegressor
    
    # Create composite model
    models = [
        SimpleNet(),
        RandomForestRegressor(),
        SimpleNet(hidden_size=32)
    ]
    
    composite_model = CompositeModel(
        models,
        weights=[0.4, 0.3, 0.3]
    )
    
    # Generate example data
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    
    # Create trainer
    trainer = ModelTraining()
    
    # Train model
    results = trainer.train_model(
        composite_model,
        torch.FloatTensor(X),
        torch.FloatTensor(y),
        validation_split=0.2,
        batch_size=32
    )
    
    print("Training results:", results)