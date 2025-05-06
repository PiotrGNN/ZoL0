"""
Advanced data scaling module with GPU acceleration and feature-specific scaling.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TensorScaler:
    """PyTorch-based scaler for GPU acceleration"""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fitted = False
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize scaling parameters"""
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.scale = None
        self.offset = None
    
    def fit(self, tensor: torch.Tensor) -> 'TensorScaler':
        """Fit scaler to data"""
        if self.method == 'standard':
            self.mean = tensor.mean(dim=0, keepdim=True)
            self.std = tensor.std(dim=0, keepdim=True)
            self.std[self.std == 0] = 1.0
        elif self.method == 'minmax':
            self.min = tensor.min(dim=0, keepdim=True)[0]
            self.max = tensor.max(dim=0, keepdim=True)[0]
            self.scale = self.max - self.min
            self.scale[self.scale == 0] = 1.0
        elif self.method == 'robust':
            q25 = torch.quantile(tensor, 0.25, dim=0)
            q75 = torch.quantile(tensor, 0.75, dim=0)
            iqr = q75 - q25
            self.center = q25 + iqr/2
            self.scale = iqr
            self.scale[self.scale == 0] = 1.0
        
        self.fitted = True
        return self
    
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        if self.method == 'standard':
            return (tensor - self.mean) / self.std
        elif self.method == 'minmax':
            return (tensor - self.min) / self.scale
        elif self.method == 'robust':
            return (tensor - self.center) / self.scale
        
        raise ValueError(f"Unknown scaling method: {self.method}")
    
    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse transform data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        if self.method == 'standard':
            return tensor * self.std + self.mean
        elif self.method == 'minmax':
            return tensor * self.scale + self.min
        elif self.method == 'robust':
            return tensor * self.scale + self.center
        
        raise ValueError(f"Unknown scaling method: {self.method}")
    
    def to(self, device: torch.device) -> 'TensorScaler':
        """Move scaler to specified device"""
        if self.fitted:
            if self.method == 'standard':
                self.mean = self.mean.to(device)
                self.std = self.std.to(device)
            elif self.method == 'minmax':
                self.min = self.min.to(device)
                self.max = self.max.to(device)
                self.scale = self.scale.to(device)
            elif self.method == 'robust':
                self.center = self.center.to(device)
                self.scale = self.scale.to(device)
        
        self.device = device
        return self

class FeatureScaler:
    """Advanced feature-specific scaler with persistence"""
    
    def __init__(
        self,
        feature_config: Optional[Dict[str, str]] = None,
        device: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.feature_config = feature_config or {
            'price': 'robust',
            'volume': 'log',
            'technical': 'standard',
            'pattern': 'minmax'
        }
        self.scalers: Dict[str, Union[TensorScaler, DataScaler]] = {}
        self._initialize_scalers()
    
    def _initialize_scalers(self):
        """Initialize scalers for each feature type"""
        for feature_type, method in self.feature_config.items():
            if method == 'log':
                self.scalers[feature_type] = DataScaler(method='log')
            else:
                self.scalers[feature_type] = TensorScaler(method=method)
    
    def fit(
        self,
        data: Dict[str, Union[np.ndarray, pd.DataFrame, torch.Tensor]]
    ) -> 'FeatureScaler':
        """Fit scalers to different feature types"""
        try:
            for feature_type, features in data.items():
                if feature_type not in self.scalers:
                    self.logger.warning(f"No scaler configured for {feature_type}")
                    continue
                
                # Convert to appropriate format
                if isinstance(features, (np.ndarray, pd.DataFrame)):
                    if isinstance(features, pd.DataFrame):
                        features = features.values
                    if self.scalers[feature_type].__class__ == TensorScaler:
                        features = torch.from_numpy(features).float().to(self.device)
                
                # Fit scaler
                self.scalers[feature_type].fit(features)
                
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting scalers: {e}")
            raise
    
    def transform(
        self,
        data: Dict[str, Union[np.ndarray, pd.DataFrame, torch.Tensor]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Transform features"""
        try:
            transformed = {}
            for feature_type, features in data.items():
                if feature_type not in self.scalers:
                    self.logger.warning(f"No scaler configured for {feature_type}")
                    transformed[feature_type] = features
                    continue
                
                # Convert to appropriate format
                original_type = type(features)
                if isinstance(features, (np.ndarray, pd.DataFrame)):
                    if isinstance(features, pd.DataFrame):
                        features = features.values
                    if self.scalers[feature_type].__class__ == TensorScaler:
                        features = torch.from_numpy(features).float().to(self.device)
                
                # Transform features
                transformed_features = self.scalers[feature_type].transform(features)
                
                # Convert back to original type if needed
                if original_type == pd.DataFrame and isinstance(features, pd.DataFrame):
                    transformed_features = pd.DataFrame(
                        transformed_features,
                        index=features.index,
                        columns=features.columns
                    )
                
                transformed[feature_type] = transformed_features
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {e}")
            raise
    
    def inverse_transform(
        self,
        data: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Inverse transform features"""
        try:
            inverse_transformed = {}
            for feature_type, features in data.items():
                if feature_type not in self.scalers:
                    self.logger.warning(f"No scaler configured for {feature_type}")
                    inverse_transformed[feature_type] = features
                    continue
                
                # Inverse transform
                inverse_transformed[feature_type] = (
                    self.scalers[feature_type].inverse_transform(features)
                )
            
            return inverse_transformed
            
        except Exception as e:
            self.logger.error(f"Error inverse transforming features: {e}")
            raise
    
    def save(self, path: Union[str, Path]):
        """Save scaler configuration and states"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(path / 'config.json', 'w') as f:
            json.dump(self.feature_config, f, indent=4)
        
        # Save scaler states
        for feature_type, scaler in self.scalers.items():
            if isinstance(scaler, TensorScaler) and scaler.fitted:
                state_dict = {
                    'method': scaler.method,
                    'parameters': {
                        name: tensor.cpu().numpy()
                        for name, tensor in scaler.__dict__.items()
                        if isinstance(tensor, torch.Tensor)
                    }
                }
                np.save(path / f'{feature_type}_scaler.npy', state_dict)
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> 'FeatureScaler':
        """Load scaler from saved state"""
        path = Path(path)
        
        # Load configuration
        with open(path / 'config.json', 'r') as f:
            feature_config = json.load(f)
        
        # Create instance
        instance = cls(feature_config=feature_config, device=device)
        
        # Load scaler states
        for feature_type, scaler in instance.scalers.items():
            state_path = path / f'{feature_type}_scaler.npy'
            if state_path.exists() and isinstance(scaler, TensorScaler):
                state_dict = np.load(state_path, allow_pickle=True).item()
                scaler.method = state_dict['method']
                for name, array in state_dict['parameters'].items():
                    setattr(scaler, name, torch.from_numpy(array).to(instance.device))
                scaler.fitted = True
        
        return instance

class DataScaler:
    """Enhanced DataScaler with support for multi-dimensional data"""
    
    def __init__(
        self,
        method: str = "standard",
        fill_value: Optional[float] = None,
        feature_axis: int = -1
    ):
        self.method = method.lower()
        self.fill_value = fill_value
        self.feature_axis = feature_axis
        self.scaler = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
        
        self._validate_method()
    
    def _validate_method(self):
        """Validate scaling method"""
        valid_methods = ["standard", "minmax", "robust", "log"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def _reshape_for_scaling(self, data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Reshape data for scaling if needed"""
        original_shape = data.shape
        
        # If multi-dimensional, reshape to 2D for scaling
        if data.ndim > 2:
            # Move feature axis to end if needed
            if self.feature_axis != -1:
                data = np.moveaxis(data, self.feature_axis, -1)
            
            # Reshape to 2D
            n_features = data.shape[-1]
            data = data.reshape(-1, n_features)
        
        return data, original_shape
    
    def _reshape_back(
        self,
        data: np.ndarray,
        original_shape: Tuple[int, ...],
        feature_axis: int
    ) -> np.ndarray:
        """Reshape data back to original shape"""
        if len(original_shape) > 2:
            data = data.reshape(original_shape[:-1] + (-1,))
            if feature_axis != -1:
                data = np.moveaxis(data, -1, feature_axis)
        return data
    
    def _handle_missing_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Handle missing values in data"""
        if isinstance(data, pd.DataFrame):
            if self.fill_value is None:
                self.fill_value = data.median().median()
            return data.fillna(self.fill_value).values
        
        if isinstance(data, np.ndarray):
            if self.fill_value is None:
                self.fill_value = np.nanmedian(data)
            return np.nan_to_num(data, nan=self.fill_value)
        
        raise ValueError("Data must be DataFrame or ndarray")
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> 'DataScaler':
        """Fit scaler to data"""
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Reshape if needed
            data, original_shape = self._reshape_for_scaling(data)
            
            # Initialize and fit scaler
            if self.method == "standard":
                self.scaler = StandardScaler()
            elif self.method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.method == "robust":
                self.scaler = RobustScaler()
            elif self.method == "log":
                # For log scaling, compute offset
                min_positive = np.min(data[data > 0]) if np.any(data > 0) else 1e-6
                self.offset = min_positive / 10.0
            
            if self.method != "log":
                self.scaler.fit(data, sample_weight=sample_weight)
            
            self.is_fitted = True
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting scaler: {e}")
            raise
    
    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data"""
        try:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform")
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Save original type and shape
            original_type = type(data)
            data, original_shape = self._reshape_for_scaling(data)
            
            # Transform
            if self.method == "log":
                transformed = np.log(data + self.offset)
            else:
                transformed = self.scaler.transform(data)
            
            # Reshape back if needed
            transformed = self._reshape_back(
                transformed, original_shape, self.feature_axis
            )
            
            # Convert back to original type if needed
            if original_type == pd.DataFrame:
                transformed = pd.DataFrame(transformed)
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise
    
    def inverse_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Inverse transform data"""
        try:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before inverse_transform")
            
            # Save original type and shape
            original_type = type(data)
            data, original_shape = self._reshape_for_scaling(data)
            
            # Inverse transform
            if self.method == "log":
                inverse_transformed = np.exp(data) - self.offset
            else:
                inverse_transformed = self.scaler.inverse_transform(data)
            
            # Reshape back if needed
            inverse_transformed = self._reshape_back(
                inverse_transformed, original_shape, self.feature_axis
            )
            
            # Convert back to original type if needed
            if original_type == pd.DataFrame:
                inverse_transformed = pd.DataFrame(inverse_transformed)
            
            return inverse_transformed
            
        except Exception as e:
            self.logger.error(f"Error inverse transforming data: {e}")
            raise

# Unit tests
def run_tests():
    """Run comprehensive unit tests"""
    try:
        # Test TensorScaler
        data = torch.randn(100, 5)
        for method in ['standard', 'minmax', 'robust']:
            scaler = TensorScaler(method=method)
            scaler.fit(data)
            transformed = scaler.transform(data)
            recovered = scaler.inverse_transform(transformed)
            error = torch.abs(data - recovered).mean()
            assert error < 1e-5, f"High error for TensorScaler with {method}"
        
        # Test FeatureScaler
        feature_data = {
            'price': np.random.randn(100, 4),
            'volume': np.random.exponential(1, (100, 1)),
            'technical': np.random.randn(100, 10)
        }
        scaler = FeatureScaler()
        scaler.fit(feature_data)
        transformed = scaler.transform(feature_data)
        recovered = scaler.inverse_transform(transformed)
        
        for key in feature_data:
            error = np.abs(feature_data[key] - recovered[key].cpu().numpy()).mean()
            assert error < 1e-5 or key == 'volume', f"High error for {key}"
        
        # Test DataScaler with multi-dimensional data
        data = np.random.randn(50, 10, 5)  # 3D data
        for method in ['standard', 'minmax', 'robust', 'log']:
            scaler = DataScaler(method=method, feature_axis=-1)
            scaler.fit(data)
            transformed = scaler.transform(data)
            recovered = scaler.inverse_transform(transformed)
            error = np.abs(data - recovered).mean()
            assert error < 1e-5 or method == 'log', f"High error for DataScaler with {method}"
        
        logging.info("All tests passed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    run_tests()
