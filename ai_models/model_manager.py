"""
Advanced AI model management module with versioning and monitoring.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import joblib
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)
import pandas as pd
import hashlib
from pathlib import Path

class ModelRegistry:
    """Model version registry with performance tracking"""
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.registry_path / "versions.json"
        self.versions = self._load_versions()
        self.logger = logging.getLogger(__name__)

    def _load_versions(self) -> Dict[str, Any]:
        """Load version history"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_versions(self) -> None:
        """Save version history"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=4)

    def register_version(
        self,
        model_name: str,
        version: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Register a new model version"""
        if model_name not in self.versions:
            self.versions[model_name] = {}
        
        self.versions[model_name][version] = {
            **metadata,
            "registered_at": datetime.now().isoformat()
        }
        self._save_versions()

    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model"""
        if model_name not in self.versions:
            return None
        return max(self.versions[model_name].keys())

    def get_best_version(
        self,
        model_name: str,
        metric: str = "accuracy"
    ) -> Optional[str]:
        """Get the best performing version of a model"""
        if model_name not in self.versions:
            return None
            
        versions = self.versions[model_name]
        return max(
            versions.keys(),
            key=lambda v: versions[v].get("metrics", {}).get(metric, float('-inf'))
        )

class ModelManager:
    """Advanced AI model manager with versioning and monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path(config.get('model_dir', 'saved_models'))
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.registry = ModelRegistry(str(self.models_dir / "registry"))
        
        # Performance monitoring
        self.performance_log = pd.DataFrame(columns=[
            'model_name', 'version', 'timestamp', 'metric', 'value'
        ])
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load available models
        self._load_available_models()
        
    def _compute_model_hash(self, model: Any) -> str:
        """Compute a hash of the model's weights"""
        if isinstance(model, tf.keras.Model):
            weights = model.get_weights()
            return hashlib.md5(
                np.concatenate([w.flatten() for w in weights]).tobytes()
            ).hexdigest()
        elif isinstance(model, torch.nn.Module):
            state_dict = model.state_dict()
            return hashlib.md5(
                str(sorted(state_dict.items())).encode()
            ).hexdigest()
        else:
            return hashlib.md5(str(model).encode()).hexdigest()

    def _load_available_models(self) -> None:
        """Load all available models from disk with version tracking"""
        try:
            for model_dir in self.models_dir.glob("*"):
                if model_dir.is_dir():
                    # Load model info
                    info_path = model_dir / 'info.json'
                    if info_path.exists():
                        with open(info_path, 'r') as f:
                            model_info = json.load(f)
                            
                        model_name = model_dir.name
                        version = model_info.get('version', 'latest')
                        
                        # Load model based on type
                        try:
                            if model_info['type'] == 'tensorflow':
                                model = tf.keras.models.load_model(
                                    str(model_dir / 'model.h5')
                                )
                            elif model_info['type'] == 'pytorch':
                                model = torch.load(str(model_dir / 'model.pt'))
                            else:  # sklearn models
                                model = joblib.load(str(model_dir / 'model.joblib'))
                            
                            self.models[f"{model_name}_v{version}"] = model
                            self.model_info[f"{model_name}_v{version}"] = model_info
                            
                            # Register version
                            self.registry.register_version(
                                model_name,
                                version,
                                {
                                    "hash": self._compute_model_hash(model),
                                    "info": model_info
                                }
                            )
                            
                            self.logger.info(
                                f"Loaded model: {model_name} (version {version})"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed to load model {model_name}: {str(e)}"
                            )
                    
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")

    def save_model(
        self,
        model_name: str,
        model: Any,
        model_info: Dict[str, Any],
        version: Optional[str] = None
    ) -> bool:
        """Save a model with versioning"""
        try:
            if version is None:
                latest = self.registry.get_latest_version(model_name)
                version = str(int(latest or "0") + 1)
            
            model_dir = self.models_dir / f"{model_name}" / f"v{version}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model info
            model_info['version'] = version
            model_info['saved_at'] = datetime.now().isoformat()
            
            with open(model_dir / 'info.json', 'w') as f:
                json.dump(model_info, f, indent=4)
            
            # Save model based on type
            if model_info['type'] == 'tensorflow':
                model.save(str(model_dir / 'model.h5'))
            elif model_info['type'] == 'pytorch':
                torch.save(model, str(model_dir / 'model.pt'))
            else:  # sklearn models
                joblib.dump(model, str(model_dir / 'model.joblib'))
            
            model_key = f"{model_name}_v{version}"
            self.models[model_key] = model
            self.model_info[model_key] = model_info
            
            # Register version
            self.registry.register_version(
                model_name,
                version,
                {
                    "hash": self._compute_model_hash(model),
                    "info": model_info
                }
            )
            
            self.logger.info(f"Saved model: {model_name} (version {version})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False

    def predict(
        self,
        model_name: str,
        data: Union[np.ndarray, torch.Tensor],
        version: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Make predictions with version control"""
        if version is None:
            version = self.registry.get_latest_version(model_name)
            if version is None:
                raise ValueError(f"No versions found for model: {model_name}")
        
        model_key = f"{model_name}_v{version}"
        if model_key not in self.models:
            raise ValueError(f"Model not found: {model_key}")
            
        try:
            model = self.models[model_key]
            model_info = self.model_info[model_key]
            
            # Convert torch tensors to numpy if needed
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            
            start_time = datetime.now()
            
            if model_info['type'] == 'tensorflow':
                predictions = model.predict(data)
                confidence = np.max(predictions, axis=1)
                prediction = np.argmax(predictions, axis=1)
            elif model_info['type'] == 'pytorch':
                with torch.no_grad():
                    outputs = model(torch.from_numpy(data))
                    predictions = torch.softmax(outputs, dim=1)
                    confidence, prediction = predictions.max(1)
                    confidence = confidence.numpy()
                    prediction = prediction.numpy()
            else:  # sklearn models
                prediction = model.predict(data)
                if hasattr(model, 'predict_proba'):
                    confidence = np.max(model.predict_proba(data), axis=1)
                else:
                    confidence = np.ones_like(prediction)
            
            # Filter by confidence threshold
            mask = confidence >= confidence_threshold
            
            # Record prediction time
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'predictions': prediction[mask],
                'confidence': confidence[mask],
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'version': version,
                'prediction_time': prediction_time,
                'threshold': confidence_threshold
            }
            
            # Log performance metrics
            self._log_prediction_metrics(
                model_name, version, prediction_time, len(data)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def _log_prediction_metrics(
        self,
        model_name: str,
        version: str,
        prediction_time: float,
        batch_size: int
    ) -> None:
        """Log prediction performance metrics"""
        metrics = pd.DataFrame([{
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metric': 'prediction_time',
            'value': prediction_time
        }, {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metric': 'batch_size',
            'value': batch_size
        }])
        
        self.performance_log = pd.concat(
            [self.performance_log, metrics],
            ignore_index=True
        )

    def validate_model(
        self,
        model_name: str,
        validation_data: Union[np.ndarray, torch.Tensor],
        validation_labels: Union[np.ndarray, torch.Tensor],
        version: Optional[str] = None
    ) -> Dict[str, float]:
        """Validate model performance with comprehensive metrics"""
        if version is None:
            version = self.registry.get_latest_version(model_name)
            if version is None:
                raise ValueError(f"No versions found for model: {model_name}")
        
        model_key = f"{model_name}_v{version}"
        if model_key not in self.models:
            raise ValueError(f"Model not found: {model_key}")
            
        try:
            model = self.models[model_key]
            model_info = self.model_info[model_key]
            
            # Convert torch tensors to numpy if needed
            if isinstance(validation_data, torch.Tensor):
                validation_data = validation_data.numpy()
            if isinstance(validation_labels, torch.Tensor):
                validation_labels = validation_labels.numpy()
            
            metrics = {}
            
            if model_info['type'] == 'tensorflow':
                eval_metrics = model.evaluate(validation_data, validation_labels)
                metrics = dict(zip(model.metrics_names, eval_metrics))
            elif model_info['type'] == 'pytorch':
                model.eval()
                with torch.no_grad():
                    outputs = model(torch.from_numpy(validation_data))
                    predictions = outputs.argmax(dim=1).numpy()
            else:  # sklearn models
                predictions = model.predict(validation_data)
            
            # Calculate comprehensive metrics
            if model_info.get('task') == 'classification':
                metrics.update({
                    'accuracy': accuracy_score(validation_labels, predictions),
                    'precision': precision_score(
                        validation_labels, predictions, average='weighted'
                    ),
                    'recall': recall_score(
                        validation_labels, predictions, average='weighted'
                    ),
                    'f1': f1_score(validation_labels, predictions, average='weighted')
                })
            else:  # regression
                metrics.update({
                    'mse': mean_squared_error(validation_labels, predictions),
                    'mae': mean_absolute_error(validation_labels, predictions),
                    'rmse': np.sqrt(mean_squared_error(validation_labels, predictions))
                })
            
            # Log validation metrics
            for metric_name, value in metrics.items():
                self.performance_log = pd.concat([
                    self.performance_log,
                    pd.DataFrame([{
                        'model_name': model_name,
                        'version': version,
                        'timestamp': datetime.now().isoformat(),
                        'metric': metric_name,
                        'value': value
                    }])
                ], ignore_index=True)
            
            # Update registry with new metrics
            self.registry.register_version(
                model_name,
                version,
                {
                    "metrics": metrics,
                    "validated_at": datetime.now().isoformat()
                }
            )
            
            return metrics
                
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            raise

    def get_model_performance(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        metric: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get model performance metrics with filtering"""
        performance = self.performance_log
        
        if model_name:
            performance = performance[performance['model_name'] == model_name]
        if version:
            performance = performance[performance['version'] == version]
        if metric:
            performance = performance[performance['metric'] == metric]
        if start_time:
            performance = performance[
                pd.to_datetime(performance['timestamp']) >= start_time
            ]
        if end_time:
            performance = performance[
                pd.to_datetime(performance['timestamp']) <= end_time
            ]
            
        return performance

    def get_model_versions(
        self,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """Get all versions of a model with their metadata"""
        if model_name not in self.registry.versions:
            return []
            
        return [
            {
                'version': version,
                **info
            }
            for version, info in self.registry.versions[model_name].items()
        ]

    def delete_model_version(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """Delete a specific version of a model"""
        try:
            model_key = f"{model_name}_v{version}"
            if model_key in self.models:
                del self.models[model_key]
            if model_key in self.model_info:
                del self.model_info[model_key]
                
            # Remove from registry
            if (model_name in self.registry.versions and
                version in self.registry.versions[model_name]):
                del self.registry.versions[model_name][version]
                self.registry._save_versions()
            
            # Remove files
            model_dir = self.models_dir / model_name / f"v{version}"
            if model_dir.exists():
                for file in model_dir.glob("*"):
                    file.unlink()
                model_dir.rmdir()
            
            self.logger.info(
                f"Deleted model version: {model_name} (version {version})"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to delete model version: {str(e)}"
            )
            return False

    def export_metrics(
        self,
        output_path: str,
        format: str = 'csv'
    ) -> bool:
        """Export performance metrics to file"""
        try:
            if format == 'csv':
                self.performance_log.to_csv(output_path, index=False)
            elif format == 'json':
                self.performance_log.to_json(output_path, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Exported metrics to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    config = {
        'model_dir': 'saved_models',
        'logging_level': 'INFO'
    }
    
    manager = ModelManager(config)
    
    # Example: Save a new model version
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    manager.save_model(
        'example_model',
        model,
        {
            'type': 'tensorflow',
            'task': 'classification',
            'description': 'Example classification model'
        }
    )
    
    # Example: Make predictions
    data = np.random.random((10, 64))
    predictions = manager.predict('example_model', data)
    
    # Example: Get model performance
    performance = manager.get_model_performance('example_model')
    print("\nModel Performance:")
    print(performance)