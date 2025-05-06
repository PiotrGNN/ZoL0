"""
anomaly_detection.py
------------------
Moduł zawierający klasę AnomalyDetector do wykrywania anomalii w danych cenowych.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import hashlib

class AnomalyDetector:
    """
    Detector anomalii wykorzystujący Isolation Forest z fallbackiem do prostszych metod.
    Zapewnia mechanizmy odporności na błędy i automatycznego przywracania.
    """
    
    def __init__(self, model_path: Optional[str] = None, sensitivity: float = 0.05):
        """
        Initialize anomaly detector with fallback mechanisms.
        
        Args:
            model_path: Optional path to saved model
            sensitivity: Contamination parameter (0 to 1)
        """
        self.logger = logging.getLogger(__name__)
        self.sensitivity = max(0.01, min(sensitivity, 0.5))  # Clamp between 1% and 50%
        self.model = None
        self.is_initialized = False
        self._backup_scores = []
        self._cache = {}
        self._cache_timeout = timedelta(minutes=30)
        self._last_cache_cleanup = datetime.now()
        
        if model_path:
            try:
                self._load_model(model_path)
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_path}: {e}")
                self._initialize_default_model()
        else:
            self._initialize_default_model()

    def _initialize_default_model(self):
        """Initialize default anomaly detection model with error handling."""
        try:
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(
                contamination=self.sensitivity,
                random_state=42,
                n_estimators=100
            )
            
            # Generate and fit with sample data to ensure model is ready
            sample_size = 1000
            sample_data = np.random.randn(sample_size, 5)  # 5 features
            self.model.fit(sample_data)
            
            self.is_initialized = True
            self.logger.info("Initialized and fitted default IsolationForest model")
        except ImportError:
            self.logger.warning("scikit-learn not available. Using statistical fallback.")
            self._initialize_statistical_fallback()
        except Exception as e:
            self.logger.error(f"Error initializing default model: {e}")
            self._initialize_statistical_fallback()

    def _initialize_statistical_fallback(self):
        """Initialize simple statistical anomaly detection as fallback."""
        self.model = None
        self.is_initialized = True
        self.logger.info("Initialized statistical fallback detection")

    def _load_model(self, model_path: str):
        """Load saved model with validation."""
        import joblib
        
        self.logger.info(f"Loading model from {model_path}")
        loaded_model = joblib.load(model_path)
        
        # Validate loaded model
        if hasattr(loaded_model, 'fit') and hasattr(loaded_model, 'predict'):
            self.model = loaded_model
            self.is_initialized = True
            self.logger.info("Successfully loaded and validated model")
        else:
            raise ValueError("Loaded model missing required methods")

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """
        Train anomaly detection model.
        
        Args:
            data: Training data
            
        Returns:
            bool: Success status
        """
        try:
            if self.model is not None:
                # Use ML model if available
                self.model.fit(self._prepare_data(data))
                self.logger.info("Successfully trained ML model")
            else:
                # Use statistical approach
                self._fit_statistical(data)
                self.logger.info("Fitted statistical parameters")
            return True
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return False

    def _fit_statistical(self, data: Union[pd.DataFrame, np.ndarray]):
        """Fit statistical parameters for fallback detection."""
        prepared_data = self._prepare_data(data)
        self._mean = np.mean(prepared_data, axis=0)
        self._std = np.std(prepared_data, axis=0)
        self._threshold = 3  # Number of standard deviations

    def detect(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Detect anomalies in data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dict with anomaly indices and scores
        """
        if not self.is_initialized:
            try:
                self._initialize_default_model()
            except Exception as e:
                return {"error": f"Initialization failed: {e}"}
        
        try:
            # Check cache
            cache_key = self._get_cache_key(data)
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            prepared_data = self._prepare_data(data)
            
            if self.model is not None:
                # Use ML model
                predictions = self.model.predict(prepared_data)
                scores = self.model.score_samples(prepared_data)
                anomaly_indices = np.where(predictions == -1)[0]
            else:
                # Use statistical approach
                z_scores = np.abs((prepared_data - self._mean) / self._std)
                anomaly_indices = np.where(np.any(z_scores > self._threshold, axis=1))[0]
                scores = -np.max(z_scores, axis=1)  # Negative to match IsolationForest convention
            
            result = {
                "anomaly_indices": anomaly_indices.tolist(),
                "anomaly_scores": scores.tolist(),
                "timestamp": datetime.now().isoformat(),
                "model_type": "ml" if self.model else "statistical"
            }
            
            # Cache result
            self._cache[cache_key] = result
            self._cleanup_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {e}")
            return {"error": str(e)}

    def _prepare_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prepare input data for processing."""
        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)

    def _get_cache_key(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """Generate cache key from input data."""
        if isinstance(data, pd.DataFrame):
            return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        return hashlib.md5(data.tobytes()).hexdigest()

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        now = datetime.now()
        if (now - self._last_cache_cleanup) > timedelta(minutes=5):
            expired = [k for k, v in self._cache.items() 
                      if 'timestamp' in v and 
                      (now - datetime.fromisoformat(v['timestamp'])) > self._cache_timeout]
            for k in expired:
                del self._cache[k]
            self._last_cache_cleanup = now

    def save_model(self, path: str) -> bool:
        """
        Save the current model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            bool: Success status
        """
        if not self.model:
            self.logger.warning("No ML model to save")
            return False
            
        try:
            import joblib
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            self.logger.info(f"Successfully saved model to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict method for scikit-learn compatibility.
        
        Args:
            data: Input data
            
        Returns:
            np.ndarray: Predictions (-1 for anomalies, 1 for normal)
        """
        try:
            result = self.detect(data)
            if "error" in result:
                raise RuntimeError(result["error"])
                
            predictions = np.ones(len(data))
            predictions[result["anomaly_indices"]] = -1
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return np.ones(len(data))  # Return all normal in case of error
