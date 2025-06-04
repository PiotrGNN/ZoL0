import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import NotFittedError


def prepare_data_for_model(data, features_count=None):
    """Convert input data to numpy array with basic validation."""
    if isinstance(data, pd.DataFrame):
        arr = data.values
    elif isinstance(data, list):
        arr = np.array(data)
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValueError("Unsupported data type")
    if features_count is not None and arr.shape[1] != features_count:
        raise ValueError("feature count mismatch")
    return arr


class ModelTrainer:
    """Very small model trainer used for tests."""

    def __init__(self, model, model_name="model", saved_model_dir="models", online_learning=True, use_gpu=False):
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise ValueError("Model must implement fit and predict")
        self.model = model
        self.model_name = model_name
        self.saved_model_dir = saved_model_dir
        self.online_learning = online_learning
        self.use_gpu = use_gpu
        os.makedirs(saved_model_dir, exist_ok=True)
        self._trained = False

    # Basic data validation
    def validate_data(self, X, y):
        if X is None or y is None:
            raise ValueError("Invalid data")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty input")
        if len(X) != len(y):
            raise ValueError("Length mismatch")

    def train(self, X, y, force_train=True):
        self.validate_data(X, y)
        if self._trained and not force_train:
            return {"success": True, "skipped": True}
        self.model.fit(X, y)
        self._trained = True
        metrics = self.evaluate(X, y)
        return {"success": True, "skipped": False, "metrics": metrics}

    def evaluate(self, X, y):
        if not self._trained:
            raise NotFittedError("Model not trained")
        preds = self.model.predict(X)
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        return {"test_score": float(self.model.score(X, y)), "mae": float(mae), "mse": float(mse), "rmse": float(rmse)}

    def update_online(self, X_new, y_new):
        if not self.online_learning:
            raise ValueError("Online learning disabled")
        return self.train(X_new, y_new, force_train=True)

    def save_model(self, path=None):
        if not self._trained:
            raise ValueError("Model not trained")
        path = path or os.path.join(self.saved_model_dir, self.model_name + ".pkl")
        joblib.dump(self.model, path)
        meta_path = path + ".meta"
        with open(meta_path, "w") as f:
            f.write("model_name:" + self.model_name)
        return path

    def load_model(self, path):
        return joblib.load(path)

    def get_feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_.tolist()
        if hasattr(self.model, "coef_"):
            return self.model.coef_.tolist()
        return None
