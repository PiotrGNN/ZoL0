import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error

class ModelEvaluator:
    """Minimal evaluator returning basic regression metrics."""
    def evaluate_model(self, model, X, y):
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, preds)
        max_err = max_error(y, preds)
        return {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse), 'r2': float(r2), 'max_error': float(max_err)}

    def get_prediction_intervals(self, model, X, confidence_level=0.95):
        raise NotImplementedError
