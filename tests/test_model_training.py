"""
Tests for model training functionality
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ai_models.model_training import ModelTrainer, prepare_data_for_model
from utils.error_handling import ModelError

@pytest.fixture
def trainer():
    model = RandomForestRegressor(n_estimators=10)
    return ModelTrainer(model=model, model_name="test_model")

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create DataFrame with features
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    # Create target Series
    y = pd.Series(np.random.randn(100))
    
    return X, y

@pytest.mark.unit
class TestModelTrainer:
    def test_initialization(self, trainer):
        """Test ModelTrainer initialization"""
        assert trainer.model_name == "test_model"
        assert isinstance(trainer.model, RandomForestRegressor)
        assert trainer.online_learning is False
        assert isinstance(trainer.model_metadata, dict)

    def test_walk_forward_split(self, trainer, sample_data):
        """Test walk-forward cross validation split"""
        X, y = sample_data
        splits = trainer.walk_forward_split(X, y, n_splits=3)
        
        assert len(splits) == 3
        for X_train, X_test, y_train, y_test in splits:
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(y_test, pd.Series)
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) == len(X_train)
            assert len(y_test) == len(X_test)

    def test_model_training(self, trainer, sample_data):
        """Test model training process"""
        X, y = sample_data
        result = trainer.train(X=X, y=y, n_splits=3, epochs=2, batch_size=32)
        
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'model' in result
        assert 'metrics' in result
        assert isinstance(result['metrics'], dict)

    def test_data_preparation(self, sample_data):
        """Test data preparation function"""
        X, _ = sample_data
        prepared_data = prepare_data_for_model(X)
        
        assert isinstance(prepared_data, np.ndarray)
        assert prepared_data.shape == X.shape

    @pytest.mark.parametrize("invalid_data", [
        None,
        "invalid",
        [],
        np.array([])
    ])
    def test_invalid_data_handling(self, trainer, invalid_data):
        """Test handling of invalid data"""
        with pytest.raises((ValueError, ModelError)):
            trainer.train(X=invalid_data, y=invalid_data)

@pytest.mark.integration
class TestModelIntegration:
    def test_end_to_end_training(self, trainer, sample_data):
        """Test end-to-end model training process"""
        X, y = sample_data
        
        # Train model
        result = trainer.train(X, y, force_train=True)
        assert result['success'] is True
        
        # Verify model can make predictions
        X_test = X.head(5)
        predictions = trainer.model.predict(prepare_data_for_model(X_test))
        assert len(predictions) == len(X_test)
        
        # Verify model metadata is saved
        assert trainer.model_metadata.get('train_date') is not None
        assert trainer.model_metadata.get('features_shape') == X.shape