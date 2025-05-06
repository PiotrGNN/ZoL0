# Trading System API Documentation

## Environment API

### MarketEnvironment

Base class for trading environments.

```python
class MarketEnvironment:
    def __init__(
        initial_capital: float = 10000,
        leverage: float = 1.0,
        risk_limit: float = 0.05,
        data: Optional[pd.DataFrame] = None,
        mode: str = "simulated"
    )
```

**Parameters:**
- `initial_capital`: Starting capital amount
- `leverage`: Trading leverage multiplier
- `risk_limit`: Maximum risk per trade (as percentage)
- `data`: Optional historical price data
- `mode`: Operating mode ("simulated", "paper", "backtesting")

**Methods:**
- `reset()`: Reset environment to initial state
- `step(action: str)`: Execute trading action and return next state
- `_get_state()`: Get current environment state

### MarketDummyEnv

Simulated market environment for testing strategies.

```python
class MarketDummyEnv:
    def __init__(
        initial_capital: float = 10000,
        volatility: float = 1.0,
        liquidity: float = 1000,
        spread: float = 0.05,
        commission: float = 0.001,
        slippage: float = 0.1
    )
```

**Parameters:**
- `volatility`: Price volatility factor
- `liquidity`: Market liquidity factor
- `spread`: Bid-ask spread
- `commission`: Trading commission rate
- `slippage`: Price slippage factor

## Model Training API

### ModelTrainer

AI model training and management.

```python
class ModelTrainer:
    def __init__(
        model: Union[tf.keras.Model, Any],
        model_name: str,
        saved_model_dir: str = "saved_models",
        online_learning: bool = False,
        use_gpu: bool = False,
        early_stopping_params: Optional[Dict] = None
    )
```

**Parameters:**
- `model`: Machine learning model instance
- `model_name`: Unique model identifier
- `saved_model_dir`: Directory for model storage
- `online_learning`: Enable continuous learning
- `use_gpu`: Use GPU acceleration if available
- `early_stopping_params`: Early stopping configuration

**Methods:**
- `train(X, y)`: Train model on data
- `walk_forward_split()`: Time series cross-validation
- `save_model()`: Save model to disk
- `load_model()`: Load model from disk

## Error Handling

### Exception Classes

- `TradingSystemError`: Base exception class
- `ModelError`: Model-related errors
- `TradeExecutionError`: Trading errors
- `ConfigurationError`: Configuration errors

### Error Handling Decorator

```python
@handle_errors(
    error_types: Union[Type[Exception], tuple],
    fallback_value: Any = None,
    log_level: str = "ERROR"
)
```

**Parameters:**
- `error_types`: Exception types to catch
- `fallback_value`: Return value on error
- `log_level`: Logging level for errors

## Configuration

Configuration is managed through `config.yaml` with the following sections:

- `environment`: System environment settings
- `trading`: Trading parameters
- `ai_models`: AI model configuration
- `api`: API server settings
- `database`: Database configuration
- `logging`: Logging settings

## Testing

Run tests using pytest:

```bash
pytest tests/            # Run all tests
pytest tests/ -m unit    # Run unit tests only
pytest tests/ -m integration  # Run integration tests
```

Test markers:
- `unit`: Unit tests
- `integration`: Integration tests
- `model`: AI model tests
- `trading`: Trading system tests