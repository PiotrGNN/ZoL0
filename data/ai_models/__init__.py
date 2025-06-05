"""Bridge package exposing top-level ai_models under data.ai_models."""
import sys
from importlib import import_module

_modules = [
    'model_loader',
    'anomaly_detection',
    'model_recognition',
    'model_training',
    'model_evaluation',
    'feature_engineering',
    'sentiment_ai'
]

for _name in _modules:
    try:
        mod = import_module(f'ai_models.{_name}')
        globals()[_name] = mod
        sys.modules[f'{__name__}.{_name}'] = mod
    except Exception:
        pass

__all__ = _modules
