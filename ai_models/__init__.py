"""
Inicjalizacja pakietu ai_models.

Ten pakiet zawiera zaawansowane modele sztucznej inteligencji i uczenia maszynowego
używane w systemie tradingowym, w tym:
- wykrywanie anomalii
- analiza sentymentu
- uczenie ze wzmocnieniem
- inżynieria cech
- predykcja trendów rynkowych
"""

from typing import List

__all__: List[str] = [
    "AnomalyDetectionModel",
    "SentimentAnalyzer",
    "FeatureEngineer",
]

# Importy modeli
try:
    from .anomaly_detection import AnomalyDetectionModel
except ImportError:
    pass

try:
    from .sentiment_ai import SentimentAnalyzer
except ImportError:
    pass

try:
    from .feature_engineering import FeatureEngineer
except ImportError:
    pass
"""
AI Models Package
----------------
Pakiet zawierający modele uczenia maszynowego i sztucznej inteligencji
wykorzystywane w systemie tradingowym.
"""

from .anomaly_detection import AnomalyDetectionModel

__all__ = [
    'AnomalyDetectionModel',
]
