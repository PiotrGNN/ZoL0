
#!/usr/bin/env python3
"""
create_models.py
---------------
Skrypt do tworzenia nowych modeli pikli (pkl) do testowania i rozwoju
"""

import os
import pickle
import logging
import numpy as np
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """Upewnij się, że katalog istnieje."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Utworzono katalog: {directory}")

def create_datascaler_model():
    """Tworzy i zapisuje model DataScaler."""
    from sklearn.preprocessing import StandardScaler
    
    # Tworzenie i trenowanie modelu
    X = np.random.randn(100, 5)  # Przykładowe dane
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Przygotowanie metadanych
    metadata = {
        "name": "datascaler_model",
        "description": "Model skalujący dane",
        "created_at": datetime.now().isoformat(),
        "input_shape": X.shape,
        "scaler_type": "StandardScaler"
    }
    
    # Tworzenie katalogu models, jeśli nie istnieje
    ensure_dir("models")
    
    # Zapisywanie modelu
    model_path = "models/datascaler_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({"model": scaler, "metadata": metadata}, f)
    
    # Zapisywanie metadanych jako JSON
    metadata_path = "models/datascaler_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Utworzono model skalera danych: {model_path}")
    return model_path

def create_random_forest_model():
    """Tworzy i zapisuje model Random Forest."""
    from sklearn.ensemble import RandomForestRegressor
    
    # Tworzenie i trenowanie modelu
    X = np.random.randn(100, 5)  # Przykładowe dane
    y = np.random.randn(100)  # Przykładowe etykiety
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Przygotowanie metadanych
    metadata = {
        "name": "random_forest_model",
        "description": "Model RandomForestRegressor do przewidywania",
        "created_at": datetime.now().isoformat(),
        "input_shape": X.shape,
        "model_type": "RandomForestRegressor",
        "params": {
            "n_estimators": 10,
            "random_state": 42
        }
    }
    
    # Tworzenie katalogu models, jeśli nie istnieje
    ensure_dir("models")
    
    # Zapisywanie modelu
    model_path = "models/random_forest_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({"model": model, "metadata": metadata}, f)
    
    # Zapisywanie metadanych jako JSON
    metadata_path = "models/random_forest_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Utworzono model Random Forest: {model_path}")
    return model_path

def recreate_sentimentanalyzer_model():
    """Odtwarza model SentimentAnalyzer."""
    from ai_models.sentiment_ai import SentimentAnalyzer
    
    # Tworzenie i trenowanie modelu
    model = SentimentAnalyzer()
    
    # Symulacja treningu
    model.fit()
    
    # Przygotowanie metadanych
    metadata = {
        "name": "sentimentanalyzer_model",
        "description": "Model analizy sentymentu",
        "created_at": datetime.now().isoformat(),
        "accuracy": model.accuracy,
        "model_type": model.model_type
    }
    
    # Tworzenie katalogu models, jeśli nie istnieje
    ensure_dir("models")
    
    # Zapisywanie modelu
    model_path = "models/sentimentanalyzer_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({"model": model, "metadata": metadata}, f)
    
    # Zapisywanie metadanych jako JSON
    metadata_path = "models/sentimentanalyzer_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Odtworzono model SentimentAnalyzer: {model_path}")
    return model_path

def recreate_reinforcement_learner_model():
    """Odtwarza model ReinforcementLearner."""
    from ai_models.reinforcement_learning import ReinforcementLearner
    
    # Tworzenie modelu
    model = ReinforcementLearner(state_size=10, action_size=3)
    
    # Przygotowanie metadanych
    metadata = {
        "name": "reinforcement_learner_model",
        "description": "Model uczenia ze wzmocnieniem",
        "created_at": datetime.now().isoformat(),
        "state_size": 10,
        "action_size": 3,
        "learning_rate": model.learning_rate
    }
    
    # Tworzenie katalogu models, jeśli nie istnieje
    ensure_dir("models")
    
    # Zapisywanie modelu
    model_path = "models/reinforcement_learner_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({"model": model, "metadata": metadata}, f)
    
    # Zapisywanie metadanych jako JSON
    metadata_path = "models/reinforcement_learner_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Odtworzono model ReinforcementLearner: {model_path}")
    return model_path

if __name__ == "__main__":
    logger.info("Rozpoczynanie tworzenia modeli...")
    
    try:
        # Tworzenie modeli
        datascaler_path = create_datascaler_model()
        random_forest_path = create_random_forest_model()
        sentiment_path = recreate_sentimentanalyzer_model()
        reinforcement_path = recreate_reinforcement_learner_model()
        
        # Podsumowanie
        logger.info("Wszystkie modele zostały pomyślnie utworzone:")
        logger.info(f"1. DataScaler: {datascaler_path}")
        logger.info(f"2. RandomForest: {random_forest_path}")
        logger.info(f"3. SentimentAnalyzer: {sentiment_path}")
        logger.info(f"4. ReinforcementLearner: {reinforcement_path}")
        
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modeli: {e}")
        raise
