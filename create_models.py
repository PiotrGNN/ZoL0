
#!/usr/bin/env python3
"""
create_models.py - Skrypt do tworzenia i zapisywania modeli AI
"""
import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Konfiguracja logowania
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/create_models.log")
    ]
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Tworzy wymaganą strukturę katalogów."""
    directories = [
        "models",
        "saved_models",
        "logs",
        "data/cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Utworzono katalog: {directory}")

def create_anomaly_detector():
    """Tworzy i zapisuje model detektora anomalii."""
    logger.info("Tworzenie modelu detektora anomalii...")

    # Tworzymy przykładowe dane treningowe
    np.random.seed(42)
    data = np.random.normal(0, 1, (1000, 5))
    df = pd.DataFrame(
        data,
        columns=["feature1", "feature2", "feature3", "feature4", "feature5"]
    )
    
    # Dodanie kilku anomalii
    for i in range(10):
        idx = np.random.randint(0, 1000)
        df.iloc[idx] = np.random.normal(10, 3, 5)
    
    # Tworzenie i trenowanie modelu
    isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )
    
    isolation_forest.fit(df)
    
    # Zapisywanie modelu
    model_path = os.path.join("models", "anomaly_detector_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(isolation_forest, f)
    
    # Zapisywanie metadanych
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "IsolationForest",
        "contamination": 0.01,
        "n_features": 5,
        "n_samples_trained": 1000
    }
    
    metadata_path = os.path.join("models", "anomaly_detector_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model detektora anomalii zapisany do: {model_path}")
    logger.info(f"Metadane modelu zapisane do: {metadata_path}")
    
    return isolation_forest

def create_sentiment_analyzer():
    """Tworzy i zapisuje prosty model analizatora sentymentu."""
    logger.info("Tworzenie prostego modelu analizatora sentymentu...")
    
    # W tym przypadku tworzymy bardzo prosty model oparty na słowniku
    sentiment_dict = {
        "positive_words": [
            "bullish", "growth", "gain", "profit", "up", "increase", "positive",
            "success", "rally", "recover", "support", "buy", "strong", "higher"
        ],
        "negative_words": [
            "bearish", "loss", "drop", "crash", "down", "decrease", "negative",
            "fail", "sell", "weak", "lower", "poor", "decline", "risk"
        ],
        "neutral_words": [
            "market", "trade", "price", "volume", "level", "range", "stable",
            "steady", "flat", "consolidation", "unchanged", "hold", "mixed"
        ]
    }
    
    # Zapisywanie modelu (w tym przypadku po prostu słownika)
    model_path = os.path.join("models", "sentimentanalyzer_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(sentiment_dict, f)
    
    # Zapisywanie metadanych
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "Dictionary-based",
        "positive_words_count": len(sentiment_dict["positive_words"]),
        "negative_words_count": len(sentiment_dict["negative_words"]),
        "neutral_words_count": len(sentiment_dict["neutral_words"])
    }
    
    metadata_path = os.path.join("models", "sentimentanalyzer_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model analizatora sentymentu zapisany do: {model_path}")
    logger.info(f"Metadane modelu zapisane do: {metadata_path}")
    
    return sentiment_dict

def create_data_scaler():
    """Tworzy i zapisuje model skalera danych."""
    logger.info("Tworzenie modelu skalera danych...")
    
    # Tworzymy przykładowe dane treningowe
    np.random.seed(42)
    data = np.random.normal(0, 1, (1000, 5))
    # Dodajmy offset do danych, aby były bardziej realistyczne
    data[:, 0] += 100  # np. cena
    data[:, 1] += 1000  # np. wolumen
    data[:, 2] += 50   # np. RSI
    data[:, 3] += 25   # np. CCI
    data[:, 4] += 10   # np. ATR
    
    df = pd.DataFrame(
        data,
        columns=["price", "volume", "rsi", "cci", "atr"]
    )
    
    # Tworzenie i trenowanie modelu
    scaler = StandardScaler()
    scaler.fit(df)
    
    # Zapisywanie modelu
    model_path = os.path.join("models", "datascaler_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(scaler, f)
    
    # Zapisywanie metadanych
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "StandardScaler",
        "feature_means": scaler.mean_.tolist(),
        "feature_stds": scaler.scale_.tolist(),
        "feature_names": df.columns.tolist()
    }
    
    metadata_path = os.path.join("models", "datascaler_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model skalera danych zapisany do: {model_path}")
    logger.info(f"Metadane modelu zapisane do: {metadata_path}")
    
    return scaler

def create_reinforcement_learner():
    """Tworzy i zapisuje prosty model uczenia ze wzmocnieniem."""
    logger.info("Tworzenie modelu uczenia ze wzmocnieniem...")
    
    # Bardzo uproszczony model uczenia ze wzmocnieniem
    reinforcement_model = {
        "weights": {
            "price_change": 0.5,
            "volume_change": 0.3,
            "trend_signal": 0.8,
            "momentum_signal": 0.6,
            "volatility_signal": -0.4
        },
        "bias": 0.1,
        "learning_rate": 0.01,
        "discount_factor": 0.95,
        "exploration_rate": 0.2
    }
    
    # Zapisywanie modelu
    model_path = os.path.join("models", "reinforcement_learner_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(reinforcement_model, f)
    
    # Zapisywanie metadanych
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "ReinforcementLearner",
        "weights_count": len(reinforcement_model["weights"]),
        "learning_rate": reinforcement_model["learning_rate"],
        "discount_factor": reinforcement_model["discount_factor"]
    }
    
    metadata_path = os.path.join("models", "reinforcementlearner_metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model uczenia ze wzmocnieniem zapisany do: {model_path}")
    logger.info(f"Metadane modelu zapisane do: {metadata_path}")
    
    return reinforcement_model

def main():
    """Funkcja główna skryptu."""
    logger.info("Rozpoczynanie procesu tworzenia modeli AI...")
    
    # Tworzenie struktury katalogów
    create_directory_structure()
    
    # Tworzenie modeli
    try:
        anomaly_detector = create_anomaly_detector()
        logger.info("Model detektora anomalii utworzony pomyślnie")
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu detektora anomalii: {e}")
    
    try:
        sentiment_analyzer = create_sentiment_analyzer()
        logger.info("Model analizatora sentymentu utworzony pomyślnie")
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu analizatora sentymentu: {e}")
    
    try:
        data_scaler = create_data_scaler()
        logger.info("Model skalera danych utworzony pomyślnie")
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu skalera danych: {e}")
    
    try:
        reinforcement_learner = create_reinforcement_learner()
        logger.info("Model uczenia ze wzmocnieniem utworzony pomyślnie")
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu uczenia ze wzmocnieniem: {e}")
    
    logger.info("Proces tworzenia modeli AI zakończony.")

if __name__ == "__main__":
    main()
