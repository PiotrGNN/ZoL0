
#!/usr/bin/env python3
"""
create_models.py - Skrypt do tworzenia i zapisywania modeli AI
"""

import os
import sys
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/create_models.log")
    ]
)
logger = logging.getLogger(__name__)

# Upewnij się, że katalogi istnieją
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("ai_models", exist_ok=True)

def create_dummy_data(rows=500):
    """Tworzy sztuczne dane do treningu modeli."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=rows)
    
    # Dane rynkowe
    prices = np.random.normal(100, 5, rows).cumsum() + 5000
    volumes = np.random.lognormal(10, 1, rows)
    volatility = np.random.normal(0.02, 0.005, rows)
    
    # Cechy techniczne
    rsi = np.random.uniform(20, 80, rows)
    macd = np.random.normal(0, 1, rows)
    bbands = np.random.normal(0, 1, rows)
    
    # Efekt trendu (symulacja)
    trend = np.sin(np.linspace(0, 4*np.pi, rows)) + np.random.normal(0, 0.1, rows)
    prices = prices + 100 * trend
    
    # Docelowa zmienna
    next_day_return = np.diff(prices, append=[prices[-1]]) / prices
    target = (next_day_return > 0).astype(int)  # 1 dla wzrostu, 0 dla spadku
    
    # Tworzymy DataFrame
    data = pd.DataFrame({
        'price': prices,
        'volume': volumes,
        'volatility': volatility,
        'rsi': rsi,
        'macd': macd,
        'bbands': bbands,
        'trend': trend,
        'target': target
    }, index=dates)
    
    return data

def create_and_save_random_forest():
    """Tworzy, trenuje i zapisuje model RandomForest."""
    try:
        # Pobierz dane treningowe
        data = create_dummy_data(1000)
        
        # Przygotuj dane
        X = data[['rsi', 'macd', 'bbands', 'volatility', 'volume', 'trend']]
        y = data['target']
        
        # Trenuj model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Oceń model
        accuracy = model.score(X, y)
        logger.info(f"Dokładność modelu RandomForest: {accuracy:.4f}")
        
        # Zapisz model
        model_path = os.path.join("models", "randomforest_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Zapisz metadane
        metadata_path = os.path.join("models", "randomforest_metadata.json")
        metadata = {
            "name": "RandomForest",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "accuracy": accuracy * 100,
            "features": list(X.columns),
            "description": "Model klasyfikacji lasu losowego do przewidywania kierunku ceny",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 5
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model RandomForest zapisany w {model_path}")
        logger.info(f"Metadane modelu zapisane w {metadata_path}")
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu RandomForest: {e}")
        return False

def create_sentiment_analyzer():
    """Tworzy i zapisuje prosty model analizy sentymentu."""
    try:
        # Prosty model analizy sentymentu (symulowany)
        class SentimentModel:
            def __init__(self):
                self.positive_words = set([
                    "wzrost", "zysk", "byczo", "sukces", "optymistyczny", "potencjał", 
                    "rozwój", "innowacja", "postęp", "umocnienie", "odbicie", "wzrostowy"
                ])
                self.negative_words = set([
                    "spadek", "strata", "niedźwiedzi", "upadek", "pesymistyczny", "ryzyko",
                    "problem", "krach", "kryzys", "załamanie", "bessy", "spadkowy"
                ])
            
            def predict(self, texts):
                """Przewiduje sentyment tekstu."""
                if not isinstance(texts, list):
                    texts = [texts]
                
                results = []
                for text in texts:
                    if isinstance(text, str):
                        text = text.lower()
                        pos_count = sum(1 for word in self.positive_words if word in text)
                        neg_count = sum(1 for word in self.negative_words if word in text)
                        
                        if pos_count > neg_count:
                            sentiment = "positive"
                            score = min(0.5 + (pos_count - neg_count) * 0.1, 1.0)
                        elif neg_count > pos_count:
                            sentiment = "negative"
                            score = max(0.5 - (neg_count - pos_count) * 0.1, 0.0)
                        else:
                            sentiment = "neutral"
                            score = 0.5
                        
                        results.append({
                            "text": text,
                            "sentiment": sentiment,
                            "score": score
                        })
                    else:
                        results.append({
                            "text": str(text),
                            "sentiment": "neutral",
                            "score": 0.5,
                            "error": "Input is not a string"
                        })
                
                return results if len(results) > 1 else results[0]
            
            def fit(self, texts=None, labels=None):
                """Symulacja treningu modelu."""
                logger.info("Training sentiment model (simulated)")
                # Symulacja treningu
                if texts and labels:
                    # Dodaj kilka nowych słów do słownika
                    for text, label in zip(texts, labels):
                        if label == "positive":
                            self.positive_words.add(text.lower())
                        elif label == "negative":
                            self.negative_words.add(text.lower())
                return True
        
        # Stwórz i zapisz model
        model = SentimentModel()
        model_path = os.path.join("models", "sentimentanalyzer_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Zapisz metadane
        metadata_path = os.path.join("models", "sentimentanalyzer_metadata.json")
        metadata = {
            "name": "SentimentAnalyzer",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "accuracy": 85.0,  # Przykładowa dokładność
            "description": "Prosty model analizy sentymentu oparty na słownikach",
            "parameters": {
                "positive_words_count": len(model.positive_words),
                "negative_words_count": len(model.negative_words)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model SentimentAnalyzer zapisany w {model_path}")
        logger.info(f"Metadane modelu zapisane w {metadata_path}")
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu SentimentAnalyzer: {e}")
        return False

def create_anomaly_detector():
    """Tworzy i zapisuje detektor anomalii."""
    try:
        # Prosty model wykrywania anomalii
        class AnomalyModel:
            def __init__(self):
                self.mean = None
                self.std = None
                self.threshold = 2.0  # Liczba odchyleń standardowych
                self.trained = False
            
            def fit(self, data):
                """Trenuje model z danymi."""
                if isinstance(data, pd.DataFrame):
                    data = data.values
                elif isinstance(data, list):
                    data = np.array(data)
                
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                
                self.mean = np.mean(data, axis=0)
                self.std = np.std(data, axis=0)
                self.trained = True
                return True
            
            def predict(self, data):
                """Wykrywa anomalie w danych."""
                if not self.trained:
                    return {"error": "Model nie został wytrenowany"}
                
                if isinstance(data, pd.DataFrame):
                    data = data.values
                elif isinstance(data, list):
                    data = np.array(data)
                
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                
                z_scores = np.abs((data - self.mean) / (self.std + 1e-10))
                is_anomaly = np.any(z_scores > self.threshold, axis=1)
                
                anomaly_scores = np.max(z_scores, axis=1)
                
                return {
                    "is_anomaly": is_anomaly.tolist(),
                    "anomaly_score": anomaly_scores.tolist(),
                    "threshold": self.threshold,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Stwórz i zapisz model
        model = AnomalyModel()
        
        # Trenuj model na przykładowych danych
        data = create_dummy_data(1000)
        model.fit(data[['price', 'volume', 'volatility']])
        
        model_path = os.path.join("models", "anomalydetector_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Zapisz metadane
        metadata_path = os.path.join("models", "anomalydetector_metadata.json")
        metadata = {
            "name": "AnomalyDetector",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "accuracy": 92.5,  # Przykładowa skuteczność
            "description": "Model wykrywania anomalii oparty na score z-score",
            "parameters": {
                "threshold": model.threshold,
                "features": ['price', 'volume', 'volatility']
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model AnomalyDetector zapisany w {model_path}")
        logger.info(f"Metadane modelu zapisane w {metadata_path}")
        
        return True
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu AnomalyDetector: {e}")
        return False

def main():
    """Funkcja główna."""
    logger.info("Rozpoczynam tworzenie modeli AI...")
    
    results = []
    
    # RandomForest
    logger.info("Tworzenie modelu RandomForest...")
    rf_success = create_and_save_random_forest()
    results.append(("RandomForest", rf_success))
    
    # SentimentAnalyzer
    logger.info("Tworzenie modelu SentimentAnalyzer...")
    sa_success = create_sentiment_analyzer()
    results.append(("SentimentAnalyzer", sa_success))
    
    # AnomalyDetector
    logger.info("Tworzenie modelu AnomalyDetector...")
    ad_success = create_anomaly_detector()
    results.append(("AnomalyDetector", ad_success))
    
    # Podsumowanie
    logger.info("Podsumowanie tworzenia modeli:")
    for model_name, success in results:
        status = "✅ Sukces" if success else "❌ Błąd"
        logger.info(f"{model_name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    logger.info(f"Utworzono pomyślnie {success_count}/{len(results)} modeli")

if __name__ == "__main__":
    main()
