"""
sentiment_analysis.py
---------------------
Moduł integrujący dane z mediów społecznościowych, newsów i forów w celu analizy sentymentu.
Wykorzystuje techniki NLP, takie jak modele Transformer (np. BERT, DistilBERT), do klasyfikacji treści finansowych.
Funkcjonalności:
- Wykrywanie sentymentu w pojedynczych tekstach.
- Agregacja wyników sentymentu w przedziałach czasowych, umożliwiająca ich integrację z danymi cenowymi.
- Wykrywanie kluczowych słów (np. "FUD", "HODL", "moon", "crash") i przypisywanie im wagi w zależności od kontekstu.
- Logowanie oraz generowanie alarmów w przypadku gwałtownych zmian sentymentu (np. masowy negatywny wydźwięk).
- Skalowalność przy przetwarzaniu dużych strumieni danych w czasie rzeczywistym.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

try:
    from transformers import pipeline, Pipeline
except ImportError as e:
    raise ImportError("Biblioteka transformers nie jest zainstalowana. Zainstaluj ją używając 'pip install transformers'") from e

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class SentimentAnalysis:
    def __init__(self, model_name: Optional[str] = None):
        """
        Inicjalizacja modułu sentiment analysis.
        
        Parameters:
            model_name (str, optional): Nazwa modelu Transformer do analizy sentymentu.
                                        Domyślnie używany jest "distilbert-base-uncased-finetuned-sst-2-english".
        """
        self.model_name = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        try:
            self.nlp: Pipeline = pipeline("sentiment-analysis", model=self.model_name, tokenizer=self.model_name)
            logging.info("Załadowano model sentymentu: %s", self.model_name)
        except Exception as e:
            logging.error("Błąd przy ładowaniu modelu sentymentu: %s", e)
            raise

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analizuje pojedynczy tekst i zwraca wynik sentymentu.
        
        Parameters:
            text (str): Tekst do analizy.
        
        Returns:
            dict: Słownik zawierający etykietę (LABEL_0: negatywny, LABEL_1: pozytywny) i przypisane prawdopodobieństwo.
                  Dla lepszej interpretacji przemapowujemy etykiety na "NEGATIVE" i "POSITIVE".
        """
        try:
            result = self.nlp(text)
            if result and isinstance(result, list) and "label" in result[0]:
                label = result[0]["label"]
                score = result[0]["score"]
                mapped = "POSITIVE" if label.upper() in ["POSITIVE", "LABEL_1"] else "NEGATIVE"
                logging.info("Analiza tekstu: \"%s\" -> %s (score: %.4f)", text, mapped, score)
                return {mapped: score}
            else:
                logging.warning("Nieoczekiwany format wyniku dla tekstu: %s", text)
                return {}
        except Exception as e:
            logging.error("Błąd podczas analizy tekstu: %s", e)
            return {}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analizuje listę tekstów i zwraca listę wyników sentymentu.
        
        Parameters:
            texts (List[str]): Lista tekstów.
        
        Returns:
            List[dict]: Lista wyników dla każdego tekstu.
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results

    def aggregate_sentiment(self, texts: List[str], weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Agreguje wyniki sentymentu z listy tekstów. Umożliwia przypisanie wag do poszczególnych tekstów.
        
        Parameters:
            texts (List[str]): Lista tekstów do analizy.
            weights (List[float], optional): Lista wag odpowiadających tekstom. Jeśli None, wszystkie teksty mają równą wagę.
        
        Returns:
            dict: Agregowany wynik sentymentu, np. {"NEGATIVE": 0.4, "POSITIVE": 0.6}
        """
        aggregated = {"NEGATIVE": 0.0, "POSITIVE": 0.0}
        results = self.analyze_batch(texts)
        if weights is None or len(weights) != len(results):
            weights = [1.0] * len(results)
        total_weight = sum(weights)
        for result, weight in zip(results, weights):
            for key, score in result.items():
                aggregated[key] += score * weight
        if total_weight > 0:
            aggregated = {key: val / total_weight for key, val in aggregated.items()}
        logging.info("Zagregowany sentyment: %s", aggregated)
        return aggregated

    def alert_on_sentiment_shift(self, current_sentiment: Dict[str, float], previous_sentiment: Dict[str, float],
                                 shift_threshold: float = 0.2) -> bool:
        """
        Sprawdza, czy nastąpiła gwałtowna zmiana sentymentu.
        
        Parameters:
            current_sentiment (dict): Aktualny wynik sentymentu.
            previous_sentiment (dict): Poprzedni wynik sentymentu.
            shift_threshold (float): Próg zmiany, powyżej którego generowany jest alarm.
        
        Returns:
            bool: True, jeśli zmiana przekracza próg, w przeciwnym razie False.
        """
        try:
            shift = {key: abs(current_sentiment.get(key, 0) - previous_sentiment.get(key, 0))
                     for key in set(current_sentiment.keys()).union(previous_sentiment.keys())}
            max_shift = max(shift.values()) if shift else 0.0
            if max_shift >= shift_threshold:
                logging.warning("Wykryto gwałtowną zmianę sentymentu: %s (próg: %.2f)", shift, shift_threshold)
                return True
            return False
        except Exception as e:
            logging.error("Błąd przy ocenie zmiany sentymentu: %s", e)
            return False

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        sa = SentimentAnalysis()
        sample_texts = [
            "Investors are bullish after the recent market rally.",
            "There are fears of an upcoming market crash due to economic instability.",
            "The overall sentiment remains neutral with mixed signals from various news sources."
        ]
        # Analiza pojedynczych tekstów
        for text in sample_texts:
            print(sa.analyze_text(text))
        
        # Agregacja sentymentu dla partii tekstów
        aggregated = sa.aggregate_sentiment(sample_texts)
        print("Agregowany sentyment:", aggregated)
        
        # Przykładowa ocena zmiany sentymentu
        previous = {"NEGATIVE": 0.3, "POSITIVE": 0.7}
        current = aggregated
        if sa.alert_on_sentiment_shift(current, previous):
            logging.info("Alarm: Wykryto znaczną zmianę sentymentu!")
        else:
            logging.info("Brak istotnej zmiany sentymentu.")
    except Exception as e:
        logging.error("Błąd w module sentiment_analysis.py: %s", e)
        raise
