
"""
sentiment_analysis.py
---------------------
Moduł do analizy nastrojów rynkowych z różnych źródeł danych.
"""

import logging
import random
from typing import Dict, List, Tuple, Union, Any

class SentimentAnalyzer:
    """
    Klasa analizująca nastroje rynkowe z różnych źródeł.
    """

    def __init__(self, sources: List[str] = None):
        """
        Inicjalizuje analizator sentymentu z określonymi źródłami.
        
        Parameters:
            sources (List[str], optional): Lista źródeł danych do analizy.
        """
        self.sources = sources or ["twitter", "news", "forum", "reddit"]
        logging.info(f"Inicjalizacja analizatora sentymentu z {len(self.sources)} źródłami.")
        self._sentiment_cache = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Analizuje nastroje rynkowe i zwraca wynik.
        
        Returns:
            Dict[str, Any]: Wynik analizy sentymentu zawierający wartość sentymentu i dane źródłowe.
        """
        try:
            # Symulujemy analizę sentymentu (w rzeczywistości powinno używać modelu ML)
            sentiment_values = {
                "twitter": random.uniform(-1.0, 1.0),
                "news": random.uniform(-0.5, 1.0),
                "forum": random.uniform(-0.7, 0.7),
                "reddit": random.uniform(-0.8, 0.8)
            }
            
            # Obliczamy średni sentyment
            available_sources = set(self.sources).intersection(sentiment_values.keys())
            if not available_sources:
                logging.warning("Brak danych sentymentu dla skonfigurowanych źródeł.")
                return {"analysis": "Neutralny", "value": 0.0, "sources": {}}
            
            avg_sentiment = sum(sentiment_values[source] for source in available_sources) / len(available_sources)
            
            # Konwertujemy wartość numeryczną na opis tekstowy
            sentiment_text = self._get_sentiment_text(avg_sentiment)
            
            result = {
                "analysis": sentiment_text,
                "value": avg_sentiment,
                "sources": {source: sentiment_values[source] for source in available_sources}
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Błąd podczas analizy sentymentu: {e}")
            return {"analysis": "Neutralny", "value": 0.0, "sources": {}, "error": str(e)}

    def get_current_sentiment(self) -> str:
        """
        Zwraca aktualny sentyment rynkowy jako tekst.
        
        Returns:
            str: Tekstowy opis aktualnego sentymentu.
        """
        result = self.analyze()
        return result["analysis"]

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analizuje sentyment dla listy tekstów.
        
        Parameters:
            texts (List[str]): Lista tekstów do analizy.
            
        Returns:
            List[Dict[str, float]]: Lista wyników analizy.
        """
        results = []
        for text in texts:
            # Symulacja analizy sentymentu (w rzeczywistości powinno używać NLP)
            sentiment_value = self._simulate_sentiment_analysis(text)
            results.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "sentiment": sentiment_value,
                "sentiment_text": self._get_sentiment_text(sentiment_value)
            })
        return results

    def weighted_sentiment(self, texts_with_source: List[Tuple[str, str]], 
                          source_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Analizuje sentyment z różnym ważeniem źródeł.
        
        Parameters:
            texts_with_source (List[Tuple[str, str]]): Lista par (tekst, źródło).
            source_weights (Dict[str, float], optional): Wagi dla różnych źródeł.
            
        Returns:
            Dict[str, Any]: Wynik ważonej analizy sentymentu.
        """
        if not source_weights:
            source_weights = {source: 1.0 for source in self.sources}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for text, source in texts_with_source:
            if source not in source_weights:
                logging.warning(f"Źródło {source} nie ma przypisanej wagi. Używam 1.0.")
                weight = 1.0
            else:
                weight = source_weights[source]
            
            sentiment_value = self._simulate_sentiment_analysis(text)
            weighted_sum += sentiment_value * weight
            total_weight += weight
        
        if total_weight == 0:
            average_sentiment = 0.0
        else:
            average_sentiment = weighted_sum / total_weight
        
        return {
            "weighted_sentiment": average_sentiment,
            "sentiment_text": self._get_sentiment_text(average_sentiment),
            "included_sources": len(texts_with_source)
        }

    def _simulate_sentiment_analysis(self, text: str) -> float:
        """
        Symuluje analizę sentymentu dla tekstu (zamiast faktycznego przetwarzania NLP).
        
        Parameters:
            text (str): Tekst do analizy.
            
        Returns:
            float: Wartość sentymentu.
        """
        # Proste symulowanie sentymentu na podstawie słów kluczowych
        positive_words = ['good', 'great', 'positive', 'excellent', 'bullish', 'up', 'gain', 'profit', 'optimistic', 'recovery']
        negative_words = ['bad', 'poor', 'negative', 'bearish', 'down', 'loss', 'crash', 'pessimistic', 'decline', 'worry']
        
        text = text.lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Dodajemy losową składową dla realistyczności
        random_component = random.uniform(-0.3, 0.3)
        
        if positive_count == negative_count:
            return random_component
        elif positive_count > negative_count:
            return 0.5 + random_component + (positive_count - negative_count) * 0.1
        else:
            return -0.5 + random_component - (negative_count - positive_count) * 0.1

    def _get_sentiment_text(self, value: float) -> str:
        """
        Konwertuje wartość liczbową sentymentu na opis tekstowy.
        
        Parameters:
            value (float): Wartość sentymentu.
            
        Returns:
            str: Tekstowy opis sentymentu.
        """
        if value > 0.5:
            return "Bardzo pozytywny"
        elif value > 0.1:
            return "Pozytywny"
        elif value > -0.1:
            return "Neutralny"
        elif value > -0.5:
            return "Negatywny"
        else:
            return "Bardzo negatywny"
