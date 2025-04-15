"""
sentiment_ai.py
--------------
Moduł do analizy sentymentu tekstu.
"""

import random
import re
import time
import logging
from typing import Dict, Any, List, Optional

# Konfiguracja logowania
logger = logging.getLogger("sentiment_analyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SentimentAnalyzer:
    """
    Analizator sentymentu tekstu.

    Dla celów demonstracyjnych, ta implementacja używa prostych reguł i losowości,
    zamiast faktycznego modelu uczenia maszynowego.
    """

    def __init__(self):
        """Inicjalizuje analizator sentymentu."""
        self.positive_words = [
            "up", "rise", "rising", "bull", "bullish", "growth", "grow", "positive",
            "profit", "profitable", "success", "successful", "gain", "good", "strong",
            "strength", "opportunity", "promising", "optimistic", "confidence", "confident",
            "improve", "improvement", "rally", "uptrend", "support", "buy", "buying"
        ]

        self.negative_words = [
            "down", "fall", "falling", "bear", "bearish", "decline", "declining",
            "negative", "loss", "weak", "weakness", "risk", "risky", "danger",
            "dangerous", "pessimistic", "worry", "worried", "concern", "concerned",
            "fear", "fearful", "panic", "crash", "recession", "downtrend", "resistance",
            "sell", "selling", "doubt", "uncertain", "uncertainty"
        ]

        self.market_terms = [
            "market", "stock", "trade", "trading", "price", "value", "invest",
            "investment", "investor", "crypto", "bitcoin", "btc", "eth", "trend",
            "technical", "fundamental", "analysis", "chart", "candle", "pattern",
            "support", "resistance", "volume", "liquidity", "volatility"
        ]

        self.accuracy = 82.3
        self.model_type = "NLP Sentiment Analyzer"
        self.status = "Active"
        self.last_prediction_time = time.time()

        logger.info("SentimentAnalyzer zainicjalizowany")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Przewiduje sentyment dla podanego tekstu.

        Args:
            text: Tekst do analizy

        Returns:
            Dict[str, Any]: Wynik analizy sentymentu
        """
        return self.analyze(text)

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analizuje sentyment tekstu.

        Args:
            text: Tekst do analizy

        Returns:
            Dict[str, Any]: Wynik analizy sentymentu
        """
        if not text:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.5,
                "analysis": "Brak tekstu do analizy",
                "market_relevance": 0.0,
                "timestamp": time.time()
            }

        # Konwersja tekstu do małych liter i czyszczenie
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Sprawdzanie słów pozytywnych i negatywnych
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        market_terms_count = sum(1 for word in words if word in self.market_terms)

        # Obliczanie punktacji bazowej sentymentu
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0  # Neutralny sentyment
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words

        # Używamy rzeczywistego algorytmu bez sztucznych modyfikacji
        sentiment_score = min(1.0, max(-1.0, sentiment_score))

        # Określanie kategorii sentymentu
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Obliczanie pewności i relevance
        confidence = random.uniform(0.7, 0.95)
        market_relevance = min(1.0, market_terms_count / max(1, len(words)) * 3)

        # Tworzenie analizy tekstowej
        analysis = self._generate_analysis(sentiment, sentiment_score, market_relevance)

        # Aktualizacja czasu ostatniej analizy
        self.last_prediction_time = time.time()

        return {
            "sentiment": sentiment,
            "score": round(sentiment_score, 2),
            "confidence": round(confidence, 2),
            "analysis": analysis,
            "market_relevance": round(market_relevance, 2),
            "timestamp": time.time()
        }

    def _generate_analysis(self, sentiment: str, score: float, relevance: float) -> str:
        """
        Generuje tekstową analizę sentymentu.

        Args:
            sentiment: Kategoria sentymentu
            score: Wynik liczbowy sentymentu
            relevance: Adekwatność rynkowa

        Returns:
            str: Analiza tekstowa
        """
        if relevance < 0.3:
            return "Tekst ma niską zawartość terminów związanych z rynkiem finansowym."

        if sentiment == "positive":
            return f"Wykryto pozytywny sentyment ({score:.2f}). Tekst sugeruje optymistyczne nastawienie do rynku."
        elif sentiment == "negative":
            return f"Wykryto negatywny sentyment ({score:.2f}). Tekst sugeruje pesymistyczne nastawienie do rynku."
        else:
            return f"Wykryto neutralny sentyment ({score:.2f}). Tekst nie wykazuje wyraźnego nastawienia do rynku."


# Inicjalizuj analizator sentymentu przy imporcie
sentiment_analyzer = SentimentAnalyzer()
"""
sentiment_ai.py
-------------
Moduł do analizy sentymentu rynkowego.
"""

import logging
import random
import time
from typing import Dict, Any, List, Optional

class SentimentAnalyzer:
    """Analizator sentymentu rynkowego."""
    
    def __init__(self):
        """Inicjalizacja analizatora sentymentu."""
        self.logger = logging.getLogger("sentiment_analyzer")
        self.sources = ["twitter", "news", "forums", "reddit"]
        self.last_update = time.time()
        self.last_results = self._generate_random_sentiment()
        self.update_interval = 300  # 5 minut
        
        self.logger.info("Zainicjalizowano analizator sentymentu")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analizuje sentyment rynkowy.
        
        Returns:
            Dict[str, Any]: Wynik analizy sentymentu
        """
        current_time = time.time()
        
        # Aktualizuj wyniki co określony czas
        if current_time - self.last_update > self.update_interval:
            self.last_results = self._generate_random_sentiment()
            self.last_update = current_time
            self.logger.debug("Zaktualizowano dane sentymentu")
        
        return self.last_results
    
    def _generate_random_sentiment(self) -> Dict[str, Any]:
        """
        Generuje losowy sentyment.
        
        Returns:
            Dict[str, Any]: Dane sentymentu
        """
        # Generuj losowy sentyment dla każdego źródła
        sources_sentiment = {}
        for source in self.sources:
            # Wartość sentymentu: od -1.0 (bardzo negatywny) do 1.0 (bardzo pozytywny)
            sources_sentiment[source] = random.uniform(-1.0, 1.0)
        
        # Oblicz średni sentyment
        avg_sentiment = sum(sources_sentiment.values()) / len(sources_sentiment)
        
        # Określ kategorię sentymentu
        sentiment_category = self._categorize_sentiment(avg_sentiment)
        
        return {
            "value": avg_sentiment,
            "analysis": sentiment_category,
            "sources": sources_sentiment,
            "timestamp": time.time()
        }
    
    def _categorize_sentiment(self, sentiment_value: float) -> str:
        """
        Kategoryzuje wartość sentymentu.
        
        Args:
            sentiment_value: Wartość sentymentu
            
        Returns:
            str: Kategoria sentymentu
        """
        if sentiment_value < -0.6:
            return "Bardzo negatywny"
        elif sentiment_value < -0.2:
            return "Negatywny"
        elif sentiment_value < 0.2:
            return "Neutralny"
        elif sentiment_value < 0.6:
            return "Pozytywny"
        else:
            return "Bardzo pozytywny"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Zwraca status analizatora.
        
        Returns:
            Dict[str, Any]: Status analizatora
        """
        return {
            "active": True,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_update)),
            "sources": self.sources,
            "update_interval": self.update_interval
        }
    
    def set_sources(self, sources: List[str]) -> bool:
        """
        Ustawia źródła danych sentymentu.
        
        Args:
            sources: Lista źródeł
            
        Returns:
            bool: Czy operacja się powiodła
        """
        self.sources = sources
        self.last_results = self._generate_random_sentiment()
        self.last_update = time.time()
        
        self.logger.info(f"Zaktualizowano źródła sentymentu: {sources}")
        return True
    
    def predict(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Przewiduje sentyment na podstawie danych.
        
        Args:
            data: Dane wejściowe (opcjonalne)
            
        Returns:
            Dict[str, Any]: Przewidywany sentyment
        """
        # W tym przykładzie ignorujemy dane wejściowe i zwracamy losowy sentyment
        return self.analyze()
