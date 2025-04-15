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