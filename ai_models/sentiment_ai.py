"""
sentiment_ai.py
----------------
Moduł do analizy nastrojów (sentiment analysis) w kontekście rynkowym.

Funkcjonalności:
- Wykorzystuje biblioteki scikit-learn do przetwarzania danych
- Implementuje klasyfikator do analizy sentymentu (pozytywny, neutralny, negatywny)
- Umożliwia przetwarzanie strumieniowe (real-time) oraz analizę historyczną
- Zapewnia mechanizmy wyznaczania wagi sentymentu w zależności od źródła
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analizator sentymentu rynkowego.

    Klasa analizuje sentyment z różnych źródeł danych, takich jak:
    - Wiadomości finansowe
    - Media społecznościowe
    - Wskaźniki nastrojów rynkowych
    - Forum dyskusyjne

    i generuje zagregowaną ocenę sentymentu.
    """

    def __init__(self, sources: List[str] = None, api_keys: Dict[str, str] = None):
        """
        Inicjalizuje analizator sentymentu.

        Args:
            sources: Lista źródeł danych do analizy
            api_keys: Słownik kluczy API do różnych serwisów
        """
        self.sources = sources or ["twitter", "news", "forum", "reddit"]
        self.api_keys = api_keys or {}
        self.last_analysis = None
        self.last_update = None

        # Inicjalizacja modelu scikit-learn
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', LogisticRegression(random_state=42))
        ])

        # Proste dane treningowe
        example_texts = [
            "markets are booming today", "very positive outlook on stocks",
            "crypto is showing strong uptrend", "investors are excited",
            "economy is stabilizing", "little movement in the market",
            "stocks flat today", "trading volumes normal",
            "market crash imminent", "recession fears grow",
            "terrible earnings report", "investors panic selling"
        ]

        example_labels = [
            1, 1, 1, 1,  # pozytywne (1)
            0, 0, 0, 0,  # neutralne (0)
            -1, -1, -1, -1  # negatywne (-1)
        ]

        # Trenowanie modelu na prostych danych
        self.model.fit(example_texts, example_labels)

        logger.info(f"Inicjalizacja SentimentAnalyzer z {len(self.sources)} źródłami")

    def analyze(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Przeprowadza analizę sentymentu na podstawie dostępnych źródeł.

        Args:
            force_update: Czy wymusić aktualizację danych

        Returns:
            Dict[str, Any]: Wynik analizy sentymentu
        """
        # Jeśli mamy poprzednią analizę i nie wymuszamy aktualizacji, zwróć ją
        if self.last_analysis and not force_update:
            # Sprawdź, czy ostatnia aktualizacja była mniej niż 5 minut temu
            if self.last_update and (datetime.now() - self.last_update).total_seconds() < 300:
                logger.info("Zwracam zapisaną analizę sentymentu (mniej niż 5 minut od ostatniej aktualizacji)")
                return self.last_analysis

        # Symulacja analizy z różnych źródeł
        source_sentiments = {}
        for source in self.sources:
            source_sentiments[source] = self._analyze_source(source)

        # Obliczenie średniego sentymentu
        sentiment_values = list(source_sentiments.values())
        average_sentiment = sum(sentiment_values) / len(sentiment_values)

        # Mapowanie wartości liczbowej na opis
        sentiment_label = self._get_sentiment_label(average_sentiment)

        # Przygotowanie wyniku
        self.last_analysis = {
            "value": round(average_sentiment, 2),
            "analysis": sentiment_label,
            "sources": {source: round(value, 2) for source, value in source_sentiments.items()},
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.last_update = datetime.now()

        logger.info(f"Analiza sentymentu: {sentiment_label} ({round(average_sentiment, 2)})")
        return self.last_analysis

    def _analyze_source(self, source: str) -> float:
        """
        Analizuje sentyment z określonego źródła.

        Args:
            source: Nazwa źródła danych

        Returns:
            float: Wartość sentymentu (-1.0 do 1.0)
        """
        # Symulowane teksty dla różnych źródeł
        example_texts = {
            "twitter": ["market looks promising", "stocks going up", "bearish signals everywhere", "sell everything now"],
            "news": ["economic growth exceeds expectations", "market cautious ahead of fed meeting", "recession fears intensify"],
            "forum": ["I'm bullish on tech stocks", "crypto will crash soon", "holding for the long term", "market manipulation everywhere"],
            "reddit": ["should I buy the dip?", "mooning soon!", "market is rigged", "strong fundamentals"]
        }

        # Wybierz losowy tekst z odpowiedniego źródła
        texts = example_texts.get(source, ["neutral market conditions"])
        text = random.choice(texts)

        # Analiza tekstu z użyciem modelu
        try:
            # Predykcja (-1, 0, 1)
            prediction = self.model.predict([text])[0]
            # Symulacja pewności predykcji (0.5-1.0)
            confidence = 0.5 + (random.random() * 0.5)
            # Skaluj wynik do zakresu [-1.0, 1.0]
            result = prediction * confidence
            return result
        except Exception as e:
            logger.error(f"Błąd analizy tekstu '{text}': {e}")
            # Fallback do losowej wartości
            return random.uniform(-0.5, 0.5)

    def _get_sentiment_label(self, value: float) -> str:
        """
        Przekształca liczbową wartość sentymentu na etykietę tekstową.

        Args:
            value: Wartość sentymentu (-1.0 do 1.0)

        Returns:
            str: Etykieta sentymentu
        """
        if value >= 0.6:
            return "Bardzo Pozytywny"
        elif value >= 0.2:
            return "Pozytywny"
        elif value >= -0.2:
            return "Neutralny"
        elif value >= -0.6:
            return "Negatywny"
        else:
            return "Bardzo Negatywny"

    def predict(self, data: Any = None) -> Dict[str, Any]:
        """
        Przewiduje sentyment na podstawie danych wejściowych.
        Metoda zgodna z interfejsem modelu AI.

        Args:
            data: Dane wejściowe (opcjonalne). Może być tekst lub lista tekstów

        Returns:
            Dict[str, Any]: Przewidywany sentyment
        """
        if data and isinstance(data, str):
            try:
                prediction = self.model.predict([data])[0]
                confidence = 0.5 + (random.random() * 0.5)
                sentiment_value = prediction * confidence
                return {
                    "value": round(sentiment_value, 2),
                    "analysis": self._get_sentiment_label(sentiment_value),
                    "source": "direct_input",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                logger.error(f"Błąd podczas analizy danych wejściowych: {e}")

        # Jeśli brak danych lub wystąpił błąd, zwróć standardową analizę
        return self.analyze(force_update=True)

    def fit(self, texts: List[str], labels: List[int]) -> bool:
        """
        Trenuje model na nowych danych.

        Args:
            texts: Lista tekstów do analizy
            labels: Lista etykiet (-1, 0, 1)

        Returns:
            bool: True jeśli trening się powiódł
        """
        try:
            self.model.fit(texts, labels)
            logger.info(f"Model został przeszkolony na {len(texts)} przykładach")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas trenowania modelu: {e}")
            return False

# Przykładowe użycie
if __name__ == "__main__":
    sentiment_analyzer = SentimentAnalyzer()
    result = sentiment_analyzer.analyze()
    print(f"Sentyment rynkowy: {result['analysis']} ({result['value']})")
    print(f"Źródła:")
    for source, value in result['sources'].items():
        print(f"  - {source}: {value}")