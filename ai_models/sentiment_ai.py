"""
sentiment_ai.py
--------------
Moduł do analizy sentymentu rynkowego.
"""

import logging
import random
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analizator sentymentu rynkowego."""

    def __init__(self, sources: List[str] = None):
        """
        Inicjalizuje analizator sentymentu.

        Parameters:
            sources (List[str]): Lista źródeł danych do analizy
        """
        self.sources = sources or ["twitter", "news", "forum", "reddit"]
        self.last_update = time.time()
        self.cache_validity = 60  # ważność cache w sekundach
        self.cached_sentiment = None
        self.active = True
        logger.info(f"Zainicjalizowano SentimentAnalyzer ze źródłami: {self.sources}")

    def analyze(self) -> Dict[str, Any]:
        """
        Analizuje sentyment na podstawie różnych źródeł.

        Returns:
            Dict[str, Any]: Wyniki analizy
        """
        # Sprawdź, czy mamy ważne dane w cache
        current_time = time.time()
        if self.cached_sentiment and current_time - self.last_update < self.cache_validity:
            return self.cached_sentiment

        # Symulowany sentyment dla celów demonstracyjnych
        sentiment_values = {
            "twitter": random.uniform(-1.0, 1.0),
            "news": random.uniform(-1.0, 1.0),
            "forum": random.uniform(-1.0, 1.0),
            "reddit": random.uniform(-1.0, 1.0)
        }

        # Filtruj tylko wybrane źródła
        sentiment_sources = {source: sentiment_values.get(source, 0)
                            for source in self.sources if source in sentiment_values}

        # Oblicz średni sentyment ze wszystkich źródeł
        average_sentiment = sum(sentiment_sources.values()) / len(sentiment_sources) if sentiment_sources else 0

        # Określ tekstowy opis sentymentu
        if average_sentiment > 0.3:
            analysis = "Bardzo pozytywny"
        elif average_sentiment > 0.1:
            analysis = "Pozytywny"
        elif average_sentiment > -0.1:
            analysis = "Neutralny"
        elif average_sentiment > -0.3:
            analysis = "Negatywny"
        else:
            analysis = "Bardzo negatywny"

        # Zapisz wyniki w cache
        self.cached_sentiment = {
            "value": average_sentiment,
            "analysis": analysis,
            "sources": sentiment_sources,
            "timestamp": current_time
        }

        self.last_update = current_time
        return self.cached_sentiment

    def get_status(self) -> Dict[str, Any]:
        """
        Zwraca status analizatora.

        Returns:
            Dict[str, Any]: Status analizatora
        """
        return {
            "active": self.active,
            "sources": self.sources,
            "last_update": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_update))
        }

    def set_sources(self, sources: List[str]) -> bool:
        """
        Ustawia źródła danych.

        Parameters:
            sources (List[str]): Lista źródeł danych

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        try:
            self.sources = sources
            # Zresetuj cache po zmianie źródeł
            self.cached_sentiment = None
            logger.info(f"Zaktualizowano źródła danych: {sources}")
            return True
        except Exception as e:
            logger.error(f"Błąd podczas ustawiania źródeł danych: {e}")
            return False

    def activate(self) -> bool:
        """
        Aktywuje analizator.

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        self.active = True
        logger.info("Aktywowano analizator sentymentu")
        return True

    def deactivate(self) -> bool:
        """
        Deaktywuje analizator.

        Returns:
            bool: True jeśli operacja się powiodła, False w przeciwnym przypadku
        """
        self.active = False
        logger.info("Deaktywowano analizator sentymentu")
        return True

if __name__ == "__main__":
    # Przykładowe użycie
    analyzer = SentimentAnalyzer()

    # Analiza sentymentu
    result = analyzer.analyze()
    print("Analiza sentymentu:", result)

    # Status analizatora
    status = analyzer.get_status()
    print("Status analizatora:", status)

    # Zmiana źródeł danych
    analyzer.set_sources(["news", "reddit"])
    result = analyzer.analyze()
    print("Analiza sentymentu po zmianie źródeł:", result)

    # Aktywacja/deaktywacja analizatora
    analyzer.deactivate()
    status = analyzer.get_status()
    print("Status analizatora po deaktywacji:", status)
    analyzer.activate()
    status = analyzer.get_status()
    print("Status analizatora po aktywacji:", status)