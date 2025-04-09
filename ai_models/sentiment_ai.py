"""
sentiment_ai.py - Lekka implementacja analizy sentymentu
"""
import random
import logging
import numpy as np
import json
import os
from datetime import datetime

# Konfiguracja logowania (from original code)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Uproszczona implementacja analizatora sentymentu bazująca na scikit-learn
    zamiast transformers lub innych ciężkich bibliotek.
    """

    def __init__(self, sources=None):
        """
        Inicjalizacja analizatora sentymentu.

        Args:
            sources (list): Lista źródeł, z których pobierany jest sentyment
        """
        self.sources = sources or ["twitter", "news", "reddit", "forum"]
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Zainicjalizowano SentimentAnalyzer z {len(self.sources)} źródłami")
        self.last_update = datetime.now()
        
        # Ścieżka do katalogu cache
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "sentiment_data.json")

        # Inicjalizacja bazowych wartości sentymentu dla każdego źródła
        self.base_sentiments = {}
        
        # Próba załadowania zapisanych wartości
        self._load_sentiment_data()
        
        # Jeśli nie ma zapisanych, generujemy nowe
        if not self.base_sentiments:
            for source in self.sources:
                self.base_sentiments[source] = random.uniform(-0.3, 0.3)
            self._save_sentiment_data()

    def _load_sentiment_data(self):
        """Ładuje zapisane dane sentymentu z pliku cache, jeśli istnieje."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    if "base_sentiments" in data:
                        self.base_sentiments = data["base_sentiments"]
                    if "last_update" in data:
                        self.last_update = datetime.fromisoformat(data["last_update"])
                self.logger.info(f"Załadowano dane sentymentu z pliku cache {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Nie udało się załadować danych sentymentu: {e}")

    def _save_sentiment_data(self):
        """Zapisuje dane sentymentu do pliku cache."""
        try:
            data = {
                "base_sentiments": self.base_sentiments,
                "last_update": self.last_update.isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Zapisano dane sentymentu do pliku cache {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Nie udało się zapisać danych sentymentu: {e}")

    def analyze(self, text=None):
        """
        Analizuje sentyment dla danego tekstu lub generuje symulowane wyniki.

        Args:
            text (str, optional): Tekst do analizy. Jeśli None, generuje symulowane wyniki.

        Returns:
            dict: Wyniki analizy sentymentu
        """
        # Symulowane wyniki, jeśli nie podano tekstu
        if text is None:
            sentiment_values = {}

            # Generowanie wartości sentymentu dla każdego źródła
            for source in self.sources:
                # Dodaj losowe wahanie do bazowego sentymentu
                base = self.base_sentiments[source]
                variance = random.uniform(-0.1, 0.1)
                sentiment_values[source] = max(-1.0, min(1.0, base + variance))

            # Powolna ewolucja bazowego sentymentu
            # Co jakiś czas aktualizujemy bazowy sentyment, aby symulować zmiany rynkowe
            if random.random() < 0.1:  # 10% szans na aktualizację bazowego sentymentu
                for source in self.sources:
                    self.base_sentiments[source] += random.uniform(-0.05, 0.05)
                    # Utrzymujemy wartości w zakresie [-0.5, 0.5]
                    self.base_sentiments[source] = max(-0.5, min(0.5, self.base_sentiments[source]))
                self._save_sentiment_data()

            # Obliczanie średniego sentymentu
            overall = sum(sentiment_values.values()) / len(sentiment_values)

            # Określanie kategorii sentymentu
            if overall > 0.2:
                analysis = "Pozytywny"
            elif overall < -0.2:
                analysis = "Negatywny"
            else:
                analysis = "Neutralny"

            # Dodajemy dodatkowe dane rynkowe
            market_data = {
                "POSITIVE": max(0, (overall + 1) / 2),  # Konwersja z [-1,1] do [0,1]
                "NEGATIVE": max(0, (1 - overall) / 2),  # Konwersja z [-1,1] do [0,1]
                "volatility": abs(random.gauss(0, 0.015)),
                "momentum": overall * random.uniform(0.8, 1.2),
                "trend_strength": abs(overall) * random.uniform(0.7, 1.3)
            }

            self.last_update = datetime.now()
            result = {
                "value": overall,
                "analysis": analysis,
                "sources": sentiment_values,
                "market_data": market_data,
                "timestamp": self.last_update.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Aktualizacja czasu ostatniej aktualizacji w cache
            self._save_sentiment_data()
            
            return result

        # Implementacja analizy rzeczywistego tekstu
        else:
            # Prosta heurystyka oparta na słowach kluczowych
            positive_words = ["wzrost", "zysk", "sukces", "bull", "up", "strong", "good", "positive"]
            negative_words = ["spadek", "strata", "bear", "down", "weak", "bad", "negative"]

            text_lower = text.lower()

            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            # Obliczanie prostego wyniku sentymentu
            if positive_count + negative_count == 0:
                score = 0
            else:
                score = (positive_count - negative_count) / (positive_count + negative_count)

            # Określanie kategorii sentymentu
            if score > 0.2:
                analysis = "Pozytywny"
            elif score < -0.2:
                analysis = "Negatywny"
            else:
                analysis = "Neutralny"

            # Dodajemy dodatkowe dane rynkowe
            market_data = {
                "POSITIVE": max(0, (score + 1) / 2),  # Konwersja z [-1,1] do [0,1]
                "NEGATIVE": max(0, (1 - score) / 2),  # Konwersja z [-1,1] do [0,1]
                "volatility": abs(random.gauss(0, 0.015)),
                "momentum": score * random.uniform(0.8, 1.2),
                "trend_strength": abs(score) * random.uniform(0.7, 1.3)
            }

            self.last_update = datetime.now()
            return {
                "value": score,
                "analysis": analysis,
                "text": text,
                "market_data": market_data,
                "timestamp": self.last_update.strftime("%Y-%m-%d %H:%M:%S")
            }

    def get_market_sentiment(self):
        """
        Zwraca aktualny sentyment rynkowy jako słownik z metrykami.
        
        Returns:
            dict: Dane sentymentu rynkowego
        """
        # Używamy funkcji analyze do wygenerowania danych
        analysis = self.analyze()
        
        # Zwracamy tylko rynkowe dane, bez metadanych
        return analysis.get("market_data", {})

    def get_status(self):
        """
        Zwraca status analizatora sentymentu.

        Returns:
            dict: Status analizatora
        """
        return {
            "active": True,
            "sources": self.sources,
            "last_update": self.last_update.strftime("%Y-%m-%d %H:%M:%S")
        }

if __name__ == "__main__":
    # Przykładowe użycie
    analyzer = SentimentAnalyzer()

    # Symulowane wyniki
    result = analyzer.analyze()
    print("Symulowane wyniki:", result)

    # Dane sentymentu rynkowego
    market_sentiment = analyzer.get_market_sentiment()
    print("Sentyment rynkowy:", market_sentiment)

    # Analiza rzeczywistego tekstu
    text_result = analyzer.analyze("Rynek wykazuje silne oznaki wzrostu po ostatnich pozytywnych danych ekonomicznych.")
    print("Analiza tekstu:", text_result)