"""
sentiment_analysis.py
--------------------
Moduł do analizy sentymentu z różnych źródeł (media społecznościowe, newsy, raporty)
w celu określenia nastrojów rynkowych.
"""

import logging
import re
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk_available = True
except ImportError:
    nltk_available = False
    logging.warning("Biblioteka NLTK nie jest dostępna. Używam symulowanych danych sentymentu.")


class SentimentAnalyzer:
    """
    Klasa analizująca sentyment rynkowy na podstawie danych tekstowych.
    Wykorzystuje modele NLP (NLTK VADER) jeśli są dostępne, w przeciwnym razie generuje dane symulacyjne.
    """

    def __init__(self, sources=None):
        """
        Inicjalizacja analizatora sentymentu.

        Args:
            sources (list): Źródła danych (np. Twitter, Reddit, itp.)
        """
        self.sources = sources or ["twitter", "reddit", "news", "forum"]
        self.data = None
        self.history = []
        self.last_update = datetime.now()
        self.update_interval = 3600  # Domyślnie co godzinę
        
    def get_current_sentiment(self):
        """
        Zwraca aktualny sentyment rynkowy w formacie tekstowym.
        
        Returns:
            str: Tekstowy opis aktualnego sentymentu rynku
        """
        # Pobieramy ostatnią analizę lub generujemy nową
        if not self.history:
            analysis = self._generate_simulated_data("1d")
        else:
            analysis = self.history[-1]
            
        # Określamy sentyment na podstawie wyniku
        score = analysis.get("overall_score", 0) if isinstance(analysis, dict) else 0
        
        if score > 0.5:
            return "Zdecydowanie pozytywny"
        elif score > 0.2:
            return "Pozytywny"
        elif score > -0.2:
            return "Neutralny"
        elif score > -0.5:
            return "Negatywny"
        else:
            return "Zdecydowanie negatywny"

        # Inicjalizacja NLTK VADER jeśli jest dostępny
        self.nltk_analyzer = None
        if nltk_available:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.nltk_analyzer = SentimentIntensityAnalyzer()
                logging.info("Zainicjalizowano analizator sentymentu NLTK VADER")
            except Exception as e:
                logging.error(f"Błąd inicjalizacji NLTK VADER: {str(e)}")

        logging.info(f"Inicjalizacja analizatora sentymentu z {len(self.sources)} źródłami.")

    def analyze(self, text=None, timeframe="1d", force_update=False):
        """
        Analizuje sentyment w podanym tekście lub generuje symulowane dane.

        Args:
            text (str, optional): Tekst do analizy
            timeframe (str): Zakres czasowy analizy
            force_update (bool): Czy wymusić aktualizację danych

        Returns:
            dict: Wyniki analizy sentymentu
        """
        # Sprawdzenie czy potrzebna jest aktualizacja
        current_time = datetime.now()
        if not force_update and self.history and (current_time - self.last_update).total_seconds() < self.update_interval:
            # Zwraca ostatnią analizę bez aktualizacji
            return self._get_latest_sentiment(timeframe)

        self.last_update = current_time

        if text:
            result = self._analyze_text(text)
        else:
            result = self._generate_simulated_data(timeframe)

        # Dodanie czasowych metadanych
        result["timestamp"] = current_time.isoformat()
        result["next_update"] = (current_time + timedelta(seconds=self.update_interval)).isoformat()

        return result

    def _analyze_text(self, text):
        """
        Analizuje sentyment tekstu przy użyciu NLTK VADER lub symulacji.

        Args:
            text (str): Tekst do analizy

        Returns:
            dict: Wyniki analizy sentymentu
        """
        # Używamy NLTK VADER jeśli jest dostępny
        if self.nltk_analyzer:
            try:
                scores = self.nltk_analyzer.polarity_scores(text)
                sentiment_score = scores['compound']  # Wynik całościowy, zakres -1 do 1

                # Klasyfikacja sentymentu
                if sentiment_score > 0.05:
                    sentiment_label = "pozytywny"
                elif sentiment_score < -0.05:
                    sentiment_label = "negatywny"
                else:
                    sentiment_label = "neutralny"

                return {
                    "score": sentiment_score,
                    "label": sentiment_label,
                    "statistics": {
                        "word_count": len(text.split()),
                        "positive": scores['pos'],
                        "negative": scores['neg'],
                        "neutral": scores['neu']
                    },
                    "method": "nltk_vader"
                }
            except Exception as e:
                logging.warning(f"Błąd analizy sentymentu NLTK: {str(e)}, używam analizy heurystycznej")

        # Symulacja rzeczywistej analizy NLP (proste podejście heurystyczne)
        word_count = len(re.findall(r'\w+', text))

        # Proste słowa kluczowe dla demonstracji
        bullish_words = ["wzrost", "zysk", "dobry", "optymistyczny", "bull", "long", "buy", "bullish", "mocny"]
        bearish_words = ["spadek", "strata", "zły", "pesymistyczny", "bear", "short", "sell", "bearish", "słaby"]

        bullish_count = sum(1 for word in bullish_words if word.lower() in text.lower())
        bearish_count = sum(1 for word in bearish_words if word.lower() in text.lower())

        # Obliczanie wyniku sentymentu (-1 do 1)
        if word_count == 0:
            sentiment_score = 0
        else:
            sentiment_score = (bullish_count - bearish_count) / (word_count * 0.2 + 1)  # +1 aby uniknąć dzielenia przez 0
            sentiment_score = max(min(sentiment_score, 1), -1)  # Ograniczenie do zakresu [-1, 1]

        # Klasyfikacja sentymentu
        if sentiment_score > 0.1:
            sentiment_label = "pozytywny"
        elif sentiment_score < -0.1:
            sentiment_label = "negatywny"
        else:
            sentiment_label = "neutralny"

        return {
            "score": sentiment_score,
            "label": sentiment_label,
            "statistics": {
                "word_count": word_count,
                "bullish_words": bullish_count,
                "bearish_words": bearish_count
            },
            "method": "heuristic"
        }

    def _generate_simulated_data(self, timeframe):
        """
        Generuje symulowane dane sentymentu.

        Args:
            timeframe (str): Zakres czasowy analizy

        Returns:
            dict: Symulowane wyniki analizy sentymentu
        """
        # Generujemy losowy sentyment, ale z pewną korelacją z poprzednimi wartościami
        seed = int((datetime.now().timestamp() + hash(timeframe)) % 10000)
        np.random.seed(seed)

        # Baza sentymentu - lekko pozytywna
        base_sentiment = 0.05

        # Jeśli mamy historię, dodajemy korelację z poprzednimi wartościami
        if self.history:
            last_sentiments = [h["overall_score"] for h in self.history[-5:]]
            mean_sentiment = np.mean(last_sentiments)
            # 40% poprzednich danych + 60% nowych
            base_sentiment = 0.4 * mean_sentiment + 0.6 * np.random.normal(0, 0.4)
        else:
            base_sentiment = np.random.normal(0, 0.4)

        # Ograniczamy do zakresu -1 do 1
        base_sentiment = max(min(base_sentiment, 1), -1)

        # Generujemy sentymenty dla różnych źródeł
        sentiments = {}
        for source in self.sources:
            # Losowe odchylenie od bazowego sentymentu
            source_offset = np.random.normal(0, 0.2)
            source_sentiment = base_sentiment + source_offset
            source_sentiment = max(min(source_sentiment, 1), -1)  # Ograniczenie do zakresu [-1, 1]

            # Przypisujemy wagę źródła (symulacja wpływu na rynek)
            if source == "twitter":
                weight = np.random.uniform(0.8, 1.2)
            elif source == "news":
                weight = np.random.uniform(1.0, 1.5)
            else:
                weight = np.random.uniform(0.6, 1.0)

            # Symulujemy objętość danych
            volume = int(np.random.gamma(10, 200) * weight)

            sentiments[source] = {
                "score": source_sentiment,
                "weight": weight,
                "volume": volume,
                "change": source_sentiment - (self.history[-1]["sources"][source]["score"] if self.history else 0)
            }

        # Obliczamy ogólny wynik jako średnią ważoną
        total_volume = sum(src["volume"] for src in sentiments.values())
        if total_volume > 0:
            overall_score = sum(src["score"] * src["volume"] for src in sentiments.values()) / total_volume
        else:
            overall_score = base_sentiment

        # Dodajemy do historii
        self.history.append({
            "timestamp": pd.Timestamp.now(),
            "overall_score": overall_score,
            "sources": sentiments,
            "timeframe": timeframe
        })

        # Ograniczamy historię do ostatnich 100 punktów
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return {
            "overall_score": overall_score,
            "sources": sentiments,
            "analysis": self._get_market_sentiment_description(overall_score),
            "timeframe": timeframe,
            "method": "simulation",
            "trend": self._calculate_sentiment_trend()
        }

    def _calculate_sentiment_trend(self):
        """
        Oblicza trend sentymentu na podstawie historycznych danych.

        Returns:
            dict: Informacje o trendzie sentymentu
        """
        if len(self.history) < 2:
            return {"direction": "neutralny", "strength": 0, "description": "Za mało danych do określenia trendu"}

        # Obliczamy zmianę w czasie
        last_scores = [h["overall_score"] for h in self.history[-10:]] if len(self.history) >= 10 else [h["overall_score"] for h in self.history]

        if len(last_scores) >= 3:
            # Prosta regresja liniowa dla trendu
            x = np.arange(len(last_scores))
            slope, _, _, _, _ = np.polyfit(x, last_scores, 1, full=True)
            slope = slope[0]
        else:
            # Prosta różnica
            slope = last_scores[-1] - last_scores[0]

        # Skala siły trendu
        trend_strength = abs(slope) * 10  # Skalujemy dla czytelności
        trend_strength = min(trend_strength, 1.0)  # Ograniczenie do 1.0

        # Kierunek trendu
        if abs(slope) < 0.01:
            trend_direction = "neutralny"
            trend_description = "Stabilny sentyment rynkowy"
        elif slope > 0:
            trend_direction = "rosnący"
            if trend_strength > 0.7:
                trend_description = "Silnie rosnący sentyment - możliwe FOMO"
            elif trend_strength > 0.3:
                trend_description = "Umiarkowanie rosnący sentyment"
            else:
                trend_description = "Lekko pozytywny trend sentymentu"
        else:
            trend_direction = "spadkowy"
            if trend_strength > 0.7:
                trend_description = "Gwałtownie spadający sentyment - możliwa panika"
            elif trend_strength > 0.3:
                trend_description = "Wyraźnie spadający sentyment"
            else:
                trend_description = "Lekko negatywny trend sentymentu"

        return {
            "direction": trend_direction,
            "strength": float(trend_strength),
            "description": trend_description,
            "raw_slope": float(slope)
        }

    def _get_market_sentiment_description(self, score):
        """
        Zwraca opis słowny sentymentu rynkowego.

        Args:
            score (float): Wynik sentymentu (-1 do 1)

        Returns:
            str: Opis słowny sentymentu
        """
        if score > 0.7:
            return "Ekstremalnie byczo nastawiony rynek"
        elif score > 0.4:
            return "Silnie pozytywny sentyment"
        elif score > 0.1:
            return "Umiarkowanie pozytywny sentyment"
        elif score > -0.1:
            return "Neutralny sentyment rynkowy"
        elif score > -0.4:
            return "Umiarkowanie negatywny sentyment"
        elif score > -0.7:
            return "Silnie negatywny sentyment"
        else:
            return "Ekstremalnie niedźwiedzi sentyment rynkowy"

    def _get_latest_sentiment(self, timeframe):
        """
        Zwraca najnowsze dane sentymentu dla danego timeframe.

        Args:
            timeframe (str): Zakres czasowy

        Returns:
            dict: Najnowsze dane sentymentu
        """
        if not self.history:
            # Jeśli nie ma danych, generujemy nowe
            return self._generate_simulated_data(timeframe)

        # Szukamy najnowszego sentymentu dla danego timeframe
        matching_sentiments = [h for h in self.history if h.get("timeframe") == timeframe]

        if matching_sentiments:
            latest = matching_sentiments[-1]

            # Dodanie czasowych metadanych
            return {
                "overall_score": latest["overall_score"],
                "sources": latest["sources"],
                "analysis": self._get_market_sentiment_description(latest["overall_score"]),
                "timeframe": timeframe,
                "timestamp": latest["timestamp"].isoformat() if isinstance(latest["timestamp"], pd.Timestamp) else latest["timestamp"],
                "next_update": (self.last_update + timedelta(seconds=self.update_interval)).isoformat(),
                "is_cached": True,
                "trend": self._calculate_sentiment_trend()
            }
        else:
            # Jeśli nie ma danych dla tego timeframe, generujemy nowe
            return self._generate_simulated_data(timeframe)

# Przykładowe użycie modułu
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Analiza pojedynczego tekstu
    result = analyzer.analyze(text="Bitcoin może osiągnąć nowy ATH w tym roku!")
    print(f"Sentyment: {result['label']}, Score: {result['score']:.2f}, Method: {result['method']}")

    # Pobieranie sentymentu rynkowego
    market_sentiment = analyzer.analyze()
    print(f"Sentyment rynkowy: {market_sentiment['analysis']}, Score: {market_sentiment['overall_score']:.2f}, Trend: {market_sentiment['trend']}")

    #Wymusić aktualizacje danych
    market_sentiment = analyzer.analyze(force_update=True)
    print(f"Sentyment rynkowy (wymuś aktualizację): {market_sentiment['analysis']}, Score: {market_sentiment['overall_score']:.2f}, Trend: {market_sentiment['trend']}")