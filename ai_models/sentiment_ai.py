
"""
sentiment_ai.py - Moduł analizujący sentyment rynkowy
"""

import random
import logging
import os
import json
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Klasa do analizy sentymentu rynkowego z wielu źródeł"""
    
    def __init__(self, sources: List[str] = None, cache_dir: str = "data/cache"):
        """
        Inicjalizacja analizatora sentymentu.
        
        Args:
            sources: Lista źródeł danych
            cache_dir: Katalog cache dla danych sentymentu
        """
        self.default_sources = ["twitter", "news", "reddit", "forum"]
        self.sources = sources if sources else self.default_sources
        self.cache_dir = cache_dir
        self.last_update = datetime.now() - timedelta(hours=1)  # Ustawienie czasu, aby wymusić aktualizację
        self.update_interval = 1800  # 30 minut
        self.sentiment_data = None
        
        # Stworzenie katalogu cache, jeśli nie istnieje
        os.makedirs(cache_dir, exist_ok=True)
        
        # Inicjalizacja na podstawie zapisu, jeśli istnieje
        self._load_cached_data()
        
        # Generowanie sentymentu przy uruchomieniu
        if not self.sentiment_data:
            self.update_sentiment()
        
        logger.info(f"SentimentAnalyzer zainicjalizowany z {len(self.sources)} źródłami")
    
    def _load_cached_data(self):
        """Ładuje zapisane dane sentymentu z pliku cache"""
        try:
            cache_file = os.path.join(self.cache_dir, "sentiment_data.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Konwersja timestamp na obiekt datetime
                if "timestamp" in data:
                    timestamp = datetime.fromisoformat(data["timestamp"])
                    # Używaj zapisanych danych tylko jeśli są wystarczająco świeże
                    if (datetime.now() - timestamp).total_seconds() < 3600:  # maksymalnie 1 godzina
                        self.sentiment_data = data
                        self.last_update = timestamp
                        logger.info(f"Załadowano dane sentymentu z cache (timestamp: {timestamp})")
                    else:
                        logger.info(f"Zapisane dane sentymentu są nieaktualne. Ostatnia aktualizacja: {timestamp}")
        except Exception as e:
            logger.error(f"Błąd podczas ładowania danych sentymentu z cache: {e}")
    
    def _save_to_cache(self, data: Dict[str, Any]):
        """Zapisuje dane sentymentu do pliku cache"""
        try:
            cache_file = os.path.join(self.cache_dir, "sentiment_data.json")
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Zapisano dane sentymentu do {cache_file}")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania danych sentymentu: {e}")

    def _generate_source_sentiment(self, source: str, current_btc_trend: float) -> Tuple[float, int]:
        """
        Generuje wartość sentymentu dla danego źródła bazując na trendzie BTC.
        
        Args:
            source: Nazwa źródła
            current_btc_trend: Wartość trendu BTC (-1 do 1)
            
        Returns:
            Tuple[float, int]: (wartość sentymentu, liczba wzmianek)
        """
        # Bazowe odchylenie dla różnych źródeł
        source_bias = {
            "twitter": 0.05,     # Twitter ma tendencję do bycia lekko pozytywnym
            "reddit": -0.07,     # Reddit ma tendencję do bycia lekko negatywnym
            "news": 0.02,        # Wiadomości są bardziej neutralne
            "forum": -0.03,      # Fora mają lekko negatywny odcień
            "stocktwits": 0.08,  # StockTwits jest bardziej pozytywne
            "telegram": 0.1,     # Telegram grupy są zwykle bardzo pozytywne
            "discord": 0.07,     # Discord podobnie
            "youtube": 0.03      # YouTube komentarze są mieszane
        }
        
        # Domyślne odchylenie, jeżeli nie ma zdefiniowanego
        bias = source_bias.get(source.lower(), 0)
        
        # Generuj wartość sentymentu bazując na trendzie BTC + losowa zmienność i odchylenie źródła
        variance = random.uniform(-0.15, 0.15)  # Losowa zmienność
        sentiment_value = current_btc_trend + variance + bias
        
        # Przytnij wartość do przedziału [-1, 1]
        sentiment_value = max(-1, min(1, sentiment_value))
        
        # Liczba wzmianek zależy od typu źródła
        if source.lower() in ["twitter", "reddit"]:
            mentions = random.randint(800, 10000)  # Duże platformy
        elif source.lower() in ["news"]:
            mentions = random.randint(50, 500)     # Źródła informacyjne
        else:
            mentions = random.randint(100, 1500)   # Inne źródła
        
        return sentiment_value, mentions
    
    def update_sentiment(self) -> Dict[str, Any]:
        """
        Aktualizuje dane sentymentu generując realistyczne wartości.
        
        Returns:
            Dict[str, Any]: Dane sentymentu
        """
        # Symulacja trendu BTC - wartość bazowa od -0.5 do 0.5
        # Można to zastąpić rzeczywistymi danymi o trendzie
        base_btc_trend = random.uniform(-0.5, 0.5)
        
        # Generowanie sentymentu dla każdego źródła
        sources_data = {}
        overall_sentiment = 0
        total_mentions = 0
        
        for source in self.sources:
            sentiment_value, mentions = self._generate_source_sentiment(source, base_btc_trend)
            sources_data[source] = {
                "score": sentiment_value,
                "volume": mentions
            }
            
            # Uwzględnij wagę źródła na podstawie liczby wzmianek
            overall_sentiment += sentiment_value * mentions
            total_mentions += mentions
        
        # Obliczenie średniego sentymentu ważonego liczbą wzmianek
        if total_mentions > 0:
            overall_sentiment /= total_mentions
        
        # Analiza słowna
        analysis = self._get_textual_analysis(overall_sentiment)
        
        # Utwórz kompletny obiekt danych
        self.sentiment_data = {
            "value": overall_sentiment,
            "analysis": analysis,
            "sources": sources_data,
            "timestamp": datetime.now().isoformat(),
            "time_range": "ostatnie 24 godziny"
        }
        
        # Zapisz do cache
        self._save_to_cache(self.sentiment_data)
        
        # Aktualizacja znacznika czasu
        self.last_update = datetime.now()
        
        return self.sentiment_data
    
    def _get_textual_analysis(self, sentiment_value: float) -> str:
        """
        Mapuje wartość liczbową sentymentu na analizę tekstową.
        
        Args:
            sentiment_value: Wartość sentymentu od -1 do 1
            
        Returns:
            str: Analiza słowna
        """
        if sentiment_value > 0.7:
            return "Skrajnie pozytywny"
        elif sentiment_value > 0.4:
            return "Bardzo pozytywny"
        elif sentiment_value > 0.2:
            return "Pozytywny"
        elif sentiment_value > 0.05:
            return "Lekko pozytywny"
        elif sentiment_value > -0.05:
            return "Neutralny"
        elif sentiment_value > -0.2:
            return "Lekko negatywny"
        elif sentiment_value > -0.4:
            return "Negatywny"
        elif sentiment_value > -0.7:
            return "Bardzo negatywny"
        else:
            return "Skrajnie negatywny"
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analizuje sentyment rynkowy i zwraca wyniki.
        
        Returns:
            Dict[str, Any]: Dane sentymentu
        """
        # Sprawdź czy musimy zaktualizować dane
        time_diff = (datetime.now() - self.last_update).total_seconds()
        if time_diff > self.update_interval or self.sentiment_data is None:
            logger.info(f"Aktualizacja danych sentymentu (ostatnia: {self.last_update.strftime('%H:%M:%S')})")
            self.update_sentiment()
        
        return self.sentiment_data
    
    def get_status(self) -> Dict[str, Any]:
        """
        Zwraca status komponentu.
        
        Returns:
            Dict[str, Any]: Status analizatora sentymentu
        """
        return {
            "active": True,
            "sources": self.sources,
            "last_update": self.last_update.strftime("%Y-%m-%d %H:%M:%S"),
            "update_interval": self.update_interval
        }

# Inicjalizuj analizator sentymentu przy imporcie
sentiment_analyzer = SentimentAnalyzer()
