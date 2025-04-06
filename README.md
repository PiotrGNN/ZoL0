
# 🚀 Trading Bot - System Analityczny

## 📝 Opis projektu
System do analizy danych rynkowych, strategii handlowych i automatycznego handlu, zoptymalizowany do działania w środowisku Replit.

## 🔧 Funkcjonalności
- Analiza danych rynkowych 
- Wykrywanie anomalii cenowych
- Symulacja strategii tradingowych
- Zarządzanie ryzykiem i kapitałem
- Modele uczenia maszynowego do predykcji

## 🏗️ Struktura projektu
```
├── ai_models/              # Modele AI i uczenia maszynowego
├── config/                 # Pliki konfiguracyjne
├── data/                   # Moduły do obsługi danych
│   ├── data/               # Pobieranie i przetwarzanie danych
│   ├── execution/          # Wykonywanie zleceń
│   ├── indicators/         # Wskaźniki techniczne
│   ├── logging/            # Komponenty logowania
│   ├── optimization/       # Optymalizacja strategii
│   ├── risk_management/    # Zarządzanie ryzykiem
│   ├── strategies/         # Strategie handlowe
│   ├── tests/              # Testy jednostkowe
│   └── utils/              # Narzędzia pomocnicze
├── logs/                   # Pliki logów
├── reports/                # Generowane raporty i wizualizacje
├── saved_models/           # Zapisane modele ML
└── main.py                 # Główny punkt wejścia aplikacji
```

## 🚀 Jak uruchomić
1. Upewnij się, że wszystkie zależności są zainstalowane:
   ```
   pip install -r requirements.txt
   ```

2. Utwórz plik `.env` na podstawie `.env.example`:
   ```
   cp .env.example .env
   ```

3. Uruchom aplikację:
   ```
   python3 main.py
   ```

4. Uruchom testy:
   ```
   pytest data/tests/
   ```

## 🔧 Narzędzia developerskie
- **Linter**: flake8 - sprawdzanie zgodności ze standardami kodu
  ```
  flake8 .
  ```

- **Formatter**: black - automatyczne formatowanie kodu
  ```
  black .
  ```

- **Testy**: pytest - uruchamianie testów jednostkowych
  ```
  pytest
  ```

## 📦 Zależności
Wszystkie wymagane biblioteki są wymienione w pliku `requirements.txt`.

## 📝 Konfiguracja
- Ustawienia można modyfikować w katalogu `config/`
- Parametry środowiskowe znajdują się w pliku `.env`
- W pliku `.replit` znajduje się konfiguracja środowiska Replit

## 📊 Przykładowe użycie
```python
from ai_models.anomaly_detection import AnomalyDetectionModel

# Inicjalizacja modelu wykrywania anomalii
detector = AnomalyDetectionModel()

# Wykrywanie anomalii w danych cenowych
anomalies = detector.detect_price_anomalies(price_data)
```

## 🔒 Bezpieczeństwo
- Klucze API są przechowywane w zmiennych środowiskowych
- Używaj trybu testowego przed przejściem do rzeczywistego handlu

## 👥 Współtwórcy
Lista osób, które przyczyniły się do rozwoju projektu.

## 📄 Licencja
Ten projekt jest licencjonowany - sprawdź plik LICENSE, aby uzyskać szczegóły.
