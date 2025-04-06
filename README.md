
# 🚀 Advanced AI Trading System

System do automatycznego tradingu z wykorzystaniem zaawansowanych algorytmów AI i ML.

## 📋 Funkcje

- 🤖 **AI/ML Trading** - Wykorzystanie modeli uczenia maszynowego do predykcji ruchu cen
- 📊 **Wykrywanie anomalii** - Identyfikacja nietypowych wzorców rynkowych
- 📈 **Optymalizacja strategii** - Automatyczne dostosowanie parametrów strategii
- 🔄 **Backtesting** - Testowanie strategii na danych historycznych
- 🛡️ **Zarządzanie ryzykiem** - Zaawansowane mechanizmy kontroli ryzyka

## 🚀 Szybki start (Replit)

1. Skonfiguruj plik `.env` (skopiuj z `.env.example`):
   ```
   cp .env.example .env
   ```

2. Kliknij przycisk ▶️ **Run** lub wybierz workflow "Start Trading Bot" z menu

## 🔧 Konfiguracja środowiska

1. Zainstaluj wymagane zależności:
   ```
   pip install -r requirements.txt
   ```

2. Uruchom system tradingowy:
   ```
   python main.py
   ```

3. Dostępne workflow:
   - **Start Trading Bot** - Uruchamia główny system tradingowy
   - **Run Tests** - Uruchamia testy jednostkowe
   - **Lint Code** - Sprawdza kod źródłowy linterem

## 📁 Struktura projektu

```
.
├── ai_models/                # Modele AI/ML
│   ├── anomaly_detection.py  # Wykrywanie anomalii rynkowych
│   ├── reinforcement_learning.py # Uczenie ze wzmocnieniem
│   └── sentiment_analysis.py # Analiza sentymentu rynkowego
├── data/
│   ├── execution/            # Wykonywanie transakcji
│   ├── indicators/           # Wskaźniki techniczne
│   ├── risk_management/      # Zarządzanie ryzykiem
│   ├── strategies/           # Strategie tradingowe
│   └── tests/                # Testy jednostkowe
├── config/                   # Konfiguracja systemu
├── logs/                     # Logi systemu
├── .env.example              # Przykładowy plik zmiennych środowiskowych
└── main.py                   # Punkt wejściowy systemu
```

## 🧪 Testowanie

Aby uruchomić testy jednostkowe, użyj workflowa "Run Tests" lub wykonaj:

```
pytest data/tests
```

## 📊 Dostępne strategie

1. **Mean Reversion** - Wykorzystuje powrót ceny do średniej
2. **Trend Following** - Wykorzystuje silne trendy rynkowe
3. **Breakout Strategy** - Wykrywa wybicia z konsolidacji
4. **AI Strategy** - Adaptacyjna strategia bazująca na uczeniu maszynowym

## 🔒 Bezpieczeństwo

- Nigdy nie przechowuj kluczy API bezpośrednio w kodzie
- Używaj pliku .env do przechowywania poufnych kluczy
- Ustaw odpowiednie limity ryzyka w konfiguracji

## 🛠 Rozwój projektu

1. Sklonuj repozytorium
2. Zainstaluj zależności developerskie: `pip install -r requirements.txt`
3. Uruchom testy przed wysłaniem zmian: `pytest data/tests`
4. Sprawdź jakość kodu z flake8: `flake8 .`
5. Sformatuj kod za pomocą black: `black .`

## 📄 Licencja

Ten projekt jest udostępniany na licencji MIT.
