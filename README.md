# 🤖 Trading Bot - System Analityczny

## 📋 Opis projektu

Trading Bot to zaawansowany system analityczny do analizy danych rynkowych, wykrywania anomalii i testowania strategii tradingowych. System wykorzystuje metody uczenia maszynowego do analizy wzorców cenowych i wolumenowych.

## 🚀 Szybki start

### Przygotowanie środowiska

1. Sklonuj repozytorium
2. W środowisku Replit system automatycznie zainstaluje wymagane zależności
3. Utwórz plik `.env` na podstawie `.env.example` i ustaw swoje klucze API

### Uruchomienie systemu

Możesz uruchomić system na dwa sposoby:

1. **Przycisk Run** - Kliknij przycisk "Run" w panelu Replit
2. **Z linii poleceń** - Uruchom `python main.py`

## 🧰 Dostępne workflows w Replit

- **Start Trading Bot** - Uruchamia główny system
- **Run Tests** - Wykonuje testy jednostkowe
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
python -m pytest data/tests
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
3. Uruchom testy przed wysłaniem zmian: `python -m pytest`
4. Formatuj kod za pomocą Black: `black .`
5. Sprawdź zgodność z PEP8: `flake8 .`

## 📝 Licencja

Ten projekt jest udostępniony na licencji MIT.