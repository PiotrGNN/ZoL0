
# Inteligentny System Tradingowy

Kompleksowy system do analizy rynku, zarządzania ryzykiem i automatycznego tradingu z wykorzystaniem API Bybit.

## 📋 Struktura projektu

```
├── ai_models/            # Modele AI do analizy rynku
├── data/                 # Komponenty przetwarzania danych
│   ├── cache/            # Dane cache
│   ├── execution/        # Moduły wykonywania transakcji
│   ├── indicators/       # Wskaźniki techniczne
│   ├── logging/          # Logowanie operacji
│   ├── optimization/     # Optymalizacja strategii
│   ├── risk_management/  # Zarządzanie ryzykiem
│   ├── strategies/       # Strategie tradingowe
│   ├── tests/            # Testy
│   └── utils/            # Narzędzia pomocnicze
├── logs/                 # Pliki logów
├── python_libs/          # Uproszczone/specjalne biblioteki
├── static/               # Statyczne zasoby web
├── templates/            # Szablony HTML
├── main.py               # Główny punkt wejścia aplikacji
└── requirements.txt      # Zależności projektu
```

## 🛠️ Wymagania systemowe

- Python 3.8 lub nowszy
- Pip (menedżer pakietów Python)
- Konto w serwisie Bybit oraz klucze API (opcjonalnie)

## 🚀 Lokalne uruchomienie projektu

### 1. Klonowanie repozytorium

```bash
git clone [adres-repozytorium]
cd inteligentny-system-tradingowy
```

### 2. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 3. Konfiguracja środowiska

Utwórz plik `.env` na podstawie `.env.example`:

```bash
cp .env.example .env
```

Edytuj plik `.env` i ustaw swoje klucze API:

```
BYBIT_API_KEY=twoj_klucz_api
BYBIT_API_SECRET=twoj_sekret_api
BYBIT_TESTNET=false
MARKET_TYPE=spot
```

### 4. Uruchomienie aplikacji

```bash
python main.py
```

Aplikacja będzie dostępna pod adresem: http://127.0.0.1:5000

## 📊 Funkcje i możliwości

- Dashboard z pełnym interfejsem użytkownika
- Handel automatyczny i półautomatyczny
- Zaawansowane zarządzanie ryzykiem
- Wielostrategiowe podejście
- Analiza techniczna i sentymentalna
- Backtesting i optymalizacja strategii
- Integracja z API Bybit

## 🛡️ Tryby działania

Aplikacja może działać w dwóch trybach:

1. **Tryb symulowany** (domyślny) - dane są generowane na podstawie symulacji, bez rzeczywistych transakcji
2. **Tryb produkcyjny** - połączenie z API Bybit, rzeczywiste transakcje

Aby przełączyć na tryb produkcyjny, upewnij się, że masz skonfigurowane klucze API w pliku `.env`.

## 🧪 Uruchamianie testów

```bash
# Testy jednostkowe
python -m pytest data/tests/

# Test połączenia z API Bybit
python test_bybit_connection.py

# Test modeli AI
python test_models.py
```

## ⚙️ Konfiguracja modeli AI

System wspiera wiele modeli analizy AI:

- **XGBoost** - model predykcji cenowej
- **Sentiment Analyzer** - analiza sentymentu rynkowego
- **Anomaly Detector** - wykrywanie anomalii na rynku

## 🔧 Rozwiązywanie problemów

1. **Problem z połączeniem API**:
   - Sprawdź poprawność kluczy API w pliku `.env`
   - Upewnij się, że masz dostęp do internetu
   - Sprawdź uprawnienia kluczy API w panelu Bybit

2. **Błędy importu modułów**:
   - Upewnij się, że wszystkie zależności zostały zainstalowane (`pip install -r requirements.txt`)
   - Usuń i zainstaluj ponownie problematyczny pakiet

3. **Błędy z dostępem do portu**:
   - Zmień port w pliku `.env` (np. na 8080) i uruchom ponownie

## 📝 Rozszerzenie systemu

Aby dodać nową strategię handlową, utwórz nowy plik w katalogu `data/strategies/` i zaimplementuj interfejs strategii. Przykładowo:

```python
class MyNewStrategy:
    def __init__(self, params):
        self.params = params

    def analyze(self, data):
        # Implementacja logiki strategii
        return signals
```
