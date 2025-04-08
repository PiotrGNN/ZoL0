# Inteligentny System Tradingowy

Kompleksowy system do analizy rynku, zarządzania ryzykiem i automatycznego tradingu.

## 📋 Zawartość projektu

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
├── static/               # Web frontend static assets
├── templates/            # HTML templates
├── main.py               # Main application entry point
└── requirements.txt      # Project dependencies
```

## 🔑 Konfiguracja środowiska

1. Utwórz plik `.env` w katalogu głównym z następującymi parametrami (lub użyj skryptów inicjalizacyjnych):

```
# API Bybit - Configuration
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_USE_TESTNET=true

# Application modes
DEBUG=True
LOG_LEVEL=INFO
IS_PRODUCTION=False
```

## 🚀 Uruchomienie aplikacji

### W środowisku lokalnym (Windows)

1. Uruchom skrypt `run_local.bat` (dla Windows):
```
run_local.bat
```

### W środowisku lokalnym (Linux/Mac)

1. Uruchom skrypt `run_local.sh` (dla Linux/Mac):
```bash
chmod +x run_local.sh
./run_local.sh
```

### W środowisku Replit

1. Kliknij przycisk "Run" w środowisku Replit.

## 🛠️ Rozwój projektu

### Dodawanie nowej strategii

Utwórz nowy plik strategii w `data/strategies/` o następującej strukturze:

```python
class MyNewStrategy:
    def __init__(self, params):
        self.params = params

    def analyze(self, data):
        # Implement strategy logic
        return signals
```

## 📊 Testowanie

1. Testy jednostkowe można uruchomić za pomocą:
```
python -m pytest data/tests/
```

2. Testy połączenia z API Bybit:
```
python test_bybit_connection.py
```

## 📈 Funkcje i możliwości

- Handel automatyczny i półautomatyczny
- Zaawansowane zarządzanie ryzykiem
- Wielostrategiowe podejście
- Analiza techniczna i sentymentalna
- Backtesting i optymalizacja strategii
- Integracja z API Bybit

## 📚 Dokumentacja API

Dokumentacja API jest dostępna pod adresem http://localhost:5000/docs po uruchomieniu aplikacji.

## 📫 Kontakt i wsparcie

W razie problemów lub pytań, prosimy o zgłaszanie ich w sekcji Issues.

## 📄 Licencja

Ten projekt jest udostępniany na licencji MIT.