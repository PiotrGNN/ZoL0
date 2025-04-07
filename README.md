
# 🤖 AI Trading System

Zaawansowany system tradingowy oparty na AI, wykorzystujący modele uczenia maszynowego do analizy rynku i wykonywania transakcji.

## 🚀 Uruchomienie projektu

### Środowisko Replit
Projekt jest w pełni skonfigurowany do pracy w środowisku Replit:

1. Kliknij przycisk **Run** aby uruchomić aplikację
2. Aplikacja webowa będzie dostępna pod adresem podanym w zakładce "Webview"

### 🔒 Konfiguracja API

Aby używać połączenia z giełdami kryptowalut:

1. Utwórz plik `.env` na podstawie `.env.example`
2. Dodaj swoje klucze API do pliku `.env` lub użyj narzędzia Secrets w Replit

```
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
```

## 🧩 Struktura projektu

- `main.py` - Główny plik uruchomieniowy aplikacji
- `ai_models/` - Modele AI do analizy i prognozowania rynku
- `data/` - Moduły do pobierania danych, zarządzania ryzykiem i strategii
- `config/` - Konfiguracja aplikacji
- `templates/` - Szablony HTML dla interfejsu webowego
- `static/` - Pliki statyczne (CSS, JS)

## 🛠️ Technologie

- Python 3.10
- Flask (backend web)
- Pandas, NumPy, SciKit-Learn (analiza danych)
- XGBoost, Optuna (modelowanie AI)
- Bybit & Binance API (integracja z giełdami)

## 🔍 Testowanie

Testy jednostkowe znajdują się w katalogu `data/tests/`. Uruchom je poleceniem:
```
python -m pytest data/tests/
```
