
# 🚀 AI Trading System

Zaawansowany system tradingowy integrujący modele sztucznej inteligencji z giełdami kryptowalut.

## 📋 Funkcjonalności

- 🧠 **Modele AI/ML** - predykcja ruchów cenowych, analiza sentymentu, wykrywanie anomalii
- 📊 **Zarządzanie ryzykiem** - dynamiczne pozycjonowanie, kontrola ryzyka, optymalizacja dźwigni
- 🔄 **Integracja z giełdami** - Bybit, Binance (REST + WebSocket)
- 📈 **Strategie handlowe** - Trend following, Mean reversion, Breakout
- 🔍 **Backtesting** - Testowanie na danych historycznych
- 💼 **Dashboard webowy** - Monitorowanie systemu, alertów i metryk

## 🛠️ Instalacja

### Wymagania

- Python 3.9+
- Konto na Bybit i/lub Binance
- Klucze API z odpowiednimi uprawnieniami

### Konfiguracja

1. Sklonuj repozytorium
2. Zainstaluj zależności:
   ```
   pip install -r requirements.txt
   ```
3. Utwórz plik `.env` na podstawie `.env.example`:
   ```
   cp .env.example .env
   ```
4. Dodaj swoje klucze API do pliku `.env` lub Replit Secrets

## 🚀 Uruchomienie

### Standardowe uruchomienie

```bash
python main.py
```

### Tryb testowy (Testnet)

```bash
# Upewnij się, że w .env jest ustawione TEST_MODE=true
python main.py
```

### Narzędzia developerskie

```bash
# Testy
pytest data/tests/

# Formatowanie kodu
black .

# Lint
flake8 .

# Testy z pokryciem kodu
pytest --cov=. --cov-report=term-missing data/tests/
```

## 📁 Struktura projektu

```
├── config/            - Konfiguracja aplikacji
├── data/              - Główny katalog modułów projektu
│   ├── data/          - Zarządzanie danymi rynkowymi
│   ├── execution/     - Integracja z giełdami i wykonywanie zleceń
│   ├── indicators/    - Wskaźniki techniczne i analizy
│   ├── logging/       - System logowania
│   ├── risk_management/ - Zarządzanie ryzykiem
│   ├── strategies/    - Strategie handlowe
│   ├── tests/         - Testy jednostkowe
│   └── utils/         - Narzędzia pomocnicze
├── logs/              - Pliki logów
├── saved_models/      - Zapisane modele AI
├── static/            - Pliki statyczne (CSS, JS)
├── templates/         - Szablony HTML
└── main.py            - Główny plik aplikacjiov=. --cov-report=term-missing data/tests/
```

## 🧩 Struktura projektu

```
├── ai_models/             # Modele AI/ML
├── config/                # Konfiguracja systemu
├── data/                  # Moduły tradingowe
│   ├── data/              # Pobieranie i przetwarzanie danych
│   ├── execution/         # Wykonywanie zleceń
│   ├── indicators/        # Wskaźniki techniczne
│   ├── risk_management/   # Zarządzanie ryzykiem
│   ├── strategies/        # Strategie handlowe
│   ├── tests/             # Testy
│   └── utils/             # Narzędzia pomocnicze
├── logs/                  # Logi systemu
├── saved_models/          # Zapisane modele AI
├── static/                # Zasoby statyczne dla GUI
├── templates/             # Szablony HTML
├── .env.example           # Przykładowy plik konfiguracyjny
├── main.py                # Punkt wejściowy
└── requirements.txt       # Zależności
```

## 🔒 Bezpieczeństwo

- Przechowuj klucze API używając Replit Secrets Manager (`Tools → Secrets`)
- Ogranicz uprawnienia kluczy API do minimum
- Używaj trybu testowego (testnet) do testowania nowych strategii
- Ustaw odpowiednie limity ryzyka w konfiguracji

## 📈 Integracja z Bybit API

System integruje się z Bybit API dla spot i futures:

```python
from data.execution.bybit_connector import BybitConnector

# Inicjalizacja
bybit = BybitConnector(test_mode=True)

# REST API
ticker = bybit.get_tickers("BTCUSDT")
orderbook = bybit.get_orderbook("BTCUSDT")
balance = bybit.get_wallet_balance()

# WebSocket
bybit.connect_websocket()
bybit.subscribe("orderbook.50.BTCUSDT", process_data_callback)
```

## 🧠 Wykorzystanie modeli AI

System obsługuje różne modele AI do predykcji rynkowych:

```python
from ai_models.model_training import train_price_predictor
from data.execution.ai_trade_bridge import AITradeBridge

# Inicjalizacja mostu AI-Trading
bridge = AITradeBridge()

# Przetwarzanie predykcji
prediction = {"action": "BUY", "confidence": 0.85, "price": 50000.0}
result = bridge.process_ai_prediction(prediction, symbol="BTCUSDT")
```

## 📊 Monitorowanie systemu

System udostępnia dashboard webowy na porcie 5000:

```
http://localhost:5000/dashboard
```

## 🏆 Dobre praktyki

- Zacznij od małych alokacji kapitału
- Regularnie monitoruj logi i metryki systemu
- Testuj nowe strategie na testnecie przed uruchomieniem na rynku rzeczywistym
- Regularnie aktualizuj modele AI nowymi danymi

## 📜 Licencja

Ten projekt jest zastrzeżony i nie jest dostępny publicznie bez zgody autora.
