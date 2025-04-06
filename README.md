# 🚀 Advanced AI Trading System

System do automatycznego tradingu z wykorzystaniem algorytmów AI i ML.

## 📋 Funkcje

- 🤖 **AI/ML Trading** - Wykorzystanie modeli uczenia maszynowego do predykcji ruchu cen
- 📊 **Wykrywanie anomalii** - Identyfikacja nietypowych wzorców rynkowych
- 📈 **Optymalizacja strategii** - Automatyczne dostosowanie parametrów strategii
- 🔄 **Backtesting** - Testowanie strategii na danych historycznych
- 🛡️ **Zarządzanie ryzykiem** - Zaawansowane mechanizmy kontroli ryzyka


## 🚀 Uruchomienie

1. Upewnij się, że masz plik `.env` (możesz skopiować z `.env.example`):
   ```
   cp .env.example .env
   ```

2. Zainstaluj wymagane zależności:
   ```
   pip install -r requirements.txt
   ```

3. Uruchom system tradingowy:
   ```
   python main.py
   ```

## 🔧 Konfiguracja

Projekt używa pliku `.env` do konfiguracji kluczy API, poziomów logowania i innych ustawień:

```
# API Keys
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# Environment
TRADING_MODE=testnet
LOG_LEVEL=INFO
```

## 🧪 Testy

Uruchomienie testów jednostkowych:

```
pytest data/tests/
```

## 📂 Struktura projektu

```
├── ai_models/            # AI and ML models
├── config/               # Configuration files
├── data/                 # Data processing and strategy modules
│   ├── data/             # Data acquisition and preprocessing
│   ├── execution/        # Order execution logic
│   ├── indicators/       # Technical indicators
│   ├── logging/          # Logging utilities
│   ├── optimization/     # Strategy optimization
│   ├── risk_management/  # Risk control modules
│   ├── strategies/       # Trading strategies
│   ├── tests/            # Unit tests
│   └── utils/            # Utility functions
├── logs/                 # Log files
├── saved_models/         # Serialized trained models
└── main.py               # Main entry point
```

## 🛠️ Development
### Testing
Run tests with:
```
pytest data/tests/
```

### Code Style
Format code with:
```
black .
```
Check code style with:
```
flake8
```

## 📄 License

This project is proprietary software.

## 🤝 Contributing

Please read the CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.