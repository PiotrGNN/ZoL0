# ZoL0-1 Trading System

## 🚀 O Projekcie

Zaawansowany system tradingowy oparty na modułowej architekturze, umożliwiający algorytmiczny handel na rynkach kryptowalut z wykorzystaniem API ByBit. System integruje analizę techniczną, zarządzanie ryzykiem i modele sztucznej inteligencji.

## 📋 Funkcjonalności

- ✅ Łączność z API ByBit (testnet i produkcja)
- ✅ Zarządzanie ryzykiem portfela 
- ✅ Wielostrategiowe podejście do handlu
- ✅ Interfejs graficzny (dashboard)
- ✅ Analiza sentymentu rynkowego
- ✅ Backtesting strategii
- ✅ Wykrywanie anomalii rynkowych

## 🛠️ Architektura Systemu

Projekt ma modułową strukturę, składającą się z:

- `data/` - moduły przetwarzania danych, wskaźników i strategii
- `python_libs/` - uproszczone biblioteki do szybkiego prototypowania
- `static/` - pliki statyczne (CSS, JavaScript)
- `templates/` - szablony HTML
- `utils/` - narzędzia pomocnicze
- `main.py` - główny punkt wejścia aplikacji

## 🚀 Uruchomienie

Aby uruchomić system:

1. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

2. Skonfiguruj środowisko:
```bash
# Skopiuj plik .env.example do .env i dostosuj ustawienia
cp .env.example .env
```

3. Uruchom aplikację:
```bash
python main.py
```

## 🔧 Diagnostyka i Rozwiązywanie Problemów

### Trading Engine Warning
Jeśli otrzymujesz ostrzeżenie "Trading Engine Warning", może to być spowodowane:
- Brakiem danych rynkowych
- Niedostępnością instrumentów
- Błędnymi ustawieniami strategii

Sprawdź logi w `logs/trading_engine.log` dla szczegółowych informacji.

### Risk Manager Warning
Ostrzeżenia Risk Managera mogą wystąpić w przypadku:
- Niepoliczonego ryzyka (brak danych cenowych)
- Pozycji, która nie została otwarta
- Przekroczenia limitów ryzyka

Sprawdź logi w `logs/portfolio_risk.log` dla diagnostyki.

## 📊 Dashboard

System posiada interfejs graficzny dostępny pod adresem:
```
http://127.0.0.1:5000/dashboard
```

## 🔒 Bezpieczeństwo

Pamiętaj, że używanie API produkcyjnego wiąże się z ryzykiem. Zawsze:
- Używaj kluczy API z ograniczeniami
- Testuj na testnet przed wdrożeniem produkcyjnym
- Monitoruj aktywnie działanie systemu

## 📄 Licencja

Ten projekt jest udostępniany na licencji MIT.

## 📁 Project Structure
```
├── ai_models/            # AI and ML models
├── data/                 # Data processing and strategy implementations
│   ├── execution/        # Exchange connectors and order execution
│   ├── indicators/       # Technical and market indicators
│   ├── optimization/     # Strategy and portfolio optimization
│   ├── risk_management/  # Risk assessment and position sizing
│   ├── strategies/       # Trading strategies
│   └── utils/            # Utility functions and helpers
├── logs/                 # System logs
├── python_libs/          # Local library modules
├── static/               # Web frontend static assets
├── templates/            # HTML templates
├── main.py               # Main application entry point
└── requirements.txt      # Project dependencies
```

## 🔑 Environment Configuration
Create a `.env` file in the root directory with the following parameters:

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

## 🛠️ Development
### Adding a New Strategy
Create a new strategy file in `data/strategies/` with the following structure:

```python
class MyNewStrategy:
    def __init__(self, params):
        self.params = params
        
    def analyze(self, data):
        # Implement strategy logic
        return signals
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## 📊 Performance Metrics
The system tracks key performance indicators:

- Win Rate
- Profit Factor
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio