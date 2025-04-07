# Trading Bot - System Automatycznego Tradingu

## 🚀 O Projekcie

System automatycznego tradingu bazujący na sztucznej inteligencji i analizie technicznej. Projekt łączy zaawansowane algorytmy uczenia maszynowego, analizę sentymentu rynkowego oraz tradycyjne strategie tradingowe, aby oferować kompleksowe rozwiązanie do handlu na rynkach kryptowalut.

## 🔧 Technologie

- **Backend**: Python, Flask
- **Analiza danych**: Pandas, NumPy, SciKit-Learn
- **AI/ML**: TensorFlow, XGBoost, NLTK, Transformers
- **Frontend**: JavaScript, Chart.js
- **Bazy danych**: SQLite, SQLAlchemy
- **Exchange API**: CCXT, python-binance

## 🚦 Instrukcja Uruchomienia

### Wymagania

- Python 3.10+
- Wszystkie zależności wymienione w `requirements.txt`

### Instalacja

1. Sklonuj repozytorium
2. Zainstaluj zależności:
   ```
   pip install -r requirements.txt
   ```
3. Utwórz plik `.env` na podstawie `.env.example` i uzupełnij kluczowe wartości
4. Uruchom aplikację:
   ```
   python main.py
   ```

### Dostęp do Dashboardu

Po uruchomieniu aplikacja będzie dostępna pod adresem: `http://0.0.0.0:5000/`

## 📊 Główne Funkcjonalności

- **Dashboard Analityczny**: Monitorowanie wyników, statystyk i otwartych pozycji
- **Strategie Tradingowe**: Zestaw wbudowanych strategii (Trend Following, Mean Reversion, Breakout)
- **Modele AI**: Predykcja cen, analiza sentymentu, wykrywanie anomalii
- **Zarządzanie Ryzykiem**: Dynamiczne zarządzanie wielkością pozycji i stop-lossami
- **Backtest**: Testowanie strategii na danych historycznych
- **Powiadomienia**: System alertów o ważnych zdarzeniach rynkowych

## 📋 Struktura Projektu

- `ai_models/` - Modele sztucznej inteligencji i uczenia maszynowego
- `data/` - Moduły do pobierania, przetwarzania i zarządzania danymi
  - `data/indicators/` - Wskaźniki techniczne i analiza sentymentu
  - `data/strategies/` - Implementacje strategii tradingowych
  - `data/risk_management/` - Zarządzanie ryzykiem i wielkością pozycji
  - `data/execution/` - Wykonywanie zleceń i połączenia z giełdami
- `static/` - Pliki statyczne dla front-endu (JS, CSS)
- `templates/` - Szablony HTML dla dashboardu
- `logs/` - Logi aplikacji

## 🔍 Testowanie

Uruchom testy jednostkowe:
```
pytest data/tests/
```

Testy z raportowaniem pokrycia kodu:
```
pytest --cov=. --cov-report=term-missing data/tests/
```

## 📝 Konfiguracja

Konfiguracja systemu odbywa się przez:
- Plik `.env` - Zmienne środowiskowe, klucze API
- `config/settings.py` - Główne ustawienia aplikacji
- Dashboard webowy - Konfiguracja parametrów strategii

## 🛡️ Bezpieczeństwo

- Klucze API przechowywane są w zmiennych środowiskowych
- Szyfrowanie wrażliwych danych
- System monitorowania bezpieczeństwa

## 📜 Licencja

Ten projekt jest udostępniany na licencji MIT.