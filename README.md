# Inteligentny System Tradingowy

System do automatycznego tradingu na giełdzie Bybit z wykorzystaniem różnych strategii i modeli AI.

## 📝 Project Description
Advanced trading system with real-time market analysis, risk management, and intelligent order execution. Utilizes machine learning algorithms and market sentiment analysis for automated decision-making.

## 🚀 Features
- Connection to Bybit API (testnet and production)
- Technical analysis and algorithmic trading strategies
- Automatic adjustment to API limits (exponential backoff)
- AI models for price movement prediction
- Transaction monitoring and notification system
- Interactive dashboard (Flask)

## 📋 System Requirements
- Windows 11 (lub 10)
- Python 3.10+
- Dostęp do internetu
- Konto na giełdzie Bybit z kluczami API


## 🚀 Instrukcja instalacji lokalnej

### 1. Pobierz projekt

Pobierz i rozpakuj projekt do wybranego folderu, np. `C:\Users\piotr\Desktop\miner\ZoL0-1`.

### 2. Utwórz plik .env

Skopiuj plik `.env.example` do `.env` i wprowadź odpowiednie wartości:

```
BYBIT_API_KEY=TwójKluczAPI
BYBIT_API_SECRET=TwójSekretAPI
BYBIT_USE_TESTNET=true  # Ustaw na false dla produkcyjnego API (z prawdziwymi środkami)
```

### 3. Zainstaluj zależności

Otwórz wiersz poleceń (CMD) w folderze projektu i wykonaj:

```cmd
pip install -r requirements.txt
```

### 4. Uruchom aplikację

Metoda 1: Użyj skryptu bat:
```cmd
run_windows.bat
```

Metoda 2: Uruchom bezpośrednio Pythona:
```cmd
python main.py
```

Aplikacja będzie dostępna pod adresem: http://localhost:5000

### 5. Test połączenia z Bybit API

Aby sprawdzić czy połączenie z Bybit API działa poprawnie, uruchom:

```cmd
python test_bybit_connection.py
```

## 🔧 Rozwiązywanie problemów

### Brak modułu

Jeśli pojawi się błąd o brakującym module, zainstaluj go ręcznie:

```cmd
pip install nazwa_modułu
```

### Problemy z połączeniem API

1. Sprawdź poprawność kluczy API w pliku `.env`
2. Upewnij się, że masz działające połączenie z internetem
3. Sprawdź czy Twoje klucze API mają odpowiednie uprawnienia na Bybit

### Inne problemy

1. Sprawdź logi w katalogu `logs/`
2. Uruchom aplikację w trybie debug: `python main.py --debug`


## 📁 Struktura projektu

- `main.py` - Główny punkt wejścia aplikacji
- `data/` - Dane, strategie i narzędzia
  - `execution/` - Moduły do interakcji z giełdą
  - `indicators/` - Wskaźniki techniczne
  - `risk_management/` - Zarządzanie ryzykiem
  - `strategies/` - Strategie tradingowe
  - `utils/` - Funkcje narzędziowe
- `ai_models/` - Modele AI/ML do analizy rynku
- `static/` - Zasoby front-end
- `templates/` - Szablony HTML
- `logs/` - Logi aplikacji

## 📜 Licencja

Ten projekt jest rozpowszechniany na licencji MIT.

## 📊 Dashboard
Access the web panel at `http://localhost:5000` to monitor:
- Current positions
- Account status
- Performance metrics
- Market sentiment and anomalies

## 🧠 AI Models
The system contains various AI models for market analysis:
- Anomaly detection
- Sentiment analysis
- Price prediction
- Reinforcement learning for trading strategies