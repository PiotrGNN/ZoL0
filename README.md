# Trading Bot - Wersja Windows

## 📋 Opis Projektu

Zaawansowany bot tradingowy zintegrowany z giełdą Bybit, wykorzystujący modele AI do generowania sygnałów.  System do automatycznego tradingu na giełdzie Bybit z wykorzystaniem różnych strategii i modeli AI.

## 🔧 Wymagania

- Windows 10/11
- Python 3.10+
- Konto na giełdzie Bybit z kluczami API (opcjonalnie)
- Dostęp do internetu


## 🚀 Instalacja i Uruchomienie

### 1. Przygotowanie środowiska

1. Sklonuj repozytorium:
```bash
git clone [adres-repozytorium]
cd [folder-projektu]
```

2. Utwórz wirtualne środowisko Python:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

### 2. Konfiguracja

1. Skopiuj plik `.env.example` do `.env`:
```bash
copy .env.example .env
```

2. Edytuj plik `.env` i dodaj swoje klucze API Bybit:
```
BYBIT_API_KEY=TwójKluczAPI
BYBIT_API_SECRET=TwójSekretAPI
BYBIT_USE_TESTNET=true  # Ustaw na false dla produkcyjnego API (z prawdziwymi środkami)
```

### 3. Uruchomienie

Możesz uruchomić bota na dwa sposoby:

1. Za pomocą pliku wsadowego (zalecane dla Windows):
```
run_windows.bat
```

2. Bezpośrednio z Pythona:
```
python main.py
```

## 📊 Funkcje

- Połączenie z API Bybit (z obsługą testnet i produkcji)
- Zaawansowane zarządzanie ryzykiem
- System cache z kontrolą limitów zapytań
- Monitorowanie wydajności
- Dashboard Flask do kontroli i wizualizacji
- Technical analysis and algorithmic trading strategies
- Automatic adjustment to API limits (exponential backoff)
- AI models for price movement prediction
- Transaction monitoring and notification system
- Interactive dashboard (Flask)


## 🧪 Testowanie

1. Test połączenia z API:
```
test_proxy_windows.bat
```

2. Uruchomienie wszystkich testów:
```
python -m pytest
```

## 🔒 Bezpieczeństwo

- Wszystkie klucze API i dane wrażliwe przechowywane są w pliku `.env` (lokalnie)
- Logika obsługi błędów z graceful fallback
- Limity zabezpieczające przed niekontrolowanymi transakcjami

## 📂 Struktura Katalogów

- `data/` - Komponenty systemu, strategie, API
- `logs/` - Pliki logów
- `templates/` - Szablony Flask dla dashboardu
- `static/` - Statyczne pliki dla dashboardu
- `ai_models/` - Modele sztucznej inteligencji
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


## 📝 Znane problemy

- Działająca integracja z Bybit zależy od aktualnej dostępności API
- Aktualny status API testnet: ✅ Działa
- Aktualny status API produkcyjnego: ✅ Działa

## 📚 Dokumentacja API

- [Bybit API V5](https://bybit-exchange.github.io/docs/v5/intro)

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