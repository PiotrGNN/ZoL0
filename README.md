# 🤖 Inteligentny System Tradingowy

## 📝 Opis projektu
Zaawansowany system tradingowy z analizą rynku w czasie rzeczywistym, zarządzaniem ryzykiem i inteligentnym wykonywaniem zleceń. Wykorzystuje algorytmy uczenia maszynowego i analizę sentymentu rynku dla automatycznego podejmowania decyzji.

## 🚀 Funkcje
- Połączenie z API Bybit (testnet i produkcyjne)
- Analiza techniczna i algorytmiczne strategie tradingowe
- Integracja z VPN/Proxy przez tunel SSH (omijanie limitów geograficznych)
- Automatyczne dostosowanie do limitów API (backoff eksponencjalny)
- Modele AI do przewidywania ruchów cenowych
- System powiadomień i monitorowania transakcji
- Interaktywny dashboard (Flask)

## 📋 Wymagania
- Python 3.8+
- Aktywne konto Bybit z kluczami API
- Dostęp do VPS (opcjonalnie, dla tunelu SSH)

## ⚙️ Instalacja
```bash
# Klonowanie repozytorium
git clone <URL_repozytorium>
cd inteligentny-system-tradingowy

# Instalacja zależności
pip install -r requirements.txt

# Konfiguracja
cp .env.example .env
# Edytuj plik .env dodając swoje klucze API
```

## 🔧 Konfiguracja
1. Utwórz klucze API w panelu Bybit
2. Uzupełnij plik `.env` swoimi kluczami
3. Ustaw `BYBIT_TESTNET=true` dla środowiska testowego lub `BYBIT_TESTNET=false` dla produkcji

### ⚠️ Środowisko produkcyjne
Gdy `BYBIT_TESTNET=false` operujesz na prawdziwych środkach! System zastosuje dodatkowe zabezpieczenia.

## 🏃 Uruchomienie
```bash
# Standardowe uruchomienie
python main.py

# Uruchomienie z VPN przez tunel SSH (jeśli potrzebne)
python setup_ssh_tunnel.py
# W nowym terminalu:
python main.py
```

## 📊 Dashboard
Dostęp do dashboardu pod adresem: `http://localhost:5000`

## 🔒 Konfiguracja VPN/Proxy
System umożliwia działanie przez tunel SSH dla stabilnego połączenia API:

1. Uruchom: `python setup_ssh_tunnel.py --host <twoj_vps_ip> --user <nazwa_uzytkownika>`
2. Ustaw `USE_PROXY=True` w pliku `.env`
3. Testuj: `python test_proxy.py`

## 🧠 AI Models
System zawiera różne modele AI do analizy rynku:
- Wykrywanie anomalii
- Analiza sentymentu
- Predykcja cen
- Reinforcement learning dla strategii tradingowych

## 📊 Dashboard
Dostęp do panelu webowego pod adresem `http://localhost:5000` do monitorowania:
- Aktualnych pozycji
- Stanu konta
- Metryk wydajności
- Sentymentu rynku i anomalii

## 🔧 Rozwiązywanie problemów

### Problemy z limitami API (403/429 Errors)
Jeśli napotkasz błędy limitów API (403/429):

1. Ustaw `BYBIT_TESTNET=true` w pliku `.env`
2. Poczekaj 5-10 minut przed ponowną próbą
3. Ustaw `USE_SIMULATED_DATA=true` dla testów bez wywoływania API
4. Rozważ użycie innego adresu IP (np. przez VPN lub proxy)
5. Do testów, użyj trybu symulacji - ustaw `USE_SIMULATED_DATA=true` w `.env`

### Problemy z zależnościami
Jeśli masz konflikty zależności, spróbuj zainstalować bez zależności:
```bash
pip install -r requirements.txt --no-deps
```

Następnie zainstaluj brakujące pakiety ręcznie.

### Błędy importu
Jeśli napotkasz błędy importu:
```bash
python fix_imports.py
```

### Testy
Aby naprawić i uruchomić testy:
```bash
python fix_tests.py
```

## 📜 Licencja
Ten projekt jest udostępniany na licencji MIT.

## 📁 Project Structure
- `main.py` - Main application entry point
- `data/` - Data processing and API integration
  - `execution/` - Exchange interaction modules
  - `indicators/` - Technical indicators and analysis
  - `risk_management/` - Risk management modules
  - `strategies/` - Trading strategies
  - `utils/` - Utility functions
- `ai_models/` - AI/ML models for market analysis
- `static/` - Frontend assets
- `templates/` - HTML templates
- `logs/` - Application logs

## 🔧 Technologie
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Analiza danych**: Pandas, NumPy
- **Integracja z giełdą**: ByBit API
- **Przechowywanie danych**: Buforowanie JSON, cache-management

## Uruchomienie
1. Skonfiguruj zmienne środowiskowe w pliku `.env`
2. Uruchom aplikację: `python main.py`

## Konfiguracja tunelu SSH SOCKS5 Proxy
Jeśli masz problemy z dostępem do API Bybit z Replit, możesz skonfigurować tunel SSH SOCKS5:

1. Edytuj plik `.env` i ustaw poniższe zmienne:
   ```
   USE_PROXY=True
   PROXY_URL=socks5h://127.0.0.1:1080
   VPS_USER=twój_użytkownik
   VPS_HOST=adres_ip_vps
   VPS_PORT=22
   VPS_KEY_PATH=/ścieżka/do/klucza/ssh  # opcjonalnie
   ```

2. Uruchom tunel SSH w jednym terminalu:
   ```
   python setup_ssh_tunnel.py
   ```

3. W drugim terminalu uruchom aplikację:
   ```
   python main.py
   ```

4. Aby przetestować tunel, użyj:
   ```
   python test_proxy.py