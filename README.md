# 🤖 Inteligentny System Tradingowy

## 📝 Opis projektu
Zaawansowany system tradingowy z analizą rynku w czasie rzeczywistym, zarządzaniem ryzykiem i inteligentnym wykonywaniem zleceń. Wykorzystuje algorytmy uczenia maszynowego i analizę sentymentu rynku dla automatycznego podejmowania decyzji.

## 🚀 Funkcje
- Połączenie z API Bybit (testnet i produkcyjne)
- Analiza techniczna i algorytmiczne strategie tradingowe
- Automatyczne dostosowanie do limitów API (backoff eksponencjalny)
- Modele AI do przewidywania ruchów cenowych
- System powiadomień i monitorowania transakcji
- Interaktywny dashboard (Flask)

## 📋 Wymagania systemowe
- Python 3.8+ (zalecany 3.10)
- Aktywne konto Bybit z kluczami API
- Połączenie internetowe
- Windows 10/11

## ⚙️ Instalacja na Windows

```cmd
# Klonowanie repozytorium (jeśli używasz Git)
git clone <URL_repozytorium>
cd inteligentny-system-tradingowy

# Alternatywnie, po pobraniu archiwum .zip
# 1. Rozpakuj plik ZIP
# 2. Otwórz wiersz poleceń (cmd) w lokalizacji rozpakowanego projektu

# Instalacja zależności
pip install -r requirements.txt

# Konfiguracja
copy .env.example .env
# Edytuj plik .env dodając swoje klucze API
```

## 🔧 Konfiguracja
1. Utwórz klucze API w panelu Bybit
2. Uzupełnij plik `.env` swoimi kluczami
3. Ustaw `BYBIT_USE_TESTNET=true` dla środowiska testowego lub `BYBIT_USE_TESTNET=false` dla produkcji

### ⚠️ Środowisko produkcyjne
Gdy `BYBIT_USE_TESTNET=false` operujesz na prawdziwych środkach! System zastosuje dodatkowe zabezpieczenia.

## 🏃 Uruchomienie na Windows
```cmd
# Standardowe uruchomienie
python main.py

# Alternatywnie można użyć batch pliku
run.bat

# Alternatywny sposób uruchomienia z pełnym logowaniem
python -u main.py > logs\app_output.log 2>&1
```

## 📊 Dashboard
Dostęp do dashboardu pod adresem: `http://localhost:5000`

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

## 🔧 Rozwiązywanie problemów na Windows

### Problemy z limitami API (403/429 Errors)
Jeśli napotkasz błędy limitów API (403/429):

1. Ustaw `BYBIT_USE_TESTNET=true` w pliku `.env`
2. Poczekaj 5-10 minut przed ponowną próbą
3. Ustaw `USE_SIMULATED_DATA=true` dla testów bez wywoływania API

### Problemy z zależnościami
Jeśli masz problemy z instalacją zależności, spróbuj zainstalować je pojedynczo:
```cmd
pip install flask requests pandas numpy python-dotenv pybit
```

### Błędy importu
Jeśli napotkasz błędy importu:
```cmd
python fix_imports.py
```

### Testy
Aby naprawić i uruchomić testy:
```cmd
python fix_tests.py
```

## 📜 Licencja
Ten projekt jest udostępniany na licencji MIT.

## 📁 Struktura projektu
- `main.py` - Główny punkt wejścia aplikacji
- `data/` - Przetwarzanie danych i integracja API
  - `execution/` - Moduły do interakcji z giełdą
  - `indicators/` - Wskaźniki techniczne i analiza
  - `risk_management/` - Moduły zarządzania ryzykiem
  - `strategies/` - Strategie tradingowe
  - `utils/` - Funkcje narzędziowe
- `ai_models/` - Modele AI/ML do analizy rynku
- `static/` - Zasoby Frontend
- `templates/` - Szablony HTML
- `logs/` - Logi aplikacji

## 🔧 Technologie
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Analiza danych**: Pandas, NumPy
- **Integracja z giełdą**: ByBit API
- **Przechowywanie danych**: Buforowanie JSON, cache-management