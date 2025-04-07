# Trading Bot z Integracją ByBit API

System tradingowy z integracją ByBit API, zoptymalizowany do działania w środowisku Replit.

## 🚀 Funkcjonalności

- Połączenie z giełdą ByBit przez REST API
- Pobieranie danych rynkowych w czasie rzeczywistym
- Obsługa kont testowych (testnet) i rzeczywistych
- Analiza techniczna z wykorzystaniem popularnych wskaźników
- Dashboard z wizualizacją danych i alarmami

## 🔧 Wymagania

- Python 3.8+
- Konto ByBit z kluczami API

## ⚙️ Konfiguracja

1. Skonfiguruj zmienne środowiskowe w zakładce Secrets w Replit:
   - `BYBIT_API_KEY` - Klucz API ByBit
   - `BYBIT_API_SECRET` - Sekret API ByBit
   - `BYBIT_USE_TESTNET` - Ustawić na "false" dla API produkcyjnego lub "true" dla środowiska testowego

UWAGA: Podczas pracy z produkcyjnymi kluczami API, upewnij się że:
- Klucze API mają ograniczone uprawnienia (tylko odczyt, jeśli nie potrzebujesz handlu)
- Włączone są dodatkowe zabezpieczenia na koncie ByBit (2FA, ograniczenia IP)
- Regularnie zmieniasz klucze API, jeśli wykryjesz jakiekolwiek niestandardowe zachowanie

2. Zainstaluj zależności:
   ```
   pip install -r requirements.txt
   ```

3. Uruchom aplikację:
   ```
   python main.py
   ```

## 📊 Dashboard

Dashboard jest dostępny na głównej stronie aplikacji i zawiera:
- Aktualne dane rynkowe
- Stan konta i otwarte pozycje
- Wykresy analityczne
- Powiadomienia i alerty

## 🔒 Bezpieczeństwo

- Nigdy nie przechowuj kluczy API w kodzie
- Używaj narzędzia Secrets w Replit do bezpiecznego przechowywania kluczy
- Regularnie weryfikuj uprawnienia kluczy API
- Rozważ użycie testnet do testowania przed użyciem rzeczywistych środków

## 📝 Licencja

Ten projekt jest dostępny na licencji MIT.

## 📋 Struktura projektu (częściowo z oryginalnego projektu)

```
├── ai_models/              # Modele AI/ML do analizy i predykcji
├── data/                   # Moduły danych, strategie i narzędzia
│   ├── data/               # Pobieranie i przetwarzanie danych
│   ├── execution/          # Wykonywanie transakcji
│   ├── indicators/         # Wskaźniki techniczne i analizy
│   ├── logging/            # System logowania i detekcji anomalii
│   ├── optimization/       # Optymalizacja strategii i backtesting
│   ├── risk_management/    # Zarządzanie ryzykiem
│   ├── strategies/         # Strategie tradingowe
│   ├── tests/              # Testy jednostkowe i integracyjne
│   └── utils/              # Narzędzia pomocnicze
├── logs/                   # Pliki logów
├── reports/                # Raporty i analizy
├── saved_models/           # Zapisane modele ML
├── static/                 # Pliki statyczne dla interfejsu
│   ├── css/                # Style CSS
│   └── js/                 # Skrypty JavaScript
├── templates/              # Szablony HTML
├── .env.example            # Przykładowy plik konfiguracyjny
└── main.py                 # Główny plik uruchomieniowy
```

## 📦 Zależności

Główne biblioteki (z oryginalnego projektu, może wymagać aktualizacji):
- Flask - framework webowy
- Pandas/NumPy - przetwarzanie danych
- Scikit-learn - modele ML do analizy i predykcji
- Matplotlib/Chart.js - wizualizacja danych
- ByBit API client library (dodatkowa biblioteka do integracji z ByBit)

## 🔧 Rozwiązywanie problemów

### Problemy z zależnościami
W przypadku konfliktów zależności, możesz użyć:
```
pip install -r requirements.txt --no-dependencies
```
Następnie doinstalować brakujące pakiety ręcznie.

### Błędy importu
Jeśli napotkasz błędy związane z importem modułów, uruchom:
```
python fix_imports.py
```

### Testy
Aby naprawić i uruchomić testy:
```
python fix_tests.py
```
# System Tradingowy - Dashboard

## Opis projektu
System tradingowy z dashboardem do monitorowania i zarządzania strategiami handlu automatycznego na giełdach kryptowalut.

## Funkcjonalności
- Połączenie z API giełdy ByBit (testnet i produkcja)
- Dashboard wizualizujący dane portfela
- Śledzenie statystyk tradingowych
- Monitoring ostatnich transakcji
- System alertów i powiadomień

## Instalacja i uruchomienie

### Wymagania
- Python 3.8+
- Przeglądarka internetowa z obsługą JavaScript

### Instalacja
1. Sklonuj repozytorium
2. Zainstaluj wymagane zależności:
```
pip install -r requirements.txt
```
3. Skonfiguruj zmienne środowiskowe w pliku `.env` (na podstawie `.env.example`)

### Uruchomienie
```
python main.py
```

Po uruchomieniu, dashboard będzie dostępny pod adresem: `http://localhost:5000`

## Konfiguracja API
Aby połączyć się z giełdą ByBit, należy:
1. Utworzyć klucze API na platformie ByBit
2. Uzupełnić dane w pliku `.env`:
```
BYBIT_API_KEY=twój_klucz_api
BYBIT_API_SECRET=twój_sekret_api
USE_PRODUCTION_API=false  # zmień na true dla połączenia produkcyjnego
```

## Bezpieczeństwo
- Używaj testnet do testowania działania systemu
- Przed użyciem produkcyjnego API, upewnij się, że Twoje klucze API mają odpowiednie ograniczenia
- Rozpoczynaj od małych transakcji, aby przetestować działanie systemu

## Rozwiązywanie problemów
Jeśli napotkasz problemy z wyświetlaniem dashboard, sprawdź:
1. Czy wszystkie biblioteki zostały poprawnie zainstalowane
2. Czy aplikacja Flask działa prawidłowo (sprawdź logi)
3. Czy masz dostęp do API ByBit (sprawdź połączenie internetowe i ważność kluczy API)
