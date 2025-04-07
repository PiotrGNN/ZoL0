# Trading Bot z Integracją ByBit API

System tradingowy z integracją ByBit API, zoptymalizowany do działania w środowisku Replit.

## 🚀 Funkcjonalności

- Połączenie z giełdą ByBit przez REST API
- Pobieranie danych rynkowych w czasie rzeczywistym
- Obsługa kont testowych (testnet) i rzeczywistych
- Analiza techniczna z wykorzystaniem popularnych wskaźników
- Dashboard z wizualizacją danych i alarmami
- Zaawansowane zarządzanie limitami API
- Inteligentne buforowanie danych

## 🔧 Wymagania

- Python 3.8+
- Konto ByBit z kluczami API

## ⚙️ Konfiguracja

1. Skonfiguruj plik `.env` w katalogu głównym (możesz wykorzystać `.env.example` jako szablon):
   ```
   BYBIT_API_KEY=twój_klucz_api
   BYBIT_API_SECRET=twój_sekret_api
   BYBIT_USE_TESTNET=true  # Zmień na false dla produkcyjnego API
   ```

   Alternatywnie możesz użyć narzędzia Secrets w Replit:
   - `BYBIT_API_KEY` - Klucz API ByBit
   - `BYBIT_API_SECRET` - Sekret API ByBit
   - `BYBIT_USE_TESTNET` - Ustawić na "false" dla API produkcyjnego lub "true" dla środowiska testowego

2. Zainstaluj zależności:
   ```
   pip install -r requirements.txt
   ```

3. Uruchom aplikację:
   ```
   python main.py
   ```

## ⚠️ Bezpieczeństwo API

UWAGA: Podczas pracy z produkcyjnymi kluczami API:
- Zawsze używaj kluczy z minimalnymi wymaganymi uprawnieniami (tylko odczyt jeśli nie potrzebujesz handlu)
- Włącz dodatkowe zabezpieczenia na koncie ByBit (2FA, ograniczenia IP)
- Regularnie zmieniaj klucze API
- Ustaw odpowiednie limity handlowe w panelu API ByBit
- Nigdy nie pushuj pliku .env do repozytorium!

## 📊 Dashboard

Dashboard jest dostępny na głównej stronie aplikacji i zawiera:
- Aktualne dane rynkowe i wskaźniki techniczne
- Stan konta i otwarte pozycje
- Alerty i powiadomienia
- Statystyki handlowe
- Status komponentów systemu

## 🔍 Rozwiązywanie problemów

Jeśli napotkasz problemy z limitami API (błędy 403 lub CloudFront):
1. Zmniejsz częstotliwość odpytywania API poprzez edycję parametrów w `data/utils/cache_manager.py`
2. Upewnij się, że używasz testnet podczas rozwoju aplikacji
3. Sprawdź logi błędów w katalogu `logs/`
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

### Problemy z limitami API (403/429 Errors)
Jeśli napotkasz błędy związane z przekroczeniem limitów API:

```
You have breached the ip rate limit. (ErrCode: 403)
```

lub błędy CloudFront:

```
The Amazon CloudFront distribution is configured to block access from your country.
```

Rozwiązania:
1. Zmodyfikuj zmienną `USE_TESTNET` w pliku `.env` na `true`
2. Poczekaj 5-10 minut przed następną próbą połączenia
3. Zmniejsz częstotliwość odpytywania API w `data/utils/cache_manager.py`
4. Korzystaj z innego adresu IP (np. przez VPN lub proxy)
5. Dla testów używaj trybu symulacji - ustaw `USE_SIMULATED_DATA=true` w `.env`

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
