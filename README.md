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

   # Parametry rate limitingu API
   API_MIN_INTERVAL=10.0
   API_MAX_CALLS_PER_MINUTE=3
   API_CACHE_TTL_MULTIPLIER=15.0
   ```

   Alternatywnie możesz użyć narzędzia Secrets w Replit:
   - `BYBIT_API_KEY` - Klucz API ByBit
   - `BYBIT_API_SECRET` - Sekret API ByBit
   - `BYBIT_USE_TESTNET` - Ustawić na "false" dla API produkcyjnego lub "true" dla środowiska testowego

2. Uruchom aplikację za pomocą przycisku "Run" w Replit lub wykonaj:
   ```
   python main.py
   ```

3. Dashboard będzie dostępny pod URL-em wygenerowanym przez Replit (zakładka Webview).

## 🛠️ Rozwiązywanie problemów

### Przekraczanie limitów API (Error 403/429)

System został zoptymalizowany pod kątem zarządzania limitami API, ale jeśli nadal występują problemy:

1. **Używaj testnet zamiast produkcyjnego API**
   - Ustaw `BYBIT_USE_TESTNET=true` w pliku `.env` lub Secrets

2. **Dostosuj parametry limitowania zapytań**
   - Zwiększ `API_MIN_INTERVAL` (np. do 15.0 sekund)
   - Zmniejsz `API_MAX_CALLS_PER_MINUTE` (np. do 2)

3. **Korzystaj z cache**
   - System automatycznie buforuje odpowiedzi API
   - Możesz zwiększyć czas buforowania zmieniając `API_CACHE_TTL_MULTIPLIER`

### Problemy z połączeniem do API

Jeśli występują problemy z połączeniem do API ByBit:

1. **Sprawdź poprawność kluczy API**
   - Upewnij się, że klucze są poprawne i mają odpowiednie uprawnienia

2. **Sprawdź ograniczenia geograficzne**
   - ByBit może blokować dostęp z niektórych lokalizacji
   - Rozważ użycie VPN, jeśli to konieczne

3. **Ograniczenia w środowisku Replit**
   - Replit może mieć ograniczenia w zakresie niektórych zapytań zewnętrznych
   - Upewnij się, że masz aktualne środowisko z właściwymi uprawnieniami

## 🔄 Aktualizacja

System został zoptymalizowany do działania w środowisku Replit. Jeśli potrzebujesz zaktualizować zależności:

```bash
pip install -r requirements.txt
```

## 📊 Dashboard

Dashboard prezentuje:
- Stan portfela
- Ostatnie transakcje
- Wykresy wartości portfela
- Statystyki handlowe
- Alerty i powiadomienia

## ⚠️ Ważne informacje

- **Testuj najpierw z testnet API** - Zawsze testuj swoje strategie najpierw z API testowym, zanim przejdziesz do prawdziwego handlu.
- **Ograniczenia API** - ByBit ma limity zapytań API. System został zoptymalizowany pod kątem zarządzania tymi limitami, ale nadal możliwe jest ich przekroczenie przy intensywnym użytkowaniu.
- **Bezpieczeństwo kluczy API** - Nigdy nie udostępniaj swoich kluczy API. W Replit używaj funkcji Secrets do przechowywania kluczy.

## 📦 Zależności
Główne biblioteki (z oryginalnego projektu, może wymagać aktualizacji):
- Flask - framework webowy
- Pandas/NumPy - przetwarzanie danych
- Scikit-learn - modele ML do analizy i predykcji
- Matplotlib/Chart.js - wizualizacja danych
- ByBit API client library (dodatkowa biblioteka do integracji z ByBit)


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

## 📝 Licencja

Ten projekt jest dostępny na licencji MIT.