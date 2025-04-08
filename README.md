# System Tradingowy z Dashboardem

## 📋 Opis projektu
Zaawansowany system tradingowy z dashboardem do monitorowania i zarządzania strategiami handlu automatycznego na giełdach kryptowalut, ze szczególnym uwzględnieniem integracji z ByBit.

## 🚀 Funkcjonalności

- **Dashboard analityczny** z wizualizacją danych portfela w czasie rzeczywistym
- **Integracja z API ByBit** (obsługa zarówno testnet jak i produkcyjnego API)
- **Zarządzanie ryzykiem** z konfigurowalnymi parametrami
- **Analiza sentymentu rynku** z wykorzystaniem AI
- **Wykrywanie anomalii cenowych** w czasie rzeczywistym
- **System alertów i powiadomień** o ważnych zdarzeniach na rynku
- **Zarządzanie portfelem** z monitorowaniem sald i historii transakcji

## 🔧 Technologie

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Analiza danych**: Pandas, NumPy
- **Integracja z giełdą**: ByBit API
- **Przechowywanie danych**: Buforowanie JSON, cache-management

## 🛠️ Instalacja i uruchomienie

### Wymagania wstępne
- Python 3.10+
- Pip

### Kroki instalacji

1. **Klonowanie repozytorium**:
   ```bash
   git clone https://github.com/username/trading-system.git
   cd trading-system
   ```

2. **Instalacja zależności**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Konfiguracja środowiska**:
   - Skopiuj plik `.env.example` do `.env`:
     ```bash
     cp .env.example .env
     ```
   - Wypełnij zmienne środowiskowe w pliku `.env` swoimi danymi, w tym kluczami API ByBit

4. **Uruchomienie aplikacji**:
   ```bash
   python main.py
   ```

5. **Dostęp do aplikacji**:
   Otwórz przeglądarkę i przejdź do adresu:
   ```
   http://localhost:5000
   ```

## ⚙️ Konfiguracja

### Zmienne środowiskowe

Główne parametry konfiguracyjne w pliku `.env`:

- `BYBIT_API_KEY` i `BYBIT_API_SECRET` - klucze dostępu do API ByBit
- `BYBIT_USE_TESTNET` - `true` dla testowej wersji, `false` dla produkcyjnej
- `RISK_LEVEL` - poziom ryzyka: `low`, `medium`, `high`
- `MAX_POSITION_SIZE` - maksymalny rozmiar pozycji jako % portfela
- `API_RATE_LIMIT` - włączenie inteligentnego limitowania zapytań API
- `API_CACHE_ENABLED` - włączenie cache'owania odpowiedzi API

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
1. Zmodyfikuj zmienną `BYBIT_USE_TESTNET` w pliku `.env` na `true`
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

## 📁 Struktura projektu

- `main.py` - Główny plik aplikacji
- `data/` - Moduły przetwarzania danych i integracji z API
  - `execution/` - Moduły wykonywania operacji na giełdzie
  - `indicators/` - Wskaźniki techniczne i analizy
  - `risk_management/` - Zarządzanie ryzykiem
  - `strategies/` - Strategie handlowe
  - `utils/` - Narzędzia pomocnicze
- `ai_models/` - Modele AI do analizy rynku
- `static/` - Pliki statyczne (CSS, JavaScript)
- `templates/` - Szablony HTML
- `logs/` - Logi aplikacji

## 📊 Przykłady użycia

### Monitorowanie portfela
Po uruchomieniu aplikacji, dashboard prezentuje aktualny stan portfela, historyczne wyniki oraz wykres wartości.

### Testowanie API
Możesz testować połączenie z API ByBit za pomocą endpointu:
```
http://localhost:5000/api/bybit/connection-test
```

### Debugowanie
Wszystkie logi znajdują się w katalogu `logs/`, co pomaga w diagnozowaniu problemów.

## 🔒 Bezpieczeństwo

- Klucze API są przechowywane tylko lokalnie w pliku `.env` (nie są wersjonowane)
- System wspiera tryb testnet - zalecany do testów i nauki
- Wbudowane limity zapytań API zapobiegają przekroczeniu limitów giełdy
- Moduł cache'owania zmniejsza liczbę zapytań i poprawia wydajność

## 📄 Licencja

Ten projekt jest dostępny na licencji MIT.