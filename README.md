# ZoL0-1 Trading System

System tradingowy oparty o uczenie maszynowe i sztuczną inteligencję

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python main.py
```

## Zarządzanie modelami AI

System zawiera narzędzia do zarządzania i testowania modeli AI:

```bash
# Uruchomienie pełnych testów modeli
python test_models.py

# Wymuszenie ponownego trenowania podczas testów
python test_models.py --force-retrain

# Zarządzanie modelami AI
python manage_ai_models.py list    # Lista dostępnych modeli
python manage_ai_models.py clean   # Usuń uszkodzone modele
python manage_ai_models.py backup  # Utwórz backup modeli
python manage_ai_models.py test    # Szybki test modeli
```

## Funkcjonalności
- Analiza rynków kryptowalutowych
- Automatyczne generowanie strategii
- Backtesting
- Panel monitorowania
- Rozpoznawanie wzorców rynkowych
- Analiza sentymentu
- Wykrywanie anomalii

## Rozwiązane problemy
- Naprawiono błąd 'tolist()' w modelach AI poprzez dodanie bezpiecznej konwersji danych
- Poprawiono zapisywanie i odczytywanie modeli z plików .pkl
- Dodano obsługę niezgodnych kształtów danych w ReinforcementLearner
- Usprawniono walidację danych wejściowych do modeli
- Dodano narzędzia do zarządzania modelami AI i tworzenia kopii zapasowych

## 🚀 Funkcjonalność

- **Modele AI i uczenie maszynowe**: Przewidywanie ruchu cen i rozpoznawanie wzorców cenowych
- **Zarządzanie Ryzykiem**: Dynamiczny mechanizm zarządzania ryzykiem i portfelem
- **Analiza Techniczna**: Wskaźniki techniczne i analiza wolumenu
- **Dashboard Web**: Wizualizacja portfela, transakcji i rezultatów modeli AI
- **API Giełdowe**: Integracja z ByBit i możliwość dodania innych giełd
- **Symulacje**: Możliwość uruchomienia w trybie symulowanym z rzeczywistymi danymi

## 📋 Wymagania

- Python 3.8 lub nowszy
- Pakiety wymienione w `requirements.txt`
- Klucze API giełdy (opcjonalnie dla trybu rzeczywistego)

## 🔧 Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/twój-użytkownik/ZoL0-1.git
cd ZoL0-1
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

3. Skonfiguruj zmienne środowiskowe:
```bash
cp .env.example .env
# Edytuj plik .env, aby dodać klucze API
```

## 🖥 Uruchomienie

### Tryb Symulowany (domyślny)
```bash
python main.py
```

### Tryb Rzeczywisty (wymaga kluczy API)
```bash
python main.py --mode real
```

### Uruchomienie Testów
```bash
python test_models.py  # Test modeli AI
python test_environment.py  # Sprawdzenie środowiska
python test_data_conversion.py  # Test konwersji danych
```

## 🔍 Dashboard i Monitoring

System posiada wbudowany dashboard dostępny pod adresem http://localhost:5000 po uruchomieniu programu. Dashboard zawiera:

- Podsumowanie portfela i pozycji
- Wizualizacje transakcji i zysków/strat
- Status modeli AI
- Analityki w czasie rzeczywistym

## 🏗 Struktura Projektu

```
ZoL0-1/
├── ai_models/            # Modele AI i uczenia maszynowego
├── data/                 # Operacje na danych i komunikacja z API
│   ├── cache/            # Dane tymczasowe i cache
│   ├── data/             # Przetwarzanie danych
│   ├── execution/        # Realizacja transakcji
│   ├── indicators/       # Wskaźniki techniczne
│   ├── logging/          # System logowania
│   ├── optimization/     # Optymalizacja strategii
│   ├── risk_management/  # Zarządzanie ryzykiem
│   ├── strategies/       # Strategie handlowe
│   ├── tests/            # Testy modułów
│   └── utils/            # Narzędzia pomocnicze
├── logs/                 # Pliki logów
├── models/               # Zapisane modele ML
├── python_libs/          # Biblioteki pomocnicze
├── reports/              # Raporty i analizy
├── saved_models/         # Zapisane i wytrenowane modele
├── static/               # Pliki statyczne dla dashboardu
│   ├── css/
│   ├── img/
│   └── js/
└── templates/            # Szablony HTML dla dashboardu
```

## 📊 Zarządzanie Portfelem

System oferuje:

- Symulowany lub rzeczywisty handel
- Podgląd stanu portfela w czasie rzeczywistym
- Szczegółowe metryki wydajności (ROI, drawdown, profit factor)
- Śledzenie wszystkich transakcji w logach

## 📃 Logi i Historia Transakcji

Wszystkie operacje są rejestrowane w folderze `logs/`:

- `app.log` - Ogólne logi aplikacji
- `portfolio.log` - Logi portfela
- `trade.log` - Historia transakcji
- `model_tests.log` - Testy modeli AI
- `detailed_trades.json` - Szczegółowa historia w formacie JSON

## 🔐 Bezpieczeństwo

- Klucze API są przechowywane tylko w pliku `.env` (niewersjonowanym)
- Domyślnie używany jest tryb symulowany, który nie wymaga kluczy API
- Tryb rzeczywisty wymaga dodatkowego potwierdzenia

## 🛠 Konfiguracja

Główna konfiguracja znajduje się w `config/settings.py`. Możesz dostosować:

- Parametry ryzyka
- Ustawienia handlowe (prowizje, limity zleceń)
- Ustawienia modeli AI
- API i środowisko

## 📜 Licencja

[MIT License](LICENSE)