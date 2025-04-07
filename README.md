
# 🚀 Trading Bot System z wykorzystaniem AI

Zaawansowany system tradingowy wykorzystujący modele sztucznej inteligencji do wykrywania anomalii rynkowych, predykcji trendów i automatycznego zarządzania transakcjami.

## 📋 Funkcjonalności

- **Detekcja anomalii rynkowych** - Wykorzystanie uczenia maszynowego do wykrywania nietypowych zachowań rynku
- **Analiza sentymentu** - Monitorowanie mediów społecznościowych i newsów w celu oceny nastrojów rynkowych
- **Dashboard monitorujący** - Interaktywny interfejs użytkownika do śledzenia pracy systemu
- **Zarządzanie ryzykiem** - Automatyczne dostosowanie pozycji do warunków rynkowych
- **Strategie tradingowe** - Implementacja różnych strategii (trend following, mean reversion, breakout)
- **Optymalizacja hiperparametrów** - Automatyczne dostrajanie parametrów modeli

## 🚀 Szybki start

### Uruchomienie w Replit

1. Kliknij przycisk "Run" w środowisku Replit
2. System zostanie automatycznie skonfigurowany i uruchomiony
3. Dostęp do dashboardu: `https://[nazwa-repl].replit.app/dashboard`

### Manualna instalacja

```bash
# Instalacja zależności
pip install -r requirements.txt

# Uruchomienie aplikacji
python main.py
```

## 📚 Struktura projektu

```
├── ai_models/              # Modele AI i uczenia maszynowego
├── config/                 # Pliki konfiguracyjne
├── data/                   # Logika biznesowa i komponenty 
│   ├── indicators/         # Wskaźniki techniczne i analizy sentymentu
│   ├── logging/            # Komponenty logowania
│   ├── optimization/       # Optymalizatory
│   ├── risk_management/    # Zarządzanie ryzykiem
│   ├── strategies/         # Strategie tradingowe
│   └── utils/              # Narzędzia pomocnicze
├── logs/                   # Logi systemowe
├── reports/                # Generowane raporty
├── saved_models/           # Zapisane modele ML
├── static/                 # Pliki statyczne dla interfejsu
│   ├── css/                # Style CSS
│   └── js/                 # Skrypty JavaScript
├── templates/              # Szablony HTML dla dashboardu
├── .env.example            # Przykładowe zmienne środowiskowe
├── main.py                 # Główny plik aplikacji
└── requirements.txt        # Zależności projektu
```

## 🔧 Konfiguracja

System można skonfigurować poprzez zmienne środowiskowe lub plik `.env`:

```
# API Keys - Replace with your actual keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Environment settings
APP_ENV=development  # production, development, test

# Trading parameters
RISK_LEVEL=low  # low, medium, high
ENABLE_AUTO_TRADING=false
```

## 🖥️ Dostępne endpointy

- `/` - Przekierowanie do dashboardu
- `/dashboard` - Panel administracyjny z wizualizacją stanu systemu
- `/api/status` - Szczegółowy status API (JSON)
- `/health` - Endpoint do monitorowania stanu aplikacji
- `/api/chart-data` - Dane do wykresów (JSON)
- `/start-simulation` - Uruchomienie symulacji (API)
- `/download-report` - Wygenerowanie i pobranie raportu

## 🧪 Testowanie

System zawiera testy jednostkowe i integracyjne:

```bash
# Uruchomienie wszystkich testów
pytest data/tests/

# Testy z pokryciem kodu
pytest --cov=. --cov-report=term-missing data/tests/
```

## 📝 Workflow Development

Dostępne workflow dla Replit:

1. **Run** - Uruchamia główną aplikację
2. **Test** - Uruchamia testy
3. **Format Code** - Formatuje kod za pomocą black
4. **Lint Code** - Sprawdza jakość kodu przez flake8
5. **Test with Coverage** - Uruchamia testy z raportem pokrycia
6. **Install Dependencies** - Instaluje zależności

## 🔒 Bezpieczeństwo

Zawsze używaj zmiennych środowiskowych do przechowywania poufnych danych jak klucze API. Nigdy nie przechowuj ich w kodzie. Użyj pliku `.env` lokalnie lub narzędzia "Secrets" w Replit.

## 📄 Licencja

Ten projekt jest udostępniony na licencji MIT.
