
# Trading Bot - System Analityczny

## 🚀 Opis projektu

System analityczny do wykrywania anomalii i optymalizacji strategii tradingowych. Projekt jest w pełni skonfigurowany do działania w środowisku Replit.

## 📋 Funkcjonalności

- **Wykrywanie anomalii** - identyfikacja nietypowych zachowań rynku
- **Analiza techniczna** - wskaźniki techniczne i ich interpretacja
- **Zarządzanie ryzykiem** - strategie zarządzania kapitałem
- **Optymalizacja strategii** - testowanie i dostrajanie strategii tradingowych

## 🛠️ Instalacja

```bash
# Klonowanie repozytorium w Replit
git clone <URL repozytorium>

# Instalacja zależności
pip install -r requirements.txt
```

## 🚀 Uruchomienie

Projekt można uruchomić na dwa sposoby:

1. **Poprzez przycisk Run w Replit**
2. **Ręcznie przez terminal:**

```bash
python main.py
```

## 📁 Struktura projektu

```
├── ai_models/             # Modele uczenia maszynowego
│   ├── anomaly_detection.py  # Wykrywanie anomalii
│   └── ...
├── data/                  # Moduły przetwarzania danych
│   ├── indicators/        # Wskaźniki techniczne
│   ├── risk_management/   # Zarządzanie ryzykiem
│   ├── strategies/        # Strategie tradingowe
│   └── ...
├── logs/                  # Logi aplikacji
├── .env.example           # Przykładowy plik konfiguracyjny
├── main.py                # Punkt wejścia aplikacji
└── requirements.txt       # Zależności projektu
```

## 🔧 Konfiguracja

Kopia pliku `.env.example` do `.env` i dostosowanie zmiennych środowiskowych:

```
# API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Trading parameters
RISK_LEVEL=low
MAX_POSITION_SIZE=0.1
ENABLE_AUTO_TRADING=false
```

## 🧪 Testowanie

Uruchomienie testów:

```bash
# Wszystkie testy
pytest data/tests/

# Testy z pokryciem kodu
pytest --cov=. --cov-report=term-missing data/tests/
```

## 🛠️ Narzędzia developerskie

```bash
# Formatowanie kodu
black .

# Analiza statyczna
flake8 .
```

## 📜 Licencja

Ten projekt jest dostępny na licencji MIT.
