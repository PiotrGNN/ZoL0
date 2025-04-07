
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
├── main.py                # Główny plik wykonawczy
└── requirements.txt       # Zależności projektu
```

## ⚙️ Konfiguracja

1. Skopiuj plik `.env.example` do `.env`
2. Uzupełnij wymagane klucze API i inne ustawienia:
    - `BINANCE_API_KEY` - klucz API Binance
    - `BINANCE_API_SECRET` - sekret API Binance

## 🧪 Testy

```bash
# Uruchomienie testów
pytest data/tests/

# Testy z raportowaniem pokrycia
pytest --cov=. --cov-report=term-missing data/tests/
```

## 🔧 Narzędzia deweloperskie

```bash
# Formatowanie kodu
black .

# Sprawdzanie zgodności ze stylem
flake8 .
```

## 📜 Licencja

Ten projekt jest udostępniany na licencji MIT.
