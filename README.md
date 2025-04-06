# 🚀 Advanced AI Trading System

System do automatycznego tradingu z wykorzystaniem zaawansowanych algorytmów AI i ML.

## 📋 Funkcje

- 🤖 **AI/ML Trading** - Wykorzystanie modeli uczenia maszynowego do predykcji ruchu cen
- 📊 **Wykrywanie anomalii** - Identyfikacja nietypowych wzorców rynkowych
- 📈 **Optymalizacja strategii** - Automatyczne dostosowanie parametrów strategii
- 🔄 **Backtesting** - Testowanie strategii na danych historycznych
- 🛡️ **Zarządzanie ryzykiem** - Zaawansowane mechanizmy kontroli ryzyka

## 🚀 Szybki start (Replit)

1. Skonfiguruj plik `.env` (skopiuj z `.env.example`):
   ```
   cp .env.example .env
   ```

2. Kliknij przycisk ▶️ **Run** lub wybierz workflow "Start Trading Bot" z menu

## 🔧 Konfiguracja środowiska

1. Zainstaluj wymagane zależności:
   ```
   pip install -r requirements.txt
   ```

2. Uruchom system tradingowy:
   ```
   python main.py
   ```

3. Dostępne workflow:
   - **Start Trading Bot** - Uruchamia główny system tradingowy
   - **Run Tests** - Uruchamia testy jednostkowe
   - **Lint Code** - Sprawdza jakość kodu

## 🛠️ Struktura projektu

```
├── ai_models/            # Modele AI i uczenia maszynowego
├── config/               # Pliki konfiguracyjne
├── data/                 # Moduły przetwarzania danych i strategii
│   ├── data/             # Pobieranie i przetwarzanie danych
│   ├── execution/        # Logika wykonywania zleceń
│   ├── indicators/       # Wskaźniki techniczne
│   ├── logging/          # Narzędzia logowania
│   ├── optimization/     # Optymalizacja strategii
│   ├── risk_management/  # Moduły kontroli ryzyka
│   ├── strategies/       # Strategie tradingowe
│   ├── tests/            # Testy jednostkowe
│   └── utils/            # Funkcje pomocnicze
├── logs/                 # Pliki logów
├── saved_models/         # Zapisane wytrenowane modele
└── main.py               # Główny punkt wejścia
```

## 📝 Wymagania

- Python 3.8+
- Biblioteki wymienione w `requirements.txt`
- Klucze API do giełd kryptowalut (dla handlu rzeczywistego)

## 🧪 Rozwój i testowanie

### Uruchomienie testów
```
python -m pytest data/tests/
```

### Formatowanie kodu
```
black .
```

### Sprawdzanie stylu kodu
```
flake8 .
```

## 📄 Licencja

Ten projekt jest udostępniany na licencji MIT. Zobacz plik LICENSE dla szczegółów.