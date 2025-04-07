# 🚀 Trading System z wykorzystaniem AI

Zaawansowany system tradingowy wykorzystujący modele sztucznej inteligencji do wykrywania anomalii rynkowych, predykcji trendów i automatycznego zarządzania transakcjami.

## 📋 Wymagania systemowe

- Python 3.8+
- Pakiety wymienione w `requirements.txt`

## 🚀 Szybki start

### Uruchomienie w Replit

1. Kliknij przycisk "Run" w środowisku Replit
2. System zostanie automatycznie skonfigurowany i uruchomiony

### Manualna instalacja

```bash
# Instalacja zależności
pip install -r requirements.txt

# Uruchomienie aplikacji
python main.py
```

## 🏗️ Struktura projektu

```
.
├── ai_models/                # Modele AI i uczenia maszynowego
├── config/                   # Pliki konfiguracyjne
├── data/                     # Przetwarzanie i zarządzanie danymi
│   ├── data/                 # Pobieranie i przetwarzanie danych
│   ├── execution/            # Wykonywanie transakcji
│   ├── indicators/           # Wskaźniki techniczne
│   ├── logging/              # Logowanie zdarzeń
│   ├── optimization/         # Optymalizacja strategii
│   ├── risk_management/      # Zarządzanie ryzykiem
│   ├── strategies/           # Strategie inwestycyjne
│   ├── tests/                # Testy jednostkowe
│   └── utils/                # Narzędzia pomocnicze
├── logs/                     # Logi aplikacji
├── saved_models/             # Zapisane modele ML
├── main.py                   # Główny punkt wejścia
└── requirements.txt          # Zależności projektu
```

## 🎛️ Dostępne workflowy

W projekcie skonfigurowano następujące workflowy:

1. **Run** - Uruchamia główny skrypt aplikacji
2. **Test** - Uruchamia testy jednostkowe
3. **Format Code** - Formatuje kod zgodnie z PEP 8 za pomocą Black
4. **Lint Code** - Sprawdza jakość kodu za pomocą Flake8
5. **Test with Coverage** - Uruchamia testy z raportem pokrycia
6. **Install Dependencies** - Instaluje wymagane zależności

## 🧪 Testowanie

```bash
# Uruchomienie wszystkich testów
pytest data/tests/

# Uruchomienie testów z raportem pokrycia
pytest --cov=. --cov-report=term-missing data/tests/
```

## 🔧 Narzędzia developerskie

- **Linter**: Flake8 - sprawdza jakość i zgodność kodu z PEP 8
- **Formatter**: Black - automatycznie formatuje kod zgodnie z PEP 8
- **Testy**: Pytest - framework do testów jednostkowych

## 📊 Funkcjonalności

1. **Wykrywanie anomalii rynkowych** - identyfikacja nietypowych wzorców cenowych
2. **Zarządzanie ryzykiem** - dynamiczne dostosowanie wielkości pozycji i stop-lossów
3. **Strategie inwestycyjne** - implementacja różnych strategii (trend following, mean reversion)
4. **Optymalizacja strategii** - backtesting i optymalizacja hiperparametrów
5. **Wykonywanie transakcji** - integracja z różnymi giełdami przez API

## 📝 Licencja

Ten projekt jest dostępny na licencji MIT.