
# 🤖 Trading Bot - System Analityczny

System do analizy rynków finansowych z wykorzystaniem algorytmów uczenia maszynowego i sztucznej inteligencji, zoptymalizowany do działania w środowisku Replit.

## 🚀 Funkcjonalności

- Pobieranie i przetwarzanie danych giełdowych
- Wykrywanie anomalii cenowych i wolumenowych
- Symulacja strategii inwestycyjnych
- Analiza ryzyka i optymalizacja portfela
- Automatyczne raportowanie wyników

## 🔧 Instalacja i konfiguracja

### Wymagania systemowe

Projekt jest skonfigurowany do automatycznego działania w środowisku Replit. W przypadku uruchamiania lokalnie, wymagane są:

- Python 3.8+
- Pakiety wymienione w pliku `requirements.txt`

### Konfiguracja środowiska

1. **Zmienne środowiskowe**:
   Skopiuj plik `.env.example` do `.env` i uzupełnij wymagane zmienne:

   ```bash
   cp .env.example .env
   ```

2. **Instalacja zależności**:
   Zależności zostaną automatycznie zainstalowane przy pierwszym uruchomieniu. Możesz też zainstalować je ręcznie:

   ```bash
   pip install -r requirements.txt
   ```

## 📊 Uruchamianie systemu

### W środowisku Replit

Naciśnij przycisk ▶️ **Run** aby uruchomić system.

### Lokalne uruchomienie

```bash
python main.py
```

## 🧪 Testy

Uruchom testy jednostkowe:

```bash
pytest data/tests/
```

Testy z raportem pokrycia:

```bash
pytest --cov=. --cov-report=term-missing data/tests/
```

## 🎛️ Dostępne workflowy

W projekcie skonfigurowano następujące workflowy:

1. **Run** - Uruchamia główny skrypt aplikacji
2. **Test** - Uruchamia testy jednostkowe
3. **Format Code** - Formatuje kod zgodnie z PEP 8 za pomocą Black
4. **Lint Code** - Sprawdza jakość kodu za pomocą Flake8
5. **Test with Coverage** - Uruchamia testy z raportem pokrycia
6. **Install Dependencies** - Instaluje wymagane zależności

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

## 🛠️ Narzędzia deweloperskie

- **Linter**: Flake8 (konfiguracja w `.flake8`)
- **Formatter**: Black
- **Testy**: pytest

## 📝 Licencja

Ten projekt jest udostępniany na licencji MIT.
