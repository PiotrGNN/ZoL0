# ZoL0-1: System Tradingowy z AI

## 🔍 Opis Projektu
ZoL0-1 to zaawansowany system tradingowy wykorzystujący sztuczną inteligencję do analizy rynków finansowych, rozpoznawania formacji cenowych i podejmowania decyzji inwestycyjnych.

## 🚀 Główne Funkcjonalności
- **Modele AI**: Głębokie uczenie, rozpoznawanie wzorców, analiza sentymentu
- **Trading Algorytmiczny**: Strategie bazujące na analizie technicznej i AI
- **Zarządzanie Ryzykiem**: Zaawansowane algorytmy kontroli ryzyka
- **Dashboard**: Interaktywny interfejs do monitorowania rynku i wyników

## 📚 Architektura Systemu
- **ai_models/**: Modele uczenia maszynowego i głębokiego
- **data/**: Obsługa danych rynkowych i wykonywania zleceń
- **python_libs/**: Podstawowe komponenty systemowe
- **static/**: Zasoby dla interfejsu webowego
- **templates/**: Szablony HTML dla dashboardu

## 🛠️ Instalacja i Uruchomienie

### Wymagania
- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, scikit-learn
- Flask

### Polecenia
1. Instalacja zależności: `pip install -r requirements.txt`
2. Uruchomienie systemu: `python main.py`
3. Testowanie modeli: `python test_models.py`
4. Czyszczenie cache: `python -c "from data.utils.cache_manager import clean_old_data; clean_old_data()"`

### Tryby Pracy
- **Symulacja**: Domyślny tryb bez realnych transakcji
- **Testnet**: Połączenie z API ByBit Testnet
- **Produkcja**: ⚠️ Rzeczywiste transakcje z portfelem

## 📊 Dashboard

Dashboard jest dostępny po uruchomieniu systemu pod adresem:
```
http://localhost:5000/
```

## 🧠 Modele AI

System wykorzystuje następujące modele AI:
- **ReinforcementLearner**: Uczenie ze wzmocnieniem dla decyzji tradingowych
- **SentimentAnalyzer**: Analiza nastrojów rynkowych
- **ModelRecognizer**: Rozpoznawanie formacji cenowych
- **AnomalyDetector**: Wykrywanie anomalii rynkowych

## 🔐 Bezpieczeństwo

⚠️ **WAŻNE**: Przed użyciem w trybie produkcyjnym:
1. Zabezpiecz klucze API w pliku `.env`
2. Ustaw odpowiednie limity środków
3. Rozpocznij od małych transakcji testowych

## 📝 Licencja

Ten projekt jest dostępny na licencji MIT.

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

## Struktura Projektu

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

## Zarządzanie Portfelem

System oferuje:

- Symulowany lub rzeczywisty handel
- Podgląd stanu portfela w czasie rzeczywistym
- Szczegółowe metryki wydajności (ROI, drawdown, profit factor)
- Śledzenie wszystkich transakcji w logach

## Logi i Historia Transakcji

Wszystkie operacje są rejestrowane w folderze `logs/`:

- `app.log` - Ogólne logi aplikacji
- `portfolio.log` - Logi portfela
- `trade.log` - Historia transakcji
- `model_tests.log` - Testy modeli AI
- `detailed_trades.json` - Szczegółowa historia w formacie JSON

## Bezpieczeństwo

- Klucze API są przechowywane tylko w pliku `.env` (niewersjonowanym)
- Domyślnie używany jest tryb symulowany, który nie wymaga kluczy API
- Tryb rzeczywisty wymaga dodatkowego potwierdzenia

## Konfiguracja

Główna konfiguracja znajduje się w `config/settings.py`. Możesz dostosować:

- Parametry ryzyka
- Ustawienia handlowe (prowizje, limity zleceń)
- Ustawienia modeli AI
- API i środowisko