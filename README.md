# ZoL0-1 Trading System

## 📊 System AI Trading z analizą rynku i inteligentnym zarządzaniem portfelem

### 🔧 Zoptymalizowane moduły

System został znacząco zoptymalizowany w następujących obszarach:

#### ✅ Naprawione problemy
- **Naprawiono błędy tolist()** w ModelRecognizer i AnomalyDetector
- **Rozwiązano problemy z kompilacją modelu Keras** w ReinforcementLearner
- **Naprawiono niezgodności wymiarów** w danych treningowych
- **Dodano automatyczne zapisywanie modeli** po trenowaniu

#### 🧠 Rozszerzenia AI i zarządzanie modelami
- Dodano narzędzia do **zarządzania modelami** (zapisywanie, ładowanie, checkpointy)
- Wdrożono **automatyczną walidację danych** przed trenowaniem
- Usprawniony `ModelTester` z obsługą cache - modele nie są retrenowane zbyt często
- Dodano funkcje bezpieczeństwa dla ładowania i zapisywania modeli

### 🚀 Jak używać

1. **Uruchomienie całego systemu:**
   ```
   python main.py
   ```

2. **Testowanie/trenowanie modeli AI:**
   ```
   python test_models.py
   ```

3. **Zarządzanie modelami AI:**
   ```
   python manage_ai_models.py list
   python manage_ai_models.py retrain
   python manage_ai_models.py backup
   python manage_ai_models.py details <nazwa_modelu>
   ```

### 📊 Dostępne modele AI

- **ModelRecognizer** - Rozpoznaje formacje rynkowe (np. "Head and Shoulders", "Bull Flag")
- **AnomalyDetector** - Wykrywa anomalie cenowe i wolumenowe
- **ReinforcementLearner** - Model uczenia przez wzmocnienie do optymalizacji decyzji
- **SentimentAnalyzer** - Analizuje sentyment rynkowy z różnych źródeł

### 📋 Instrukcja instalacji

1. Zainstaluj wymagane pakiety:
   ```
   pip install -r requirements.txt
   ```

2. Upewnij się, że katalogi istnieją:
   ```
   mkdir -p models logs saved_models/checkpoints
   ```

3. Uruchom system:
   ```
   python main.py
   ```

### 📝 Uwagi

- System używa cache do ograniczenia liczby wywołań API i unikania zbyt częstego trenowania
- Modele są zapisywane po trenowaniu wraz z metadanymi (accuracy, timestamp, itp.)
- Dashboard jest dostępny pod adresem http://0.0.0.0:5000/ po uruchomieniu systemu


## 🛠️ Instalacja i Uruchomienie

### Wymagania
- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, scikit-learn
- Flask


## 🔐 Bezpieczeństwo

⚠️ **WAŻNE**: Przed użyciem w trybie produkcyjnym:
1. Zabezpiecz klucze API w pliku `.env`
2. Ustaw odpowiednie limity środków
3. Rozpocznij od małych transakcji testowych

## 📝 Licencja

Ten projekt jest dostępny na licencji MIT.

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