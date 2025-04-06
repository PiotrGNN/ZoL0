# 🤖 Trading Bot - System Analityczny

## 📋 Opis Projektu

Trading Bot to system analityczny do analizy danych rynkowych, wykrywania anomalii i symulacji strategii inwestycyjnych. Projekt jest zoptymalizowany do działania w środowisku Replit.

## 🚀 Szybki Start

1. **Instalacja zależności**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatywnie, użyj predefiniowanego workflow "Install Dependencies"

2. **Uruchomienie projektu**:
   ```bash
   python main.py
   ```
   Lub naciśnij przycisk "Run" w interfejsie Replit

3. **Uruchomienie testów**:
   ```bash
   pytest data/tests/
   ```
   Lub użyj workflow "Test"

## 🧰 Struktura Projektu

```
├── ai_models/              # Modele sztucznej inteligencji
│   ├── anomaly_detection.py  # Wykrywanie anomalii
│   └── ...
├── config/                 # Konfiguracja aplikacji
├── data/                   # Moduły przetwarzania danych
│   ├── data/               # Źródła danych i preprocessing
│   ├── execution/          # Wykonywanie transakcji
│   ├── indicators/         # Wskaźniki techniczne
│   ├── logging/            # Logowanie operacji
│   ├── optimization/       # Optymalizacja strategii
│   ├── risk_management/    # Zarządzanie ryzykiem
│   ├── strategies/         # Strategie inwestycyjne
│   ├── tests/              # Testy jednostkowe
│   └── utils/              # Funkcje pomocnicze
├── logs/                   # Pliki logów
├── saved_models/           # Zapisane modele AI
├── .env.example            # Przykładowe zmienne środowiskowe
├── requirements.txt        # Zależności projektu
└── main.py                 # Główny skrypt aplikacji
```

## 🛠️ Funkcjonalności

- Analiza historycznych danych giełdowych
- Wykrywanie anomalii cenowych i wolumenowych
- Symulacja strategii inwestycyjnych
- Analiza ryzyka i optymalizacja portfela
- Automatyczne raportowanie wyników

## 🔍 Workflow i Narzędzia

W projekcie skonfigurowano następujące workflowy:

1. **Run** - Uruchamia główny skrypt aplikacji
2. **Test** - Uruchamia testy jednostkowe
3. **Format Code** - Formatuje kod zgodnie z PEP 8 za pomocą Black
4. **Lint Code** - Sprawdza jakość kodu za pomocą Flake8
5. **Test with Coverage** - Uruchamia testy z raportem pokrycia
6. **Install Dependencies** - Instaluje wymagane zależności

## 🔑 Konfiguracja Środowiska

Przed uruchomieniem aplikacji, skopiuj plik `.env.example` do `.env` i uzupełnij wymagane zmienne środowiskowe:

```
# API Keys - Replace with your actual keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
```

## 📊 Przykładowe Użycie

```python
# Inicjalizacja modelu wykrywania anomalii
from ai_models.anomaly_detection import AnomalyDetectionModel

model = AnomalyDetectionModel(method='isolation_forest', contamination=0.05)
model.fit(data)
anomalies = model.detect_anomalies(data)
```

## 📝 Licencja

Ten projekt jest licencjonowany na warunkach MIT License.