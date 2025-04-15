# 🚀 ZoL0-1: System Tradingowy z AI

## 📋 Opis projektu
ZoL0-1 to zaawansowany system tradingowy wykorzystujący różne modele sztucznej inteligencji do analizy rynków finansowych. System integruje analizę techniczną, sentiment oraz algorytmy uczenia maszynowego do generowania sygnałów tradingowych.

## 🛠️ Funkcje
- Wykrywanie anomalii rynkowych
- Analiza sentymentu na podstawie danych tekstowych (dokładność ~82%)
- Przewidywanie cen na podstawie RandomForest z automatycznym zapisem modeli
- Rozpoznawanie wzorców rynkowych (ModelRecognizer) z walidacją danych
- Dashboard webowy do monitorowania stanu systemu
- Zarządzanie ryzykiem i portfelem
- Integracja z giełdą ByBit
- Inteligentne buforowanie danych z automatycznym czyszczeniem

## 🧠 Status modeli AI
| Model | Status | Dokładność | Zapisywanie |
|-------|--------|------------|-------------|
| RandomForestRegressor | ✅ Działa | Zmienna | Automatyczne (.pkl) |
| SentimentAnalyzer | ✅ Działa | ~82% | Automatyczne (.pkl) |
| Sequential (Keras) | ✅ Naprawiony | Zmienna | Automatyczne (.h5) |
| ModelRecognizer | ✅ Działa | Wysoka | Nie dotyczy |
| AnomalyDetector | ✅ Działa | Zmienna | Nie dotyczy |

## 📦 Wymagania
- Python 3.8+
- Biblioteki zainstalowane z pliku requirements.txt
- Przestrzeń dyskowa na cache i modele (~500MB)

## 🚀 Uruchomienie
System można uruchomić na dwa sposoby:

### 1. Uruchomienie pełnego systemu
```bash
python main.py
```
Uruchamia backend wraz z API webowym.

### 2. Uruchomienie testów modeli
```bash
python test_models.py
```
Testuje modele AI w systemie.

### 3. Uruchomienie testów konwersji danych
```bash
python test_data_conversion.py
```
Testuje poprawność konwersji danych między różnymi formatami.

## 📊 Dashboard
System udostępnia dashboard webowy dostępny pod adresem: `http://localhost:5000`

## 🧪 Modele AI
- **RandomForestRegressor** - przewidywanie cen
- **Sequential (Keras)** - uczenie ze wzmocnieniem
- **SentimentAnalyzer** - analiza sentymentu
- **AnomalyDetector** - wykrywanie anomalii
- **ModelRecognizer** - rozpoznawanie wzorców rynkowych

## 📁 Struktura projektu
```
├── ai_models/            # Modele AI do analizy rynku
├── data/                 # Komponenty przetwarzania danych
│   ├── cache/            # Dane cache
│   ├── execution/        # Moduły wykonywania transakcji
│   ├── indicators/       # Wskaźniki techniczne
│   ├── logging/          # Logowanie operacji
│   ├── optimization/     # Optymalizacja strategii
│   ├── risk_management/  # Zarządzanie ryzykiem
│   ├── strategies/       # Strategie tradingowe
│   ├── tests/            # Testy
│   └── utils/            # Narzędzia pomocnicze
├── logs/                 # Pliki logów
├── python_libs/          # Uproszczone/specjalne biblioteki
├── static/               # Statyczne zasoby web
├── templates/            # Szablony HTML
├── main.py               # Główny punkt wejścia aplikacji
├── models/               # Zapisane modele w formacie .pkl
├── config/               # Pliki konfiguracyjne
├── requirements.txt      # Zależności projektu
└── test_data_conversion.py # Testy konwersji danych

```

## 🔧 Konfiguracja
Ustawienia znajdują się w katalogu `config/`. Skopiuj `.env.example` do `.env` i dostosuj parametry.

## 📝 Logi
Logi systemu zapisywane są w katalogu `logs/`.

## 🤝 Autorzy
- ZoL0-1 Team

## 📄 Licencja
Copyright © 2025