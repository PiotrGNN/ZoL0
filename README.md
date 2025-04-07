
# 🤖 Trading Bot System

## Opis projektu
System tradingowy oparty na sztucznej inteligencji, zoptymalizowany do działania w środowisku Replit. System umożliwia analizę rynku, wykrywanie anomalii cenowych, generowanie sygnałów tradingowych oraz przeprowadzanie symulacji strategii inwestycyjnych.

## 🔑 Główne funkcje

- **Dashboard analityczny** - wizualizacja danych rynkowych, statusów komponentów i wykrytych anomalii
- **Detekcja anomalii** - wykrywanie nietypowych zachowań rynku z wykorzystaniem algorytmów ML
- **Analiza sentymentu** - ocena wpływu informacji rynkowych na ceny instrumentów
- **Strategie tradingowe** - zestaw zaimplementowanych strategii (trend following, mean reversion, breakout)
- **Zarządzanie ryzykiem** - zaawansowane mechanizmy kontroli ryzyka i optymalizacji portfela
- **System powiadomień** - alerty o istotnych zdarzeniach rynkowych i błędach systemu

## 🧪 Tryby pracy

System może działać w trzech trybach:
1. **Symulacja** - symulowanie danych rynkowych i działania strategii (domyślny)
2. **Analiza** - analiza historycznych danych bez wykonywania transakcji
3. **Testowanie** - uruchamianie testów jednostkowych i integracyjnych

## 📋 Struktura projektu

```
├── ai_models/              # Modele AI/ML do analizy i predykcji
├── data/                   # Moduły danych, strategie i narzędzia
│   ├── data/               # Pobieranie i przetwarzanie danych
│   ├── execution/          # Wykonywanie transakcji
│   ├── indicators/         # Wskaźniki techniczne i analizy
│   ├── logging/            # System logowania i detekcji anomalii
│   ├── optimization/       # Optymalizacja strategii i backtesting
│   ├── risk_management/    # Zarządzanie ryzykiem
│   ├── strategies/         # Strategie tradingowe
│   ├── tests/              # Testy jednostkowe i integracyjne
│   └── utils/              # Narzędzia pomocnicze
├── logs/                   # Pliki logów
├── reports/                # Raporty i analizy
├── saved_models/           # Zapisane modele ML
├── static/                 # Pliki statyczne dla interfejsu
│   ├── css/                # Style CSS
│   └── js/                 # Skrypty JavaScript
├── templates/              # Szablony HTML
├── .env.example            # Przykładowy plik konfiguracyjny
└── main.py                 # Główny plik uruchomieniowy
```

## 🚀 Uruchomienie projektu

1. **Instalacja zależności**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Konfiguracja środowiska**:
   - Skopiuj `.env.example` do `.env` i uzupełnij zmienne środowiskowe
   - Dostosuj ustawienia w `config/settings.py` według potrzeb

3. **Uruchomienie systemu**:
   ```bash
   python main.py
   ```

4. **Dostęp do dashboardu**:
   - Po uruchomieniu, dashboard jest dostępny pod adresem: `http://localhost:5000/dashboard`

## 🔧 Dostępne workflowy

- **Run** - uruchamia główny system
- **Test** - wykonuje testy jednostkowe
- **Format Code** - formatuje kod za pomocą black
- **Lint Code** - sprawdza jakość kodu za pomocą flake8
- **Test with Coverage** - uruchamia testy z pomiarem pokrycia kodu
- **Install Dependencies** - instaluje wymagane zależności

## 📊 Funkcje API

- `/api/chart-data` - zwraca dane do wykresów
- `/api/system-status` - zwraca status systemu
- `/api/components` - zwraca status poszczególnych komponentów
- `/api/anomalies` - zwraca wykryte anomalie

## 🔍 Diagnostyka i debugowanie

W przypadku problemów, sprawdź:
- Logi w katalogu `logs/`
- Status komponentów w dashboardzie
- Błędy importów i zależności

## 📦 Zależności

Główne biblioteki:
- Flask - framework webowy
- Pandas/NumPy - przetwarzanie danych
- Scikit-learn - modele ML do analizy i predykcji
- Matplotlib/Chart.js - wizualizacja danych

## 👨‍💻 Rozwój projektu

System jest zaprojektowany modułowo, co umożliwia łatwe dodawanie nowych:
- Strategii tradingowych
- Modeli AI/ML
- Wskaźników technicznych
- Mechanizmów zarządzania ryzykiem

## ⚠️ Uwagi

System działa domyślnie w trybie symulacji i nie wykonuje rzeczywistych transakcji.
Aby włączyć tryb handlu rzeczywistego, należy dostosować odpowiednie parametry w pliku konfiguracyjnym.
