# Trading Engine - System Tradingowy 🚀

## 📋 Spis treści
- [Instalacja](#-instalacja)
- [Uruchomienie](#-uruchomienie)
- [Struktura projektu](#-struktura-projektu)
- [Rozwiązywanie problemów](#-rozwiązywanie-problemów)
- [Komponenty systemu](#-komponenty-systemu)

## 🔧 Instalacja

### W środowisku Replit
Projekt jest skonfigurowany do pracy w środowisku Replit. Wszystkie niezbędne pakiety są określone w `replit.nix` i zostaną automatycznie zainstalowane.

1. **Przygotowanie środowiska**:
   - W Replit wybierz przycisk `Run` lub uruchom workflow `Init Project` w panelu `Shell`
   - Uruchomienie `init_project.py` automatycznie stworzy wymagane katalogi i pliki konfiguracyjne

2. **Konfiguracja API**:
   - Utwórz plik `.env` w głównym katalogu projektu (lub użyj narzędzia Secrets w Replit)
   - Dodaj klucze API Bybit zgodnie z plikiem `.env.example`

## 🚀 Uruchomienie

### Workflow'y (zalecane)
W Replit dostępne są gotowe workflow:

- **Run/Run Server** - uruchamia aplikację Flask
- **Test Bybit Connection** - testuje połączenie z API Bybit
- **Init Project** - inicjalizuje projekt

### Ręczne uruchomienie
Alternatywnie możesz uruchomić aplikację ręcznie:

```bash
python3 run_server.py
```

## 📁 Struktura projektu

```
trading_engine/
├── data/                  # Główny pakiet danych
│   ├── execution/         # Komunikacja z giełdą (Bybit)
│   ├── indicators/        # Wskaźniki techniczne i analizy
│   ├── strategies/        # Strategie tradingowe
│   └── utils/             # Narzędzia pomocnicze
├── ai_models/             # Modele uczenia maszynowego
├── static/                # Pliki statyczne dla interfejsu
├── templates/             # Szablony HTML
├── main.py                # Główny plik aplikacji
└── run_server.py          # Skrypt uruchamiający serwer
```

## 🔍 Rozwiązywanie problemów

### Problemy z portami
Jeśli występuje problem z portem 5000/8080, skrypt `run_server.py` automatycznie znajdzie dostępny port.

### Problemy z API Bybit
Uruchom test połączenia:
```bash
python3 test_bybit_connection.py
```

### Problemy z importami
Jeśli występują problemy z importami modułów:
1. Sprawdź czy pliki `__init__.py` istnieją w każdym katalogu pakietu
2. Uruchom aplikację z flagą debug: `python3 -m pdb run_server.py`

## 🧩 Komponenty systemu

- **Dashboard** - interfejs webowy (Flask) pokazujący stan systemu
- **BybitConnector** - komunikacja z API Bybit
- **Strategie tradingowe** - logika podejmowania decyzji inwestycyjnych
- **Modele AI** - predykcja ruchów cen i analiza sentymentu
- **System powiadomień** - alerty i komunikaty
- **Risk management** - zarządzanie ryzykiem i wielkością pozycji

## 📦 Zależności
Główne biblioteki (z oryginalnego projektu, może wymagać aktualizacji):
- Flask - framework webowy
- Pandas/NumPy - przetwarzanie danych
- Scikit-learn - modele ML do analizy i predykcji
- Matplotlib/Chart.js - wizualizacja danych
- ByBit API client library (dodatkowa biblioteka do integracji z ByBit)

## 📝 Licencja

Ten projekt jest dostępny na licencji MIT.