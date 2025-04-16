# ZoL0 Trading System

## Opis
Zaawansowany system tradingowy z modułami AI, analizą techniczną i dashboardem monitorującym.

## Funkcje główne
- Śledzenie danych rynkowych w czasie rzeczywistym
- Integracja z giełdami (Binance, Bybit, CCXT)
- Modele AI do analizy rynku
- Backtesting strategii tradingowych
- Dashboard z wizualizacją danych

## Uruchamianie systemu

System składa się z dwóch głównych komponentów:
1. **Backend API** - serwer Flask obsługujący dane dla dashboardu
2. **Dashboard** - interfejs użytkownika zbudowany w Streamlit

### Uruchamianie w trybie developerskim

```bash
# Instalacja zależności
pip install -r requirements.txt

# Uruchomienie backendu
python run.py    # Uruchamia główny system z API

# Uruchomienie dashboardu (w osobnym terminalu)
streamlit run dashboard.py
```

### Zarządzanie modelami AI

```bash
# Generowanie nowych modeli AI
python create_models.py

# Testowanie modeli AI
python test_models.py
```

## Struktura projektu

- `dashboard.py` - Interfejs użytkownika zbudowany w Streamlit
- `dashboard_api.py` - API do dostarczania danych dla dashboardu
- `run.py` - Główny skrypt uruchomieniowy dla backendu
- `create_models.py` - Skrypt do generowania podstawowych modeli AI
- `test_models.py` - Skrypt do testowania modeli AI
- `ai_models/` - Katalog zawierający modele AI
- `python_libs/` - Biblioteki pomocnicze
- `models/` - Zapisane modele ML w formacie pkl
- `static/img/` - Obrazy, wykresy i inne zasoby statyczne

## Konfiguracja
System konfiguruje się przez plik `.env` - skopiuj `.env.example` i dostosuj ustawienia.

## Wymagania
System wymaga Pythona 3.8+ oraz zainstalowanych bibliotek z pliku `requirements.txt`.
```bash
pip install -r requirements.txt