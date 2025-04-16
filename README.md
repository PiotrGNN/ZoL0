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

### Główna aplikacja (API)
```bash
python run.py
```

### Dashboard (Streamlit)
```bash
streamlit run dashboard.py
```

### Tworzenie modeli AI
```bash
python create_models.py
```

## Struktura projektu
- `ai_models/` - Modele AI i ich implementacje
- `data/` - Moduły dostępu do danych rynkowych
- `python_libs/` - Podstawowe biblioteki systemu
- `static/` - Pliki statyczne dla interfejsu
- `templates/` - Szablony HTML
- `models/` - Zapisane modele AI
- `reports/` - Raporty z backtestingu

## Konfiguracja
System konfiguruje się przez plik `.env` - skopiuj `.env.example` i dostosuj ustawienia.

## Wymagania
System wymaga Pythona 3.8+ oraz zainstalowanych bibliotek z pliku `requirements.txt`.
```bash
pip install -r requirements.txt