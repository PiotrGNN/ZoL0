# ZoL0 Trading System

## Opis projektu
System tradingowy wykorzystujący modele AI do analizy rynków kryptowalut i automatycznego handlu.

## Funkcje
- Analiza sentymentu rynkowego
- Wykrywanie anomalii cenowych
- Automatyczne strategie handlowe
- Integracja z giełdą Bybit (testnet i produkcja)
- Dashboard z wizualizacją danych

## Instalacja

### Wymagania
System wymaga Python 3.8+ i zainstalowanych bibliotek z pliku requirements.txt.

```bash
pip install -r requirements.txt
```

### Konfiguracja
1. Skopiuj plik `.env.example` do `.env`
2. Ustaw swoje klucze API Bybit:
   - `BYBIT_API_KEY` - Twój klucz API
   - `BYBIT_API_SECRET` - Twój sekret API
   - `BYBIT_TESTNET` - `true` dla testnet, `false` dla produkcji

## Uruchomienie

### Tryb lokalny
```bash
python run.py
```

### Tworzenie modeli AI
```bash
python create_models.py
```

## Struktura projektu
- `ai_models/` - Modele AI i narzędzia do ich obsługi
- `data/` - Moduły do pobierania i przetwarzania danych
- `python_libs/` - Uproszczone wersje bibliotek
- `static/` - Zasoby statyczne dla dashboardu
- `templates/` - Szablony HTML dla dashboardu
- `logs/` - Logi systemu
- `models/` - Zapisane modele AI
- `reports/` - Raporty wyników tradingu

## Dashboard
Dashboard jest dostępny pod adresem http://localhost:5000 po uruchomieniu systemu.

## Tryb symulacji vs. produkcja
System domyślnie działa w trybie symulacji (testnet Bybit).
Aby uruchomić w trybie produkcyjnym:

```bash
python run.py --mode real
```

**UWAGA:** Tryb produkcyjny wykonuje rzeczywiste transakcje z użyciem prawdziwych środków!