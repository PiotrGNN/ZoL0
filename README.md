
# CryptoTrader Dashboard

Dashboard do tradingu kryptowalut z backendem w FastAPI i frontendem w React/TailwindCSS.

## Funkcjonalności

- 📊 Wyświetlanie danych z giełdy (saldo, ceny, transakcje)
- 📈 Real-time aktualizacje przez WebSocket
- 💰 Panel tradingowy do składania zleceń
- 📉 Wykresy i statystyki wydajności portfela
- 🤖 Integracja z modelami AI (predykcje)
- 🔄 Automatyczne strategie tradingowe

## Wymagania

- Python 3.8+
- Node.js 16+ (opcjonalnie dla rozwoju frontendu)

## Instalacja

1. Sklonuj to repozytorium
2. Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

3. Skonfiguruj zmienne środowiskowe:

Skopiuj plik .env.example do .env i wypełnij wymagane dane:

```
BYBIT_API_KEY=YourApiKeyHere
BYBIT_API_SECRET=YourApiSecretHere
BYBIT_TESTNET=true
```

## Uruchomienie

### Uruchomienie serwera FastAPI:

```bash
python app.py
```

Serwer zostanie uruchomiony na porcie 5000. Dashboard będzie dostępny pod adresem http://localhost:5000/static/dashboard.html

Możesz uruchomić skrypt główny, który także zainicjalizuje środowisko:

```bash
python run.py
```

## Struktura projektu

```
├── app.py                # Główny plik aplikacji FastAPI
├── run.py                # Skrypt do lokalnego uruchomienia systemu
├── requirements.txt      # Zależności
├── .env                  # Konfiguracja środowiska
├── static/               # Pliki statyczne
│   ├── css/              # Style CSS
│   ├── js/               # Skrypty JavaScript
│   └── img/              # Obrazy
├── templates/            # Szablony HTML
├── data/                 # Moduły do obsługi danych
│   ├── execution/        # Klasy do komunikacji z giełdami
│   ├── strategies/       # Strategie tradingowe
│   └── utils/            # Narzędzia pomocnicze
├── ai_models/            # Modele AI
└── logs/                 # Logi aplikacji
```

## API Endpoints

- `GET /api/health` - Sprawdzenie stanu aplikacji
- `GET /api/balance` - Pobranie salda portfela
- `GET /api/ticker/{symbol}` - Pobranie tickera dla symbolu
- `GET /api/klines/{symbol}` - Pobranie danych świecowych
- `POST /api/order` - Złożenie zlecenia
- `GET /api/market_summary` - Podsumowanie rynku
- `GET /api/portfolio_performance` - Wydajność portfela
- `GET /api/strategies` - Lista dostępnych strategii
- `WebSocket /ws` - Endpoint do aktualizacji w czasie rzeczywistym

## Rozwijanie projektu

### Backend (FastAPI)

1. Dodawanie nowych endpointów: Otwórz `app.py` i dodaj nowe funkcje obsługi żądań.
2. Modyfikacja strategii: Edytuj pliki w katalogu `data/strategies/`.
3. Zmiana konfiguracji: Dostosuj ustawienia w pliku `.env`.

### Frontend

1. Modyfikacja interfejsu: Edytuj pliki w `templates/` i `static/`.
2. Dodawanie funkcjonalności JavaScript: Edytuj plik `static/js/trading-dashboard.js`.

## Bezpieczeństwo

- Upewnij się, że API_KEY i API_SECRET są bezpieczne.
- Domyślnie używane jest testnet API, aby uniknąć przypadkowego użycia prawdziwych środków.
- Jeśli chcesz używać prawdziwego API, zmień `BYBIT_TESTNET=false` w pliku `.env` i potwierdź to dodatkowymi ustawieniami.

## Licencja

MIT
