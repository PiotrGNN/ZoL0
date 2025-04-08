# Inteligentny System Tradingowy

## Opis Projektu
Kompleksowy system tradingowy integrujący sztuczną inteligencję, analizę techniczną, zarządzanie ryzykiem oraz automatyzację handlu na giełdzie kryptowalut Bybit.

## Instalacja

```bash
git clone https://github.com/twoj-username/trading-system.git
cd trading-system
pip install -r requirements.txt
python init_project.py
```

## Konfiguracja
1. Skopiuj plik `.env.example` do `.env`
2. Uzupełnij zmienne środowiskowe w pliku `.env`, w tym klucze API Bybit
3. Dostosuj parametry systemu w pliku `config/settings.yml`

## Uruchomienie

```bash
python main.py
```

Aplikacja będzie dostępna pod adresem http://0.0.0.0:5000

## Struktura Projektu

- `ai_models/` - Modele AI i uczenia maszynowego
- `data/` - Wszystkie moduły przetwarzania danych
  - `execution/` - Wykonywanie zleceń i komunikacja z giełdą
  - `indicators/` - Wskaźniki techniczne i analiza sentymentu
  - `strategies/` - Strategie handlowe
  - `risk_management/` - Zarządzanie ryzykiem
- `static/` - Pliki statyczne (CSS, JavaScript)
- `templates/` - Szablony HTML
- `logs/` - Pliki logów
- `config/` - Pliki konfiguracyjne

## Główne Funkcje
- Dashboard z podglądem stanu systemu i portfela
- Automatyczne strategie handlowe z możliwością dynamicznego przełączania
- Analiza sentymentu rynkowego
- Wykrywanie anomalii rynkowych
- Zarządzanie ryzykiem portfela
- Optymalizacja strategii na podstawie danych historycznych

## Status Komponentów
- Trading Engine: Automatyczny system handlowy
- API Connector: Połączenie z API giełdy
- Data Processor: Przetwarzanie danych rynkowych
- Risk Manager: Zarządzanie ryzykiem portfela

## Rozwijanie Projektu
Aby dodać nową strategię, utwórz nowy plik w folderze `data/strategies/` i zaimplementuj wymagane metody.

## 📚 Dokumentacja
Szczegółowa dokumentacja API i modułów dostępna jest w katalogu `docs/`.  Dodatkowo: [Bybit API V5](https://bybit-exchange.github.io/docs/v5/intro)