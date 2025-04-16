
# Trading Dashboard System

System tradingowy z zaawansowanym dashboardem i integracją modeli AI.

## Funkcjonalności

- Dashboard z wizualizacją danych tradingowych
- Integracja z różnymi giełdami kryptowalut (Binance, Bybit)
- Modele AI do analizy rynku i predykcji
- Backtesting i symulacja strategii

## Instalacja

Zainstaluj wymagane zależności:

```bash
pip install -r requirements.txt
```

## Uruchamianie

### Dashboard

Aby uruchomić dashboard:

```bash
python run_dashboard.py
```

Dashboard będzie dostępny pod adresem http://0.0.0.0:8501

### System tradingowy

Aby uruchomić główny system tradingowy:

```bash
python run.py
```

### Tworzenie modeli

Aby wygenerować modele AI:

```bash
python create_models.py
```

## Struktura projektu

- `ai_models/` - Modele AI
- `data/` - Moduły przetwarzania danych
- `python_libs/` - Biblioteki pomocnicze
- `dashboard.py` - Interfejs użytkownika (Streamlit)
- `dashboard_api.py` - API dla dashboardu (Flask)
- `run.py` - Skrypt uruchamiający główny system
- `run_dashboard.py` - Skrypt uruchamiający dashboard

## Testowanie

Aby przetestować załadowane modele:

```bash
python -m python_libs.model_tester
```

## Rozwiązywanie problemów

Jeśli napotkasz problemy z ładowaniem modeli, upewnij się, że:

1. Katalog `models/` istnieje
2. Modele są w odpowiednim formacie (.pkl dla serializowanych modeli, .py dla klas)
3. Logi są czytelne w katalogu `logs/`
