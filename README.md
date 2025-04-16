# System Tradingowy AI

## 🚀 Szybkie uruchomienie

1. Kliknij przycisk `Run` w górnej części ekranu Replit.
2. Poczekaj na inicjalizację środowiska i uruchomienie aplikacji.
3. Aplikacja zostanie uruchomiona i będzie dostępna pod wskazanym adresem.

## 📂 Struktura projektu

```
project/
├── ai_models/            # Modele sztucznej inteligencji
├── python_libs/          # Biblioteki pomocnicze
├── data/                 # Dane i cache
│   └── cache/            # Pliki tymczasowe
├── models/               # Zapisane modele
├── saved_models/         # Dodatkowe zapisane modele
├── logs/                 # Pliki logów
├── reports/              # Generowane raporty
├── static/               # Pliki statyczne
├── main.py               # Główny plik aplikacji
├── run.py                # Skrypt startowy
├── create_models.py      # Skrypt tworzący przykładowe modele
└── test_models.py        # Skrypt testujący modele
```

## 🤖 Zarządzanie modelami AI

1. **Tworzenie przykładowych modeli**:
   ```
   python create_models.py
   ```

2. **Testowanie modeli**:
   ```
   python test_models.py
   ```

3. **Uruchomienie aplikacji**:
   ```
   python run.py
   ```

## 🛠️ Rozwiązywanie problemów

Jeśli wystąpią problemy:

1. Sprawdź logi w katalogu `logs/`.
2. Usuń pliki blokad, jeśli aplikacja została nieprawidłowo zamknięta:
   ```
   rm running.lock models_tested.lock
   ```
3. Uruchom tworzenie modeli przykładowych:
   ```
   python create_models.py
   ```

## 📋 Lista funkcjonalności

- **Automatyczne wykrywanie modeli** - system automatycznie znajduje i ładuje modele AI.
- **Testowanie modeli** - testy sprawdzają poprawność modeli przed użyciem.
- **Zarządzanie danymi** - system obsługuje dane z różnych źródeł.
- **Wizualizacja wyników** - dashboard z wykresami i statystykami.
- **Symulacja tradingu** - możliwość testowania strategii bez ryzyka.

## 🔧 Konfiguracja

Konfiguracja znajduje się w pliku `.env` (utworzonym z `.env.example` przy pierwszym uruchomieniu).

## Zarządzanie modelami AI
System zawiera narzędzie do zarządzania modelami AI, które można uruchomić:
```
python manage_ai_models.py [KOMENDA]
```

Dostępne komendy:
- `list` - Wyświetl listę dostępnych modeli
- `backup` - Utwórz kopię zapasową modeli
- `test` - Testuj wszystkie modele
- `retrain [MODEL]` - Trenuj wybrane modele (użyj "all" dla wszystkich)
- `details [MODEL]` - Wyświetl szczegóły modelu


Przykłady:
```bash
# Wyświetl listę modeli
python manage_ai_models.py list

# Testuj wszystkie modele
python manage_ai_models.py test

# Trenuj wybrany model
python manage_ai_models.py retrain AnomalyDetector
```

Możesz uruchomić testy wszystkich modeli za pomocą:
```
python test_models.py
```

Opcjonalnie, możesz wymusić ponowne trenowanie podczas testów:
```
python test_models.py --force-retrain