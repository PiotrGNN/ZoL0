# Trading Platform

## 📌 Opis

Kompleksowa platforma do automatycznego tradingu, wykorzystująca zaawansowane algorytmy i sztuczną inteligencję do analizy rynków kryptowalut.

## 🚀 Funkcje

- Podłączenie do Bybit API z obsługą kont testowych i produkcyjnych
- Moduły analizy technicznej i sentymentu
- Wykrywanie anomalii rynkowych
- Strategie oparte na uczeniu maszynowym
- Zarządzanie ryzykiem i kapitałem
- Interfejs webowy do monitorowania i kontroli

## 🛠️ Instalacja

### Wymagania

- Python 3.10+
- Windows 11/10, Linux lub macOS

### Kroki instalacji

1. Sklonuj repozytorium:
```
git clone https://github.com/username/trading-platform.git
cd trading-platform
```

2. Utwórz i aktywuj wirtualne środowisko (opcjonalnie, ale zalecane):
```
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Zainstaluj zależności:
```
pip install -r requirements.txt
```

4. Skonfiguruj ustawienia:
- Skopiuj `.env.example` do `.env`
- Edytuj plik `.env` i wprowadź swoje klucze API

```
cp .env.example .env
```

## 🚀 Uruchomienie aplikacji

### Windows

```
run_windows.bat
```

lub

```
python main.py
```

### Linux/macOS

```
python main.py
```

## 🧪 Testowanie

### Test połączenia z Bybit API

```
python test_bybit_connection.py
```

## 📚 Dokumentacja

Szczegółowa dokumentacja API i modułów dostępna jest w katalogu `docs/`.  Dodatkowo: [Bybit API V5](https://bybit-exchange.github.io/docs/v5/intro)


## ⚠️ Uwagi

- Domyślnie system używa trybu TESTNET - aby używać prawdziwego tradingu, zmień `BYBIT_USE_TESTNET=false` w pliku `.env`
- Zachowaj ostrożność przy korzystaniu z produkcyjnego API - wszystkie transakcje są rzeczywiste i mogą prowadzić do strat finansowych

## 🔧 Rozwiązywanie problemów
### Brak modułu
Jeśli pojawi się błąd o brakującym module, zainstaluj go ręcznie:
```cmd
pip install nazwa_modułu
```

### Problemy z połączeniem API
1. Sprawdź poprawność kluczy API w pliku `.env`
2. Upewnij się, że masz działające połączenie z internetem
3. Sprawdź czy Twoje klucze API mają odpowiednie uprawnienia na Bybit

### Inne problemy
1. Sprawdź logi w katalogu `logs/`
2. Uruchom aplikację w trybie debug: `python main.py --debug`


## 📝 Licencja

To oprogramowanie jest dostępne na licencji MIT.