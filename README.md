# ZoL0 - System Tradingowy z AI

## 📋 Opis
System tradingowy wspierany sztuczną inteligencją, zoptymalizowany do pracy z giełdą Bybit. System integruje strategie tradingowe, zarządzanie ryzykiem i modele AI w jednej platformie z interfejsem webowym.

## 🚀 Uruchomienie

### W środowisku Replit
1. Kliknij przycisk **Run** - aplikacja uruchomi się automatycznie
2. Poczekaj na załadowanie wszystkich zależności
3. Panel webowy będzie dostępny automatycznie w oknie przeglądarki

### Lokalnie
1. Zainstaluj zależności: `pip install -r requirements.txt`
2. Ustaw zmienne środowiskowe w pliku `.env` (skopiuj z `.env.example`)
3. Uruchom aplikację: `python main.py`
4. Przejdź do `http://localhost:5000` w przeglądarce

## 🔧 Konfiguracja
1. Aby połączyć się z giełdą Bybit, ustaw zmienne w pliku `.env`:
   ```
   BYBIT_API_KEY=your_api_key
   BYBIT_API_SECRET=your_api_secret
   TEST_MODE=true  # zmień na "false" w środowisku produkcyjnym
   ```

## 🧪 Testy
Uruchom testy jednostkowe: `python -m unittest discover -s data/tests`

## 📂 Struktura projektu
- `ai_models/` - Modele AI i uczenie maszynowe
- `data/` - Główne komponenty systemu
  - `execution/` - Wykonywanie zleceń
  - `indicators/` - Wskaźniki techniczne
  - `risk_management/` - Zarządzanie ryzykiem
  - `strategies/` - Strategie handlowe
  - `tests/` - Testy jednostkowe
- `static/` i `templates/` - Interfejs użytkownika
- `main.py` - Główny plik aplikacji


## 🔒 Bezpieczeństwo
- Przechowuj klucze API używając Replit Secrets Manager (`Tools → Secrets`)
- Ogranicz uprawnienia kluczy API do minimum
- Używaj trybu testowego (testnet) do testowania nowych strategii
- Ustaw odpowiednie limity ryzyka w konfiguracji

## 📜 Licencja
Ten projekt jest zastrzeżony i nie jest dostępny publicznie bez zgody autora.