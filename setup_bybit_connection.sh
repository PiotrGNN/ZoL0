#!/bin/bash
# setup_bybit_connection.sh
# Skrypt konfiguracyjny do ustawiania zmiennych środowiskowych dla połączenia z ByBit

# Kolory dla czytelniejszych komunikatów
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Wyświetl nagłówek
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}   Konfiguracja połączenia z ByBit     ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Domyślne wartości
USE_TESTNET=true
SYMBOL="BTCUSDT"
TIMEFRAME="15m"
DURATION=60
INITIAL_CAPITAL=10000

# Pytanie o tryb (testnet/produkcja)
echo -e "\n${YELLOW}Wybierz tryb połączenia:${NC}"
echo "1) Testnet (środowisko testowe bez prawdziwych pieniędzy)"
echo "2) Produkcja (UWAGA: używa prawdziwego konta i środków!)"
read -p "Wybierz opcję [1]: " mode_choice
mode_choice=${mode_choice:-1}

if [ "$mode_choice" -eq 2 ]; then
    USE_TESTNET=false
    echo -e "${RED}UWAGA! Wybrano tryb produkcyjny!${NC}"
    echo -e "${RED}W tym trybie będą składane PRAWDZIWE transakcje z użyciem PRAWDZIWYCH środków!${NC}"
    read -p "Czy na pewno chcesz kontynuować? (tak/nie) [nie]: " confirm
    confirm=${confirm:-nie}
    
    if [ "$confirm" != "tak" ]; then
        USE_TESTNET=true
        echo -e "${GREEN}Przywrócono bezpieczny tryb testnet.${NC}"
    else
        echo -e "${RED}Używanie trybu produkcyjnego! Zachowaj ostrożność!${NC}"
    fi
fi

# Pobranie klucza API
echo -e "\n${YELLOW}Podaj klucz API ByBit:${NC}"
read -p "API Key: " API_KEY

# Pobranie sekretu API
echo -e "\n${YELLOW}Podaj sekret API ByBit:${NC}"
read -p "API Secret: " API_SECRET

# Pobranie symbolu
echo -e "\n${YELLOW}Podaj symbol pary walutowej [${SYMBOL}]:${NC}"
read -p "Symbol: " symbol_input
SYMBOL=${symbol_input:-$SYMBOL}

# Pobranie timeframe
echo -e "\n${YELLOW}Podaj przedział czasowy (np. 1m, 5m, 15m, 1h) [${TIMEFRAME}]:${NC}"
read -p "Timeframe: " timeframe_input
TIMEFRAME=${timeframe_input:-$TIMEFRAME}

# Pobranie czasu trwania symulacji
echo -e "\n${YELLOW}Podaj czas trwania symulacji w minutach [${DURATION}]:${NC}"
read -p "Czas trwania (min): " duration_input
DURATION=${duration_input:-$DURATION}

# Pobranie początkowego kapitału
echo -e "\n${YELLOW}Podaj początkowy kapitał [${INITIAL_CAPITAL}]:${NC}"
read -p "Kapitał: " capital_input
INITIAL_CAPITAL=${capital_input:-$INITIAL_CAPITAL}

# Ustawienie zmiennych środowiskowych
export BYBIT_API_KEY=$API_KEY
export BYBIT_API_SECRET=$API_SECRET
export BYBIT_TESTNET=$USE_TESTNET
export TRADING_SYMBOL=$SYMBOL
export TRADING_TIMEFRAME=$TIMEFRAME
export TRADING_DURATION=$DURATION
export TRADING_CAPITAL=$INITIAL_CAPITAL

# Jeśli używamy trybu produkcyjnego, wymagane jest dodatkowe potwierdzenie
if [ "$USE_TESTNET" = false ]; then
    export BYBIT_PRODUCTION_CONFIRMED=true
    export BYBIT_PRODUCTION_ENABLED=true
    echo -e "\n${RED}Ustawiono zmienne dla trybu PRODUKCYJNEGO!${NC}"
else
    echo -e "\n${GREEN}Ustawiono zmienne dla trybu TESTNET.${NC}"
fi

# Wyświetl podsumowanie
echo -e "\n${BLUE}=======================================${NC}"
echo -e "${GREEN}Konfiguracja zakończona:${NC}"
echo -e "API Key: ${YELLOW}${API_KEY:0:5}*****${NC}"
echo -e "API Secret: ${YELLOW}*****${NC}"
echo -e "Tryb: ${YELLOW}$([ "$USE_TESTNET" = true ] && echo "Testnet" || echo "PRODUKCJA")${NC}"
echo -e "Symbol: ${YELLOW}${SYMBOL}${NC}"
echo -e "Timeframe: ${YELLOW}${TIMEFRAME}${NC}"
echo -e "Czas trwania: ${YELLOW}${DURATION} minut${NC}"
echo -e "Kapitał początkowy: ${YELLOW}${INITIAL_CAPITAL}${NC}"
echo -e "${BLUE}=======================================${NC}"

# Generowanie przykładowej komendy uruchomieniowej
echo -e "\n${GREEN}Aby uruchomić trading z tymi ustawieniami, wykonaj:${NC}"
echo -e "${BLUE}python run_live_trading.py --realtime $([ "$USE_TESTNET" = true ] && echo "--testnet") --symbol $SYMBOL --timeframe $TIMEFRAME --duration $DURATION --capital $INITIAL_CAPITAL --api-key \"$API_KEY\" --api-secret \"$API_SECRET\"${NC}"

# Zapisanie komendy do pliku dla łatwego uruchomienia
COMMAND="python run_live_trading.py --realtime $([ "$USE_TESTNET" = true ] && echo "--testnet") --symbol $SYMBOL --timeframe $TIMEFRAME --duration $DURATION --capital $INITIAL_CAPITAL --api-key \"$API_KEY\" --api-secret \"$API_SECRET\""
echo -e "#!/bin/bash\n$COMMAND" > run_live_bybit.sh
chmod +x run_live_bybit.sh

echo -e "\n${GREEN}Zapisano komendę do pliku run_live_bybit.sh${NC}"