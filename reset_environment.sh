
#!/bin/bash

# Kolory dla lepszej czytelności
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 Czyszczę cache pakietów...${NC}"
rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__

echo -e "${BLUE}📦 Instaluję zależności z requirements.txt...${NC}"
pip install -r requirements.txt

# Sprawdź czy python_libs zawiera wszystkie potrzebne pliki
echo -e "${BLUE}🔍 Sprawdzam czy wszystkie wymagane moduły są zainstalowane...${NC}"
python -c "import os; required_files = ['simplified_trading_engine.py', 'simplified_risk_manager.py', 'simplified_strategy.py', 'model_tester.py', 'simulated_bybit.py', 'simulation_results.py']; missing = [f for f in required_files if not os.path.exists(os.path.join('python_libs', f))]; print('Brakujące pliki:', missing) if missing else print('Wszystkie wymagane pliki są obecne')"

# Test czy AI modele można załadować
echo -e "${BLUE}🔍 Sprawdzam czy wszystkie modele AI są poprawnie ładowane...${NC}"
python -c "import sys; sys.path.insert(0, '.'); import ai_models"

# Tworzenie struktury katalogów
echo -e "${BLUE}📁 Tworzę strukturę katalogów...${NC}"
mkdir -p logs data/cache reports static/img

# Czyszczenie logów
echo -e "${BLUE}🧹 Czyszczę stare logi...${NC}"
rm -f logs/*.log

echo -e "${GREEN}🎉 Gotowe! Środowisko jest zaktualizowane i gotowe do pracy.${NC}"
