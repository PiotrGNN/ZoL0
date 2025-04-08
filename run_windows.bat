
@echo off
echo === Inteligentny System Tradingowy - Uruchamianie na Windows ===
echo.

echo Tworzenie wymaganych katalogów...
mkdir logs 2>nul
mkdir data\cache 2>nul

echo Sprawdzanie i instalacja wymaganych pakietów...
pip install -r requirements.txt

echo Uruchamianie systemu tradingowego...
python main.py

echo.
pause
