
@echo off
echo Starting Trading Bot...

REM Konfiguracja środowiska
set FLASK_APP=main.py
set FLASK_ENV=development
set PORT=5000

REM Upewnij się, że wszystkie katalogi istnieją
if not exist "logs" mkdir logs
if not exist "data\cache" mkdir data\cache

REM Uruchomienie aplikacji
python main.py

pause
