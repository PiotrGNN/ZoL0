
@echo off
echo Testing Bybit API Connection...

REM Upewnij się, że wszystkie katalogi istnieją
if not exist "logs" mkdir logs

REM Uruchomienie testu połączenia
python test_bybit_connection.py

pause
