
@echo off
echo === Uruchamianie aplikacji tradingowej ===
echo Upewnij się, że masz aktywowane środowisko wirtualne

echo Tworzenie katalogów, jeśli nie istnieją...
if not exist "logs" mkdir logs
if not exist "data\cache" mkdir data\cache

echo Sprawdzanie czy jest plik .env...
if not exist ".env" (
    echo Tworzenie pliku .env na podstawie .env.example...
    copy .env.example .env
)

echo Uruchamianie aplikacji...
python main.py

pause
