
@echo off
echo === Uruchamianie aplikacji tradingowej (Wersja lokalna) ===

echo Tworzenie katalogów, jeśli nie istnieją...
if not exist "logs" mkdir logs
if not exist "data\cache" mkdir data\cache
if not exist "python_libs" mkdir python_libs

echo Sprawdzanie czy jest plik .env...
if not exist ".env" (
    echo Tworzenie pliku .env na podstawie .env.example...
    copy .env.example .env
    echo WAŻNE: Edytuj plik .env, aby dodać swoje klucze API!
    pause
)

echo Instalacja wymaganych pakietów...
pip install -r requirements.txt

echo Uruchamianie aplikacji...
python main.py

pause
