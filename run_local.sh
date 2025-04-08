
#!/bin/bash
echo "=== Uruchamianie aplikacji tradingowej (Wersja lokalna) ==="

echo "Tworzenie katalogów, jeśli nie istnieją..."
mkdir -p logs
mkdir -p data/cache
mkdir -p python_libs

echo "Sprawdzanie czy jest plik .env..."
if [ ! -f ".env" ]; then
    echo "Tworzenie pliku .env na podstawie .env.example..."
    cp .env.example .env
    echo "WAŻNE: Edytuj plik .env, aby dodać swoje klucze API!"
    read -p "Naciśnij Enter, aby kontynuować..."
fi

echo "Instalacja wymaganych pakietów..."
pip install -r requirements.txt

echo "Uruchamianie aplikacji..."
python main.py
