
#!/bin/bash

echo "🧹 Czyszczę cache pakietów..."
rm -rf ~/.cache/pip

echo "📦 Instaluję zależności z requirements.txt..."
# Użyj bezpośrednio instancji Pythona z Replit
pip3 install --upgrade pip || python3 -m pip install --upgrade pip
pip3 install -r requirements.txt || python3 -m pip install -r requirements.txt

echo "🔍 Sprawdzam czy wszystkie modele AI są poprawnie ładowane..."
python3 test_models.py || echo "⚠️ Test modeli nie powiódł się, ale kontynuuję inicjalizację"

echo "🎉 Gotowe! Środowisko jest zaktualizowane i gotowe do pracy."
