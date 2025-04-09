
#!/bin/bash

echo "🧹 Czyszczę cache pakietów..."
rm -rf ~/.cache/pip

echo "📦 Instaluję zależności z requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🔍 Sprawdzam czy wszystkie modele AI są poprawnie ładowane..."
python test_models.py || echo "⚠️ Test modeli nie powiódł się, ale kontynuuję inicjalizację"

echo "🎉 Gotowe! Środowisko jest zaktualizowane i gotowe do pracy."
