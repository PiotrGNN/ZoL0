
#!/bin/bash

# 🔁 RESET ŚRODOWISKA PYTHON

echo "🧹 Czyszczę cache pakietów..."
rm -rf ~/.cache/pip

echo "📦 Instaluję podstawowe zależności..."
pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost matplotlib

echo "📦 Instaluję pozostałe zależności z requirements.txt..."
pip install -r requirements.txt

echo "🔍 Sprawdzam czy wszystkie modele AI są poprawnie ładowane..."
python test_models.py || echo "⚠️ Test modeli nie powiódł się, ale kontynuuję inicjalizację"

echo "🎉 Gotowe! Środowisko jest zaktualizowane i gotowe do pracy."
