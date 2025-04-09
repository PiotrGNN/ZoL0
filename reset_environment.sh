
#!/bin/bash
# 🔁 RESET ŚRODOWISKA PYTHON W REPLICIE

echo "🧹 Czyszczę cache pakietów..."
rm -rf ~/.cache/pip

echo "📦 Instaluję zależności z requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🔍 Sprawdzam czy wszystkie modele AI są poprawnie ładowane..."
python3 test_models.py

echo "🎉 Gotowe! Środowisko jest zaktualizowane i gotowe do pracy."
