
#!/bin/bash

# Skrypt do resetowania środowiska aplikacji

echo "🧹 Czyszczę cache pakietów..."
rm -rf __pycache__ */__pycache__ */*/__pycache__

echo "📦 Instaluję zależności z requirements.txt..."
python -m pip install --upgrade pip --user
python -m pip install -r requirements.txt --user

echo "🔍 Sprawdzam czy wszystkie modele AI są poprawnie ładowane..."
python test_models.py

echo "🧠 Przygotowuję katalogi do działania symulacji..."
mkdir -p logs
mkdir -p data/cache
mkdir -p reports
mkdir -p static/img
mkdir -p saved_models

# Zapisz wersję modeli do celów diagnostycznych
echo "📝 Zapisuję listę dostępnych modeli..."
python -c "
try:
    import os, importlib
    models_dir = 'ai_models'
    if os.path.exists(models_dir):
        files = [f[:-3] for f in os.listdir(models_dir) if f.endswith('.py') and not f.startswith('__')]
        print(f'Dostępne modele: {files}')
        
        for model_file in files:
            try:
                module = importlib.import_module(f'{models_dir}.{model_file}')
                print(f'✅ Moduł {model_file} załadowany pomyślnie')
            except Exception as e:
                print(f'❌ Błąd ładowania modułu {model_file}: {e}')
except Exception as e:
    print(f'Błąd: {e}')
" > logs/models_status.log

echo "🎉 Gotowe! Środowisko jest zaktualizowane i gotowe do pracy."
