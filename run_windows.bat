
@echo off
echo Starting Intelligent Trading System...
echo Creating logs directory if it doesn't exist...
if not exist logs mkdir logs

echo Activating Python environment if it exists...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. If you want to create one:
    echo python -m venv venv
    echo venv\Scripts\activate.bat
)

echo Starting application...
python main.py

pause
