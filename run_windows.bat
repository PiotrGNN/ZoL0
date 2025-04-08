
@echo off
echo Starting Inteligentny System Tradingowy...
echo.
echo Checking directories...
if not exist "logs" mkdir logs
if not exist "data\cache" mkdir data\cache
if not exist "reports" mkdir reports
if not exist "saved_models" mkdir saved_models

echo.
echo Verifying environment...
if not exist ".env" (
    echo Creating .env from .env.example
    copy .env.example .env
    echo Please edit .env with your API keys before continuing!
    pause
)

echo.
echo Starting application...
python main.py
pause
