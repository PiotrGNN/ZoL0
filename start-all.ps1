# Skrypt PowerShell do uruchamiania całego stacku lokalnie
# 1. Backend Python (main.py)
# 2. Express (npm run start)
# 3. Usługi infrastrukturalne przez docker compose

# Uruchom backend Python w nowym oknie
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd "${PWD}"; python main.py'

# Uruchom Express (TypeScript) w nowym oknie
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd "${PWD}"; npm install; npm run start'

# Uruchom usługi infrastrukturalne (Redis, Kafka, Prometheus, Grafana, Loki) w nowym oknie
Start-Process powershell -ArgumentList '-NoExit', '-Command', 'cd "${PWD}"; docker compose up redis kafka prometheus grafana loki'

Write-Host "\nWszystkie serwisy uruchomione w osobnych oknach PowerShell.\nAby zakończyć, zamknij odpowiednie okna lub zatrzymaj procesy.\n" -ForegroundColor Green
