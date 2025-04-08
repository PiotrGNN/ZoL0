// Konfiguracja aplikacji
const CONFIG = {
    updateInterval: 15000, // Interwał odświeżania dashboardu (ms)
    chartUpdateInterval: 60000, // Interwał odświeżania wykresów (ms)
    statusUpdateInterval: 30000, // Interwał odświeżania statusów (ms)
    maxErrors: 3, // maksymalna liczba błędów przed wyświetleniem komunikatu
    errorRetryDelay: 5000, // ms - opóźnienie przed ponowną próbą po błędzie
    errorBackoff: 1.5, // mnożnik opóźnienia przy kolejnych błędach
    apiEndpoints: {
        dashboard: '/api/dashboard/data',
        portfolio: '/api/portfolio',
        trades: '/api/recent-trades',
        alerts: '/api/alerts',
        tradingStats: '/api/trading-stats',
        components: '/api/component-status',
        aiModels: '/api/ai-models-status',
        system: '/api/system/status',
        bybitServerTime: 'https://api.bybit.com/v2/public/time', // Updated to production endpoint
        bybitBalance: 'https://api.bybit.com/v2/private/wallet/balance', // Updated to production endpoint
        bybitConnectionTest: 'https://api.bybit.com/v2/public/time', // Updated to production endpoint
        notifications: '/api/notifications',
        chartData: '/api/chart-data'
    }
};

// Stan aplikacji
const appState = {
    activeDashboard: true,
    errorCounts: {},
    retryDelays: {},
    portfolioData: null,
    dashboardData: null,
    lastUpdated: {
        portfolio: 0,
        dashboard: 0,
        chart: 0,
        components: 0
    },
    portfolioChart: null
};

// Inicjalizacja po załadowaniu dokumentu
document.addEventListener('DOMContentLoaded', function() {
    console.log("Dashboard załadowany");
    setupEventListeners();
    initializeUI();
    startDataUpdates();
});

// Inicjalizacja interfejsu użytkownika
function initializeUI() {
    // Inicjalizacja wykresu portfela
    initializePortfolioChart();

    // Pobierz początkowe dane
    updateDashboardData();
    updateComponentStatus();
    updatePortfolioData();
}

// Rozpoczęcie automatycznych aktualizacji danych
function startDataUpdates() {
    // Regularne aktualizacje danych dashboardu
    setInterval(function() {
        if (appState.activeDashboard) {
            updateDashboardData();
            updateComponentStatus();
        }
    }, CONFIG.updateInterval);

    // Regularne aktualizacje wykresu (rzadziej)
    setInterval(function() {
        if (appState.activeDashboard) {
            updateChartData();
        }
    }, CONFIG.chartUpdateInterval);
}

// Inicjalizacja wykresu portfela
function initializePortfolioChart() {
    const ctx = document.getElementById('portfolio-chart');
    if (!ctx) return;

    appState.portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Wartość Portfela',
                data: [],
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                borderWidth: 2,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// Aktualizacja danych wykresu
function updateChartData() {
    if (!appState.portfolioChart) return;

    fetch(CONFIG.apiEndpoints.chartData)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                appState.portfolioChart.data.labels = data.data.labels;
                appState.portfolioChart.data.datasets[0].data = data.data.datasets[0].data;
                appState.portfolioChart.update();
                appState.lastUpdated.chart = Date.now();
            } else {
                console.error("Błąd podczas pobierania danych wykresu:", data.error);
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania danych wykresu:", error);
            handleApiError('chartData');
        });
}

// Aktualizacja danych portfela
function updatePortfolioData() {
    fetch(CONFIG.apiEndpoints.portfolio)
        .then(response => response.json())
        .then(data => {
            appState.portfolioData = data;
            appState.lastUpdated.portfolio = Date.now();
            updatePortfolioUI(data);
        })
        .catch(error => {
            console.error("Błąd podczas pobierania danych portfela:", error);
            // Wyświetl przyjazny komunikat błędu w interfejsie
            document.getElementById('portfolio-container').innerHTML = `
                <div class="alert alert-warning">
                    Nie udało się pobrać danych portfela. Sprawdź połączenie z API.
                    <button class="btn btn-sm btn-outline-primary ml-2" onclick="updatePortfolioData()">Spróbuj ponownie</button>
                </div>
            `;
            handleApiError('portfolio');
        });
}

// Aktualizacja UI portfela
function updatePortfolioUI(data) {
    try {
        if (!data || !data.balances) {
            return;
        }

        const portfolioContainer = document.getElementById('portfolio-balance');
        if (!portfolioContainer) return;

        // Wyczyść kontener
        portfolioContainer.innerHTML = '';

        // Dla każdej waluty w portfelu
        for (const [currency, balance] of Object.entries(data.balances)) {
            if (balance.equity > 0) {
                const balanceItem = document.createElement('div');
                balanceItem.className = 'balance-item';

                balanceItem.innerHTML = `
                    <span class="currency">${currency}</span>
                    <span class="amount">${parseFloat(balance.equity).toFixed(currency === 'BTC' ? 8 : 2)}</span>
                `;

                portfolioContainer.appendChild(balanceItem);
            }
        }

        // Aktualizuj status połączenia
        const connectionStatus = document.getElementById('api-status');
        if (connectionStatus) {
            if (data.success) {
                connectionStatus.textContent = 'Connected';
                connectionStatus.className = 'status-badge online';
            } else {
                connectionStatus.textContent = 'Error';
                connectionStatus.className = 'status-badge offline';
            }
        }

        // Aktualizuj czas ostatniej aktualizacji
        updateLastRefreshed();
    } catch (error) {
        console.error("Błąd podczas aktualizacji UI portfela:", error);
    }
}

// Aktualizacja głównych danych dashboardu
function updateDashboardData() {
    console.log("Aktualizacja danych dashboardu...");
    fetch(CONFIG.apiEndpoints.dashboard)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                appState.dashboardData = data;
                appState.lastUpdated.dashboard = Date.now();
                updateDashboardUI(data);
            } else {
                console.error("Błąd podczas pobierania danych dashboardu:", data.error);
                handleApiError('dashboard');
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania danych dashboardu:", error);
            // Wyświetl przyjazny komunikat błędu w interfejsie
            document.getElementById('dashboard-container').innerHTML = `
                <div class="alert alert-warning">
                    Nie udało się pobrać danych dashboardu. Sprawdź połączenie z API.
                    <button class="btn btn-sm btn-outline-primary ml-2" onclick="updateDashboardData()">Spróbuj ponownie</button>
                </div>
            `;
            handleApiError('dashboard');
        });

    // Pobierz również dane portfela podczas aktualizacji dashboardu
    updatePortfolioData();

    // Aktualizuj czas ostatniej aktualizacji
    updateLastRefreshed();
}

// Aktualizacja UI dashboardu
function updateDashboardUI(data) {
    try {
        // Aktualizacja podstawowych statystyk
        updateElementValue('current-balance', data.balance, true, '$');
        updateElementValue('profit-loss', data.profit_loss, true, '$');
        updateElementValue('open-positions', data.open_positions);
        updateElementValue('total-trades', data.total_trades);
        updateElementValue('win-rate', data.win_rate, true, '%');
        updateElementValue('max-drawdown', data.max_drawdown, true, '%');
        updateElementValue('market-sentiment', data.market_sentiment);

        // Obsługa anomalii
        updateAnomalies(data.anomalies);

        // Aktualizuj czas ostatniej aktualizacji
        const lastUpdatedElement = document.getElementById('last-updated');
        if (lastUpdatedElement) {
            lastUpdatedElement.textContent = data.last_updated || new Date().toLocaleString();
        }
    } catch (error) {
        console.error("Błąd podczas aktualizacji UI dashboardu:", error);
    }
}

// Aktualizacja wykrytych anomalii
function updateAnomalies(anomalies) {
    const anomalyContainer = document.getElementById('anomalies-list');
    if (!anomalyContainer) return;

    // Wyczyść kontener
    anomalyContainer.innerHTML = '';

    if (!anomalies || anomalies.length === 0) {
        anomalyContainer.innerHTML = '<div class="no-data">Brak wykrytych anomalii</div>';
        return;
    }

    // Dla każdej anomalii
    anomalies.forEach(anomaly => {
        const anomalyItem = document.createElement('div');
        anomalyItem.className = 'anomaly-item';

        anomalyItem.innerHTML = `
            <div class="anomaly-symbol">${anomaly.symbol}</div>
            <div class="anomaly-type">${anomaly.type}</div>
            <div class="anomaly-time">${anomaly.time}</div>
            <div class="anomaly-severity ${getSeverityClass(anomaly.severity)}">${anomaly.severity}</div>
        `;

        anomalyContainer.appendChild(anomalyItem);
    });
}

// Funkcja pomocnicza do aktualizacji wartości elementu
function updateElementValue(elementId, value, isNumeric = false, prefix = '', suffix = '') {
    const element = document.getElementById(elementId);
    if (!element) return;

    if (value === undefined || value === null) {
        element.textContent = 'N/A';
        return;
    }

    if (isNumeric) {
        const numValue = parseFloat(value);
        const formattedValue = isNaN(numValue) ? value : numValue.toFixed(2);
        element.textContent = `${prefix}${formattedValue}${suffix}`;

        // Dodaj klasę dla wartości dodatnich/ujemnych
        if (numValue > 0) {
            element.classList.add('positive');
            element.classList.remove('negative');
        } else if (numValue < 0) {
            element.classList.add('negative');
            element.classList.remove('positive');
        } else {
            element.classList.remove('positive', 'negative');
        }
    } else {
        element.textContent = `${prefix}${value}${suffix}`;
    }
}

// Aktualizacja statusu komponentów
function updateComponentStatus() {
    console.log("Aktualizacja statusów komponentów...");
    fetch(CONFIG.apiEndpoints.components)
        .then(response => response.json())
        .then(data => {
            appState.lastUpdated.components = Date.now();

            if (data.components) {
                data.components.forEach(component => {
                    const statusElement = document.getElementById(`${component.id}-status`);
                    if (statusElement) {
                        statusElement.className = `status-badge ${component.status}`;
                        statusElement.textContent = capitalizeFirstLetter(component.status);
                    }
                });
            }

            // Aktualizacja czasu ostatniej aktualizacji
            updateLastRefreshed();
        })
        .catch(error => {
            console.error("Błąd podczas aktualizacji statusu komponentów:", error);
            handleApiError('components');
        });
}

// Aktualizacja czasu ostatniego odświeżenia
function updateLastRefreshed() {
    const refreshElement = document.getElementById('last-refreshed');
    if (refreshElement) {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        refreshElement.textContent = `Ostatnia aktualizacja: ${timeString}`;
    }
}

// Funkcje pomocnicze
function capitalizeFirstLetter(string) {
    if (!string) return '';
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function getSeverityClass(severity) {
    switch(severity.toLowerCase()) {
        case 'high':
            return 'high-severity';
        case 'medium':
            return 'medium-severity';
        case 'low':
            return 'low-severity';
        default:
            return '';
    }
}

// Funkcje zarządzania stanem tradingu
function startTrading() {
    fetch('/api/trading/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('success', data.message || 'Trading automatyczny uruchomiony');
            updateComponentStatus();
        } else {
            showNotification('error', data.error || 'Nie udało się uruchomić tradingu');
        }
    })
    .catch(error => {
        console.error('Błąd podczas uruchamiania tradingu:', error);
        showNotification('error', 'Nie udało się uruchomić tradingu');
    });
}

function stopTrading() {
    fetch('/api/trading/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('success', data.message || 'Trading automatyczny zatrzymany');
            updateComponentStatus();
        } else {
            showNotification('error', data.error || 'Nie udało się zatrzymać tradingu');
        }
    })
    .catch(error => {
        console.error('Błąd podczas zatrzymywania tradingu:', error);
        showNotification('error', 'Nie udało się zatrzymać tradingu');
    });
}

function resetSystem() {
    fetch('/api/system/reset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('success', data.message || 'System zresetowany');
            updateDashboardData();
            updateComponentStatus();
        } else {
            showNotification('error', data.error || 'Nie udało się zresetować systemu');
        }
    })
    .catch(error => {
        console.error('Błąd podczas resetowania systemu:', error);
        showNotification('error', 'Nie udało się zresetować systemu');
    });
}

// Setup Event Listeners
function setupEventListeners() {
    // Trading controls
    const startTradingBtn = document.getElementById('start-trading-btn');
    if (startTradingBtn) {
        startTradingBtn.addEventListener('click', startTrading);
    }

    const stopTradingBtn = document.getElementById('stop-trading-btn');
    if (stopTradingBtn) {
        stopTradingBtn.addEventListener('click', stopTrading);
    }

    const resetSystemBtn = document.getElementById('reset-system-btn');
    if (resetSystemBtn) {
        resetSystemBtn.addEventListener('click', resetSystem);
    }

    // Visibility change (to pause updates when tab is not visible)
    document.addEventListener('visibilitychange', function() {
        appState.activeDashboard = !document.hidden;
    });
}

// Obsługa błędów API
function handleApiError(endpoint) {
    // Zwiększ licznik błędów dla danego endpointu
    appState.errorCounts[endpoint] = (appState.errorCounts[endpoint] || 0) + 1;

    // Jeśli przekroczono limit błędów, pokaż komunikat
    if (appState.errorCounts[endpoint] >= CONFIG.maxErrors) {
        showErrorMessage(`Zbyt wiele błędów podczas komunikacji z API (${endpoint}). Sprawdź logi.`);
    }
}

// Wyświetlanie komunikatu o błędzie
function showErrorMessage(message) {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.textContent = message;
        errorContainer.style.display = 'block';

        // Automatyczne ukrycie po 10 sekundach
        setTimeout(() => {
            errorContainer.style.display = 'none';
        }, 10000);
    }
}

// Wyświetlanie powiadomień
function showNotification(type, message) {
    // Stwórz element powiadomienia
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    // Dodaj do kontenera powiadomień
    const container = document.getElementById('notifications-container');
    if (container) {
        container.appendChild(notification);

        // Usuń po 5 sekundach
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => {
                container.removeChild(notification);
            }, 500);
        }, 5000);
    }
}