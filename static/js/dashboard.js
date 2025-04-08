// Konfiguracja aplikacji zostanie zachowana z pierwszej deklaracji

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
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            const portfolioContainer = document.getElementById('portfolio-data');

            if (data && data.success === true && data.balances && Object.keys(data.balances).length > 0) {
                portfolioContainer.innerHTML = ''; // Wyczyść kontener

                // Przetwarzanie danych portfela
                for (const [currency, details] of Object.entries(data.balances)) {
                    if (details) {
                        const balanceItem = document.createElement('div');
                        balanceItem.className = 'balance-item';

                        // Bezpieczne wyświetlanie wartości z obsługą wartości null/undefined
                        const equity = typeof details.equity === 'number' ? details.equity.toFixed(4) : '0.0000';
                        const available = typeof details.available_balance === 'number' ? details.available_balance.toFixed(4) : '0.0000';
                        const wallet = typeof details.wallet_balance === 'number' ? details.wallet_balance.toFixed(4) : '0.0000';

                        balanceItem.innerHTML = `
                            <div class="currency">${currency}</div>
                            <div class="balance-details">
                                <div class="balance-row"><span>Equity:</span> <span>${equity}</span></div>
                                <div class="balance-row"><span>Available:</span> <span>${available}</span></div>
                                <div class="balance-row"><span>Wallet:</span> <span>${wallet}</span></div>
                            </div>
                        `;
                        portfolioContainer.appendChild(balanceItem);
                    }
                }
            } else {
                portfolioContainer.innerHTML = '<div class="error-message">Brak danych portfela lub problem z połączeniem z ByBit. Sprawdź klucze API w ustawieniach.</div>';

                // Wyświetl dane diagnostyczne w logach konsoli
                console.log("Otrzymane dane portfela:", data);
                if (data && data.error) {
                    console.log("Błąd API:", data.error);
                }
            }
        })
        .catch(err => {
            console.error("Błąd podczas pobierania danych portfela:", err);
            const portfolioContainer = document.getElementById('portfolio-data');
            portfolioContainer.innerHTML = '<div class="error-message">Błąd podczas pobierania danych portfela. Sprawdź połączenie internetowe.</div>';
        });
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
    fetch('/api/component-status')
        .then(response => response.json())
        .then(data => {
            if (data.components) {
                data.components.forEach(component => {
                    const statusElement = document.getElementById(`${component.id}-status`);
                    if (statusElement) {
                        // Ustawienie odpowiedniego statusu i klasy CSS w zależności od faktycznego statusu
                        let statusClass, statusText;

                        if (component.status === 'online') {
                            statusClass = 'online';
                            statusText = 'Online';
                        } else if (component.status === 'warning') {
                            statusClass = 'warning';
                            statusText = 'Warning';
                        } else if (component.status === 'offline') {
                            statusClass = 'offline';
                            statusText = 'Offline';
                        } else {
                            // Domyślnie, jeśli status jest nieznany
                            statusClass = 'warning';
                            statusText = 'Unknown';
                        }

                        statusElement.textContent = statusText;
                        statusElement.className = `status-badge ${statusClass}`;
                    }
                });
            }
        })
        .catch(err => {
            console.error("Błąd podczas aktualizacji statusu komponentów:", err);
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

    // Obsługa zakładek w dashboardzie
    const tabButtons = document.querySelectorAll('.tab-button');
    if (tabButtons.length > 0) {
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Usunięcie klasy active ze wszystkich przycisków
                tabButtons.forEach(btn => btn.classList.remove('active'));
                // Dodanie klasy active do klikniętego przycisku
                this.classList.add('active');

                // Ukrycie wszystkich sekcji zawartości
                const tabId = this.getAttribute('data-tab');
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => {
                    content.style.display = 'none';
                });

                // Pokazanie wybranej sekcji
                const selectedTab = document.getElementById(tabId);
                if (selectedTab) {
                    selectedTab.style.display = 'block';
                }
            });
        });
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