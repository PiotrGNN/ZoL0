// Dashboard Configuration
window.CONFIG = {
    REFRESH_INTERVAL: 30000,  // 30 sekund - zwiększone dla zmniejszenia obciążenia API
    RETRY_INTERVAL: 60000,    // 1 minuta po błędzie
    MAX_ERRORS: 3             // Maksymalna liczba błędów przed dłuższym odstępem
};

// Liczniki błędów dla różnych endpointów
window.errorCounts = {
    dashboard: 0,
    portfolio: 0,
    trades: 0,
    chart: 0,
    notifications: 0,
    components: 0
};

// Interwały odświeżania dla poszczególnych komponentów (ms)
window.refreshRates = {
    dashboard: 30000,   // co 30s
    portfolio: 60000,   // co 1 min
    trades: 60000,      // co 1 min
    chart: 300000,      // co 5 min
    notifications: 60000, // co 1 min
    components: 30000   // co 30s
};

// Document Ready Function
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard załadowany');

    // Initialize all components
    initializeComponents();
    setupEventListeners();

    // Load initial data
    updateDashboardData();
    updateChartData();
    updateTradingStats();
    updateRecentTrades();
    updateAlerts();
    updateComponentStatus();
    updateNotifications();

    // Set up refresh intervals
    setInterval(updateDashboardData, CONFIG.REFRESH_INTERVAL);
    setInterval(updateChartData, CONFIG.chartRefresh);
    setInterval(updateRecentTrades, CONFIG.tradesRefresh);
    setInterval(updateAlerts, CONFIG.notificationsRefresh);
    setInterval(updateComponentStatus, CONFIG.componentStatusRefresh);
    setInterval(updateTradingStats, CONFIG.statsRefresh);
    setInterval(updateNotifications, CONFIG.notificationsRefresh);
});

// Initialize UI Components
function initializeComponents() {
    // Set up tabs
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');

            // Hide all tabs
            tabContents.forEach(tab => {
                tab.style.display = 'none';
            });

            // Deactivate all buttons
            tabButtons.forEach(btn => {
                btn.classList.remove('active');
            });

            // Show target tab and activate button
            document.getElementById(targetTab).style.display = 'block';
            button.classList.add('active');
        });
    });
}

// Setup Event Listeners
function setupEventListeners() {
    // Control buttons
    const startTradingBtn = document.getElementById('start-trading-btn');
    const stopTradingBtn = document.getElementById('stop-trading-btn');
    const resetSystemBtn = document.getElementById('reset-system-btn');

    if (startTradingBtn) {
        startTradingBtn.addEventListener('click', startTrading);
    }

    if (stopTradingBtn) {
        stopTradingBtn.addEventListener('click', stopTrading);
    }

    if (resetSystemBtn) {
        resetSystemBtn.addEventListener('click', resetSystem);
    }
}

// Update Dashboard Data
function updateDashboardData() {
    console.log('Aktualizacja danych dashboardu...');
    fetch('/api/dashboard/data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Update balance and profit/loss
                updateElementText('account-balance', `$${data.balance}`);
                updateElementText('profit-loss', `$${data.profit_loss}`);
                updateElementText('open-positions', data.open_positions);
                updateElementText('total-trades', data.total_trades);
                updateElementText('win-rate', `${data.win_rate}%`);
                updateElementText('max-drawdown', `${data.max_drawdown}%`);
                updateElementText('market-sentiment', data.market_sentiment);
                updateElementText('last-updated', data.last_updated);

                // Reset error counter on success
                errorCounts.dashboard = 0;
            } else {
                handleApiError('dashboard');
            }
        })
        .catch(error => {
            handleApiError('dashboard', error);
        });
}

// Update Chart Data
function updateChartData() {
    fetch('/api/chart-data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Update chart if Chart.js is available
                if (window.Chart && data.data) {
                    updatePortfolioChart(data.data);
                }

                // Reset error counter on success
                errorCounts.chart = 0;
            } else {
                handleApiError('chart');
            }
        })
        .catch(error => {
            handleApiError('chart', error);
        });
}

// Update Portfolio Chart
function updatePortfolioChart(chartData) {
    try {
        const ctx = document.getElementById('portfolio-chart');
        if (ctx) {
            // Check if chart already exists
            if (window.portfolioChart) {
                // Update existing chart
                window.portfolioChart.data = chartData;
                window.portfolioChart.update();
            } else {
                // Create new chart
                window.portfolioChart = new Chart(ctx, {
                    type: 'line',
                    data: chartData,
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: false
                            }
                        }
                    }
                });
            }
        }
    } catch (error) {
        handleApiError('chart');
    }
}

// Update Trading Stats
function updateTradingStats() {
    fetch('/api/trading-stats')
        .then(response => {
            // Sprawdź kod HTTP odpowiedzi
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateElementText('profit-value', data.profit);
            updateElementText('trades-count-value', data.trades_count);
            updateElementText('win-rate-value', data.win_rate);
            updateElementText('drawdown-value', data.max_drawdown);

            // Wyświetl komunikat o sukcesie w konsoli
            console.log("Statystyki tradingowe zaktualizowane poprawnie:", data);

            // Dodaj informacje do panelu diagnostycznego, jeśli istnieje
            if (document.getElementById('debug-panel')) {
                const debugInfo = document.createElement('div');
                debugInfo.className = 'debug-success';
                debugInfo.innerHTML = `<small>${new Date().toLocaleTimeString()}: Stats OK</small>`;
                document.getElementById('debug-panel').prepend(debugInfo);

                // Ogranicz liczbę wpisów
                const entries = document.getElementById('debug-panel').children;
                if (entries.length > 20) {
                    document.getElementById('debug-panel').removeChild(entries[entries.length - 1]);
                }
            }
        })
        .catch(error => {
            // Szczegółowe logowanie błędu
            console.error("Błąd podczas aktualizacji statystyk tradingowych:", error);

            // Dodaj panel debugowania, jeśli nie istnieje
            if (!document.getElementById('debug-panel')) {
                const panel = document.createElement('div');
                panel.id = 'debug-panel';
                panel.className = 'debug-panel';
                panel.innerHTML = '<h4>Diagnostyka API</h4>';
                document.querySelector('.content-area').appendChild(panel);

                // Dodaj style CSS
                const style = document.createElement('style');
                style.textContent = `
                    .debug-panel {
                        position: fixed;
                        bottom: 10px;
                        right: 10px;
                        max-width: 400px;
                        max-height: 300px;
                        overflow-y: auto;
                        background: rgba(0,0,0,0.8);
                        color: #eee;
                        border-radius: 5px;
                        padding: 10px;
                        font-size: 12px;
                        z-index: 1000;
                    }
                    .debug-error { color: #ff6b6b; margin-bottom: 5px; }
                    .debug-success { color: #51cf66; margin-bottom: 5px; }
                `;
                document.head.appendChild(style);
            }

            // Dodaj informacje o błędzie do panelu
            const debugInfo = document.createElement('div');
            debugInfo.className = 'debug-error';
            debugInfo.innerHTML = `<small>${new Date().toLocaleTimeString()}: ${error.message}</small>`;
            document.getElementById('debug-panel').prepend(debugInfo);

            // Inkrementuj licznik błędów
            window.errorCounts.dashboard++;
        });
}

// Update Recent Trades
function updateRecentTrades() {
    fetch('/api/recent-trades')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const tradesContainer = document.getElementById('recent-trades-container');
            if (tradesContainer && data.trades) {
                // Clear container
                tradesContainer.innerHTML = '';

                // Add trades
                data.trades.forEach(trade => {
                    const tradeElement = document.createElement('div');
                    tradeElement.className = `trade-item ${trade.profit >= 0 ? 'profit' : 'loss'}`;
                    tradeElement.innerHTML = `
                        <span class="trade-symbol">${trade.symbol}</span>
                        <span class="trade-type">${trade.type}</span>
                        <span class="trade-time">${trade.time}</span>
                        <span class="trade-profit ${trade.profit >= 0 ? 'positive' : 'negative'}">
                            ${trade.profit >= 0 ? '+' : ''}${trade.profit}%
                        </span>
                    `;
                    tradesContainer.appendChild(tradeElement);
                });

                // Reset error counter on success
                errorCounts.trades = 0;
            }
        })
        .catch(error => {
            handleApiError('trades', error);
        });
}

// Update Alerts
function updateAlerts() {
    fetch('/api/alerts')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const alertsContainer = document.getElementById('alerts-container');
            if (alertsContainer && data.alerts) {
                // Clear container
                alertsContainer.innerHTML = '';

                // Add alerts
                data.alerts.forEach(alert => {
                    const alertElement = document.createElement('div');
                    alertElement.className = `alert-item ${alert.level_class}`;
                    alertElement.innerHTML = `
                        <span class="alert-time">${alert.time}</span>
                        <span class="alert-message">${alert.message}</span>
                        <span class="alert-level">${alert.level}</span>
                    `;
                    alertsContainer.appendChild(alertElement);
                });

                // Reset error counter on success
                errorCounts.alerts = 0;
            }
        })
        .catch(error => {
            handleApiError('alerts', error);
        });
}

// Update Component Status
function updateComponentStatus() {
    console.log('Aktualizacja statusów komponentów...');
    fetch('/api/component-status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.components) {
                data.components.forEach(component => {
                    const element = document.getElementById(component.id);
                    if (element) {
                        // Remove all status classes
                        element.classList.remove('online', 'warning', 'offline');
                        // Add current status class
                        element.classList.add(component.status);
                    }
                });

                // Reset error counter on success
                errorCounts.components = 0;
            }
        })
        .catch(error => {
            handleApiError('components', error);
        });
}

// Update Notifications
function updateNotifications() {
    fetch('/api/notifications')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.notifications) {
                const notificationsContainer = document.getElementById('notifications-container');
                if (notificationsContainer) {
                    // Clear container
                    notificationsContainer.innerHTML = '';

                    // Add notifications
                    data.notifications.forEach(notification => {
                        const notifElement = document.createElement('div');
                        notifElement.className = `notification-item ${notification.type}`;
                        notifElement.innerHTML = `
                            <span class="notification-time">${notification.timestamp}</span>
                            <span class="notification-message">${notification.message}</span>
                        `;
                        notificationsContainer.appendChild(notifElement);
                    });

                    // Update badge
                    const badge = document.getElementById('notifications-badge');
                    if (badge) {
                        badge.textContent = data.notifications.length;
                        badge.style.display = data.notifications.length > 0 ? 'inline' : 'none';
                    }
                }

                // Reset error counter on success
                errorCounts.notifications = 0;
            }
        })
        .catch(error => {
            handleApiError('notifications', error);
        });
}

// Control Functions
function startTrading() {
    fetch('/api/trading/start', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage('success', data.message);
        } else {
            showMessage('error', data.error);
        }
    })
    .catch(error => {
        showMessage('error', 'Błąd podczas uruchamiania handlu: ' + error.message);
    });
}

function stopTrading() {
    fetch('/api/trading/stop', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showMessage('success', data.message);
        } else {
            showMessage('error', data.error);
        }
    })
    .catch(error => {
        showMessage('error', 'Błąd podczas zatrzymywania handlu: ' + error.message);
    });
}

function resetSystem() {
    if (confirm('Czy na pewno chcesz zresetować system? Ta operacja zatrzyma handel i wyczyści obecne sesje.')) {
        fetch('/api/system/reset', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showMessage('success', data.message);

                // Reload all data
                updateDashboardData();
                updateChartData();
                updateTradingStats();
                updateRecentTrades();
                updateAlerts();
                updateComponentStatus();
            } else {
                showMessage('error', data.error);
            }
        })
        .catch(error => {
            showMessage('error', 'Błąd podczas resetowania systemu: ' + error.message);
        });
    }
}

// Helper Functions
function updateElementText(elementId, text) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = text;
    }
}

function showMessage(type, message) {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
        errorContainer.style.display = 'block';

        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorContainer.style.display = 'none';
        }, 5000);
    }
}

function handleApiError(component, error) {
    // Increment error counter
    errorCounts[component]++;

    console.log(`Błąd podczas aktualizacji ${component}:`, error);

    // Show error after 3 consecutive failures
    if (errorCounts[component] >= 3) {
        const errorMessage = `Błąd podczas aktualizacji danych ${component}. Sprawdź połączenie sieciowe.`;
        showMessage('error', errorMessage);

        // Reset counter to prevent spam
        errorCounts[component] = 0;
    }

    // Retry after delay if not exceeding max retries
    if (errorCounts[component] <= CONFIG.maxRetries) {
        setTimeout(() => {
            // Call appropriate update function based on component
            switch(component) {
                case 'dashboard':
                    updateDashboardData();
                    break;
                case 'chart':
                    updateChartData();
                    break;
                case 'trades':
                    updateRecentTrades();
                    break;
                case 'alerts':
                    updateAlerts();
                    break;
                case 'components':
                    updateComponentStatus();
                    break;
                case 'stats':
                    updateTradingStats();
                    break;
                case 'notifications':
                    updateNotifications();
                    break;
            }
        }, CONFIG.retryDelay);
    }
}