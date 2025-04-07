// Dashboard Configuration
const CONFIG = {
    dashboardRefresh: 30000,   // Odświeżanie dashboardu co 30s
    componentStatusRefresh: 30000, // Odświeżanie statusu komponentów co 30s
    chartRefresh: 60000,    // Odświeżanie wykresu co 60s
    tradesRefresh: 60000,   // Odświeżanie listy transakcji co 60s
    notificationsRefresh: 30000, // Odświeżanie powiadomień co 30s
    statsRefresh: 30000     // Odświeżanie statystyk co 30s
};

// Liczniki błędów dla poszczególnych elementów
let errorCounts = {
    dashboard: 0,
    chart: 0,
    trades: 0,
    notifications: 0,
    components: 0,
    stats: 0
};

// Opóźnienia odświeżania w razie błędów
let refreshRates = {
    normal: {
        dashboardRefresh: 30000,
        componentStatusRefresh: 30000,
        chartRefresh: 60000,
        tradesRefresh: 60000,
        notificationsRefresh: 30000,
        statsRefresh: 30000
    },
    reduced: {
        dashboardRefresh: 60000,
        componentStatusRefresh: 60000,
        chartRefresh: 120000,
        tradesRefresh: 120000,
        notificationsRefresh: 60000,
        statsRefresh: 60000
    },
    minimal: {
        dashboardRefresh: 300000,
        componentStatusRefresh: 300000,
        chartRefresh: 600000,
        tradesRefresh: 600000,
        notificationsRefresh: 300000,
        statsRefresh: 300000
    }
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
    setInterval(updateDashboardData, CONFIG.dashboardRefresh);
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

            // Reset error counter on success
            errorCounts.stats = 0;
        })
        .catch(error => {
            handleApiError('stats', error);
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