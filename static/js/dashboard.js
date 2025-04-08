// Dashboard.js - Główny skrypt dla dashboardu tradingowego
// Zoptymalizowana wersja z lepszym zarządzaniem zapytaniami API

// Konfiguracja globalna
const CONFIG = {
    // Częstotliwość odświeżania poszczególnych elementów (ms)
    refreshRates: {
        dashboard: 30000,     // Dashboard (podstawowe dane) - zwiększono z 15000
        portfolio: 60000,     // Portfolio (dane konta) - zwiększono z 30000
        charts: 120000,       // Wykresy (cięższe dane) - zwiększono z 60000
        trades: 90000,        // Transakcje - zwiększono z 45000
        components: 60000     // Statusy komponentów - zwiększono z 20000
    },
    // Maksymalna liczba błędów przed wyświetleniem ostrzeżenia
    maxErrors: 3,
    // Parametry retry dla zapytań API
    retry: {
        maxRetries: 3,
        delayMs: 5000,
        backoffMultiplier: 2.0
    }
};

// Stan aplikacji
const appState = {
    errorCounts: {},    // Liczniki błędów dla różnych endpointów
    timers: {},         // Identyfikatory setTimeout dla różnych odświeżeń
    activeDashboard: true, // Czy dashboard jest aktywną zakładką
    lastApiCall: {}     // Czas ostatniego wywołania dla każdego API
};

// Inicjalizacja po załadowaniu strony
document.addEventListener('DOMContentLoaded', function() {
    console.log("Dashboard załadowany");
    setupEventListeners();
    initializeDashboard();
});

function initializeDashboard() {
    // Pierwsza inicjalizacja danych
    updateDashboardData();
    updateComponentStatuses();

    // Ustawienie timerów dla cyklicznego odświeżania
    scheduleDataRefresh();
}

function scheduleDataRefresh() {
    // Czyszczenie istniejących timerów
    Object.values(appState.timers).forEach(timer => clearTimeout(timer));

    // Ustawienie nowych timerów dla różnych typów danych
    appState.timers.dashboard = setTimeout(() => {
        updateDashboardData();
        scheduleDataRefresh();
    }, CONFIG.refreshRates.dashboard);

    appState.timers.portfolio = setTimeout(() => {
        if (appState.activeDashboard) fetchPortfolioData();
    }, CONFIG.refreshRates.portfolio);

    appState.timers.charts = setTimeout(() => {
        if (appState.activeDashboard) updateChartData();
    }, CONFIG.refreshRates.charts);

    appState.timers.trades = setTimeout(() => {
        if (appState.activeDashboard) updateRecentTrades();
    }, CONFIG.refreshRates.trades);

    appState.timers.components = setTimeout(() => {
        updateComponentStatuses();
    }, CONFIG.refreshRates.components);
}

// Funkcje API - z obsługą rate limiting
async function fetchWithRateLimit(url, options = {}) {
    const endpoint = url.split('?')[0]; // Podstawowy endpoint bez parametrów

    // Sprawdzenie czasu od ostatniego wywołania tego endpointu
    const now = Date.now();
    const lastCall = appState.lastApiCall[endpoint] || 0;
    const timeSinceLastCall = now - lastCall;

    // Minimalny czas między wywołaniami tego samego endpointu (3 sekundy)
    const minTimeBetweenCalls = 3000;

    if (timeSinceLastCall < minTimeBetweenCalls) {
        // Jeśli nie minęło wystarczająco dużo czasu, czekamy
        const timeToWait = minTimeBetweenCalls - timeSinceLastCall;
        await new Promise(resolve => setTimeout(resolve, timeToWait));
    }

    try {
        // Aktualizacja czasu ostatniego wywołania
        appState.lastApiCall[endpoint] = Date.now();

        // Wykonanie zapytania
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        // Increment error counter for endpoint
        appState.errorCounts[endpoint] = (appState.errorCounts[endpoint] || 0) + 1;
        console.error(`Błąd podczas pobierania danych z ${endpoint}:`, error);
        throw error;
    }
}

// Aktualizacja danych dashboardu
async function updateDashboardData() {
    console.log("Aktualizacja danych dashboardu...");
    try {
        // Pobieranie danych dashboardu
        const dashboardData = await fetchWithRateLimit('/api/dashboard/data');

        // Aktualizacja każdej sekcji
        updateTradingStats();
        updateAlerts();
        updateRecentTrades();
        updateNotifications();
        updateChartData();
        updateComponentStatuses();
        fetchPortfolioData();

    } catch (error) {
        handleApiError('dashboard_data');
        console.error('Błąd podczas aktualizacji danych dashboardu:', error);
    }
}

// Aktualizacja danych portfela
async function fetchPortfolioData() {
    try {
        const data = await fetchWithRateLimit(`/api/portfolio?_=${Date.now()}`);
        if (data && data.balances) {
            updatePortfolioUI(data);
        }
    } catch (error) {
        handleApiError('portfolio');
        console.error('Błąd podczas pobierania danych portfela:', error);
    }
}

// Aktualizacja interfejsu portfela
function updatePortfolioUI(data) {
    try {
        const portfolioContainer = document.getElementById('portfolio-data');
        if (!portfolioContainer) return;

        let html = '<table class="portfolio-table"><thead><tr><th>Aktywo</th><th>Saldo</th><th>Dostępne</th></tr></thead><tbody>';

        for (const [coin, details] of Object.entries(data.balances)) {
            html += `
                <tr>
                    <td>${coin}</td>
                    <td>${details.wallet_balance.toFixed(5)}</td>
                    <td>${details.available_balance.toFixed(5)}</td>
                </tr>
            `;
        }

        html += '</tbody></table>';

        if (data.source && data.source.includes('simulation')) {
            html += '<div class="simulation-notice">Dane symulowane</div>';
        }

        portfolioContainer.innerHTML = html;
    } catch (error) {
        console.error('Błąd podczas aktualizacji UI portfela:', error);
    }
}

// Aktualizacja ostatnich transakcji
async function updateRecentTrades() {
    try {
        const data = await fetchWithRateLimit('/api/recent-trades');
        if (data && data.trades) {
            const tradesContainer = document.getElementById('recent-trades');
            if (tradesContainer) {
                let html = '';
                data.trades.forEach(trade => {
                    const profitClass = trade.profit >= 0 ? 'profit-positive' : 'profit-negative';
                    html += `
                        <div class="trade-item">
                            <div class="trade-symbol">${trade.symbol}</div>
                            <div class="trade-type ${trade.type.toLowerCase()}">${trade.type}</div>
                            <div class="trade-time">${trade.time}</div>
                            <div class="trade-profit ${profitClass}">${trade.profit >= 0 ? '+' : ''}${trade.profit}%</div>
                        </div>
                    `;
                });
                tradesContainer.innerHTML = html;
            }
        }
    } catch (error) {
        handleApiError('trades');
        console.error('Błąd podczas pobierania ostatnich transakcji:', error);
    }
}

// Aktualizacja statystyk tradingowych
async function updateTradingStats() {
    try {
        const data = await fetchWithRateLimit('/api/trading-stats');
        if (data) {
            // Aktualizacja każdego elementu statystyk
            const stats = {
                'profit-value': data.profit,
                'trades-count': data.trades_count,
                'win-rate': data.win_rate,
                'max-drawdown': data.max_drawdown
            };

            for (const [id, value] of Object.entries(stats)) {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = value;
                }
            }
        }
    } catch (error) {
        handleApiError('trading_stats');
        console.error('Błąd podczas aktualizacji statystyk tradingowych:', error);
    }
}

// Aktualizacja alertów
async function updateAlerts() {
    try {
        const data = await fetchWithRateLimit('/api/alerts');
        if (data && data.alerts) {
            const alertsContainer = document.getElementById('alerts-container');
            if (alertsContainer) {
                let html = '';
                data.alerts.forEach(alert => {
                    html += `
                        <div class="alert-item">
                            <div class="alert-level ${alert.level_class}">${alert.level}</div>
                            <div class="alert-time">${alert.time}</div>
                            <div class="alert-message">${alert.message}</div>
                        </div>
                    `;
                });
                alertsContainer.innerHTML = html;
            }
        }
    } catch (error) {
        handleApiError('alerts');
        console.error('Błąd podczas aktualizacji alertów:', error);
    }
}

// Aktualizacja powiadomień
async function updateNotifications() {
    try {
        const data = await fetchWithRateLimit('/api/notifications');
        if (data && data.notifications) {
            const notificationsContainer = document.getElementById('notifications-container');
            if (notificationsContainer) {
                let html = '';
                data.notifications.forEach(notification => {
                    html += `
                        <div class="notification-item notification-${notification.type}">
                            <div class="notification-time">${notification.timestamp}</div>
                            <div class="notification-message">${notification.message}</div>
                        </div>
                    `;
                });
                notificationsContainer.innerHTML = html;
            }
        }
    } catch (error) {
        handleApiError('notifications');
        console.error('Błąd podczas pobierania powiadomień:', error);
    }
}

// Aktualizacja statusów komponentów
async function updateComponentStatuses() {
    console.log("Aktualizacja statusów komponentów...");
    try {
        const data = await fetchWithRateLimit('/api/component-status');
        if (data && data.components) {
            // Aktualizacja każdego komponentu
            data.components.forEach(component => {
                const componentElement = document.getElementById(component.id);
                if (componentElement) {
                    // Usunięcie wszystkich klas statusów
                    componentElement.classList.remove('status-online', 'status-warning', 'status-error', 'status-offline');

                    // Dodanie klasy odpowiadającej aktualnemu statusowi
                    componentElement.classList.add(`status-${component.status}`);

                    // Aktualizacja tekstu statusu
                    const statusText = componentElement.querySelector('.status-text');
                    if (statusText) {
                        statusText.textContent = component.status.charAt(0).toUpperCase() + component.status.slice(1);
                    }
                }
            });
        }
    } catch (error) {
        handleApiError('components');
        console.error('Błąd podczas aktualizacji statusów komponentów:', error);
    }
}

// Aktualizacja wykresu głównego
async function updateChartData() {
    try {
        const data = await fetchWithRateLimit('/api/chart-data');

        if (data && data.success && data.data) {
            const ctx = document.getElementById('portfolio-chart');
            if (!ctx) return;

            // Sprawdź czy wykres już istnieje
            if (window.portfolioChart) {
                // Aktualizacja danych istniejącego wykresu
                window.portfolioChart.data = data.data;
                window.portfolioChart.update();
            } else {
                // Utworzenie nowego wykresu
                window.portfolioChart = new Chart(ctx, {
                    type: 'line',
                    data: data.data,
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: false,
                                grid: {
                                    color: 'rgba(200, 200, 200, 0.1)'
                                }
                            },
                            x: {
                                grid: {
                                    display: false
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false
                            }
                        },
                        elements: {
                            line: {
                                tension: 0.3
                            }
                        }
                    }
                });
            }
        }
    } catch (error) {
        handleApiError('chart');
        console.error('Błąd podczas aktualizacji wykresu:', error);
    }
}

// Funkcje związane z systemem tradingowym
function startTrading() {
    fetch('/api/trading/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('success', 'Trading automatyczny uruchomiony');
            document.getElementById('trading-status').textContent = 'Aktywny';
            document.getElementById('trading-status').className = 'status-online';
        } else {
            showNotification('error', `Błąd: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Błąd podczas uruchamiania tradingu:', error);
        showNotification('error', 'Nie udało się uruchomić tradingu automatycznego');
    });
}

function stopTrading() {
    fetch('/api/trading/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('success', 'Trading automatyczny zatrzymany');
            document.getElementById('trading-status').textContent = 'Nieaktywny';
            document.getElementById('trading-status').className = 'status-offline';
        } else {
            showNotification('error', `Błąd: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Błąd podczas zatrzymywania tradingu:', error);
        showNotification('error', 'Nie udało się zatrzymać tradingu automatycznego');
    });
}

function resetSystem() {
    if (confirm('Czy na pewno chcesz zresetować system? Wszystkie aktywne operacje zostaną zakończone.')) {
        fetch('/api/system/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification('success', 'System został zresetowany');
                // Odświeżenie wszystkich danych
                updateDashboardData();
            } else {
                showNotification('error', `Błąd: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Błąd podczas resetowania systemu:', error);
            showNotification('error', 'Nie udało się zresetować systemu');
        });
    }
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
        errorContainer.innerHTML = `<div class="error-message">${message}</div>`;
        errorContainer.style.display = 'block';
    }
}

// Wyświetlanie powiadomienia
function showNotification(type, message) {
    const notificationContainer = document.getElementById('notification-popup');
    if (!notificationContainer) {
        // Utwórz kontener powiadomień, jeśli nie istnieje
        const container = document.createElement('div');
        container.id = 'notification-popup';
        document.body.appendChild(container);
    }

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    document.getElementById('notification-popup').appendChild(notification);

    // Automatyczne ukrycie po 5 sekundach
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 5000);
}