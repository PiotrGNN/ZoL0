
// Dashboard.js - Główny skrypt dla dashboardu tradingowego
// Zoptymalizowana wersja z lepszym zarządzaniem zapytaniami API

// Konfiguracja globalna
const CONFIG = {
    // Częstotliwość odświeżania poszczególnych elementów (ms)
    refreshRates: {
        dashboard: 15000,     // Dashboard (podstawowe dane)
        portfolio: 30000,     // Portfolio (dane konta)
        charts: 60000,        // Wykresy (cięższe dane)
        trades: 45000,        // Transakcje
        components: 20000     // Statusy komponentów
    },
    // Maksymalna liczba błędów przed wyświetleniem ostrzeżenia
    maxErrors: 3,
    // Parametry retry dla zapytań API
    retry: {
        maxRetries: 3,
        delayMs: 2000,
        backoffMultiplier: 1.5
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

    // Minimalny czas między wywołaniami tego samego endpointu (1 sekunda)
    const minDelayMs = 1000;

    if (timeSinceLastCall < minDelayMs) {
        await new Promise(resolve => setTimeout(resolve, minDelayMs - timeSinceLastCall));
    }

    // Aktualizacja czasu ostatniego wywołania
    appState.lastApiCall[endpoint] = Date.now();

    // Próby wykonania zapytania z ponowieniami
    let retries = 0;
    let delay = CONFIG.retry.delayMs;

    while (retries <= CONFIG.retry.maxRetries) {
        try {
            const response = await fetch(url, options);

            if (response.status === 429 || response.status === 403) {
                // Rate limit przekroczony
                throw new Error(`Rate limit exceeded (status: ${response.status})`);
            }

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            retries++;

            if (retries > CONFIG.retry.maxRetries) {
                // Zwiększ licznik błędów dla danego endpointu
                appState.errorCounts[endpoint] = (appState.errorCounts[endpoint] || 0) + 1;

                // Zwróć błąd po wyczerpaniu prób
                throw error;
            }

            // Opóźnienie przed ponowną próbą z wykładniczym backoff
            await new Promise(resolve => setTimeout(resolve, delay));
            delay *= CONFIG.retry.backoffMultiplier;
        }
    }
}

// Pobieranie i aktualizacja głównych danych dashboardu
async function updateDashboardData() {
    console.log("Aktualizacja danych dashboardu...");

    try {
        // Pobierz dane portfela jeśli widoczny jest dashboard
        if (document.getElementById('dashboard-tab') && document.getElementById('dashboard-tab').style.display === 'block') {
            fetchPortfolioData();
        }
        updateTradingStats();
        updateRecentTrades();
        updateAlerts();
        updateNotifications();
    } catch (error) {
        handleApiError('dashboard');
        console.error('Błąd podczas aktualizacji danych dashboardu:', error);
    }
}

// Pobieranie danych portfela
async function fetchPortfolioData() {
    try {
        // Dodajemy timestamp do URL, aby uniknąć cache'owania
        const data = await fetchWithRateLimit(`/api/portfolio?_=${Date.now()}`);

        // Aktualizacja UI z danymi portfela
        const portfolioContainer = document.getElementById('portfolio-container');
        if (!portfolioContainer) return;

        if (data.balances) {
            let html = '';

            // Iteracja po wszystkich walutach
            for (const [currency, balanceData] of Object.entries(data.balances)) {
                html += `
                    <div class="balance-item">
                        <div class="balance-header">
                            <div class="balance-symbol">${currency}</div>
                            <div class="balance-value">${parseFloat(balanceData.equity).toFixed(5)}</div>
                        </div>
                        <div class="balance-details">
                            <div class="balance-detail">
                                <span>Available:</span>
                                <span>${parseFloat(balanceData.available_balance).toFixed(5)}</span>
                            </div>
                            <div class="balance-detail">
                                <span>Wallet:</span>
                                <span>${parseFloat(balanceData.wallet_balance).toFixed(5)}</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            // Dodawanie informacji o źródle danych (API/symulacja)
            if (data.source) {
                html += `<div class="data-source ${data.source.includes('simulation') ? 'simulation' : ''}">
                    Source: ${data.source}
                </div>`;
            }

            // Jeśli dane są symulowane, pokazujemy odpowiednią informację
            if (data.warning) {
                html += `<div class="warning-note">${data.warning}</div>`;
            }

            portfolioContainer.innerHTML = html;
        } else if (data.error) {
            portfolioContainer.innerHTML = `<div class="error-message">Error: ${data.error}</div>`;
        }
    } catch (error) {
        handleApiError('portfolio');
        console.error('Błąd podczas pobierania danych portfela:', error);
    }
}

// Aktualizacja statusów komponentów systemu
async function updateComponentStatuses() {
    console.log("Aktualizacja statusów komponentów...");

    try {
        // Pobieranie statusu wszystkich komponentów
        const data = await fetchWithRateLimit('/api/component-status');

        if (data && data.components) {
            // Aktualizacja wskaźników statusu na UI
            data.components.forEach(component => {
                const componentElement = document.getElementById(component.id);
                if (componentElement) {
                    // Usunięcie wszystkich klas statusu
                    componentElement.classList.remove('status-online', 'status-offline', 'status-warning');
                    // Dodanie właściwej klasy statusu
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
            const ctx = document.getElementById('main-chart');
            if (!ctx) return;

            // Sprawdź czy wykres już istnieje
            if (window.mainChart) {
                // Aktualizuj istniejący wykres
                window.mainChart.data = data.data;
                window.mainChart.update();
            } else {
                // Utwórz nowy wykres
                window.mainChart = new Chart(ctx, {
                    type: 'line',
                    data: data.data,
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
        console.error('Błąd podczas aktualizacji wykresu:', error);
    }
}

// Aktualizacja statystyk tradingowych
async function updateTradingStats() {
    try {
        const data = await fetchWithRateLimit('/api/trading-stats');

        // Aktualizacja elementów UI - dodano sprawdzenie czy element istnieje
        if (data) {
            safeUpdateElement('profit-value', data.profit || '$0.00');
            safeUpdateElement('trades-count', data.trades_count || '0');
            safeUpdateElement('win-rate', data.win_rate || '0%');
            safeUpdateElement('max-drawdown', data.max_drawdown || '0%');
        }
    } catch (error) {
        handleApiError('stats');
        console.error('Błąd podczas aktualizacji statystyk tradingowych:', error);
    }
}

// Pobieranie ostatnich transakcji
async function updateRecentTrades() {
    try {
        const data = await fetchWithRateLimit('/api/recent-trades');

        const tradeTableBody = document.getElementById('trades-table-body');
        if (!tradeTableBody) return;

        if (data && data.trades && data.trades.length > 0) {
            let html = '';

            data.trades.forEach((trade, index) => {
                const profitClass = parseFloat(trade.profit) >= 0 ? 'profit-positive' : 'profit-negative';

                html += `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${trade.time}</td>
                        <td>${trade.symbol}</td>
                        <td class="${trade.type.toLowerCase() === 'buy' ? 'type-buy' : 'type-sell'}">${trade.type}</td>
                        <td>$${(Math.random() * 1000 + 20000).toFixed(2)}</td>
                        <td>${(Math.random() * 2).toFixed(4)}</td>
                        <td>$${(Math.random() * 5000 + 1000).toFixed(2)}</td>
                        <td class="${profitClass}">${trade.profit > 0 ? '+' : ''}${trade.profit}%</td>
                        <td>Completed</td>
                    </tr>
                `;
            });

            tradeTableBody.innerHTML = html;
        } else {
            tradeTableBody.innerHTML = '<tr><td colspan="9" class="no-data">Brak dostępnych transakcji</td></tr>';
        }
    } catch (error) {
        handleApiError('trades');
        console.error('Błąd podczas pobierania ostatnich transakcji:', error);
    }
}

// Pobieranie i wyświetlanie alertów
async function updateAlerts() {
    try {
        const data = await fetchWithRateLimit('/api/alerts');

        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;

        if (data && data.alerts && data.alerts.length > 0) {
            let html = '';

            data.alerts.forEach(alert => {
                html += `
                    <div class="alert-item alert-${alert.level_class}">
                        <div class="alert-time">${alert.time}</div>
                        <div class="alert-level">${alert.level}</div>
                        <div class="alert-message">${alert.message}</div>
                    </div>
                `;
            });

            alertsContainer.innerHTML = html;
        } else {
            alertsContainer.innerHTML = '<div class="no-data">Brak alertów</div>';
        }
    } catch (error) {
        handleApiError('alerts');
        console.error('Błąd podczas pobierania alertów:', error);
    }
}

// Pobieranie i wyświetlanie powiadomień
async function updateNotifications() {
    try {
        const data = await fetchWithRateLimit('/api/notifications');

        const notificationsContainer = document.getElementById('notifications-container');
        if (!notificationsContainer) return;

        if (data && data.success && data.notifications && data.notifications.length > 0) {
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

            // Aktualizacja licznika powiadomień
            const badge = document.getElementById('notifications-badge');
            if (badge) {
                badge.textContent = data.notifications.length;
                badge.style.display = 'inline-block';
            }
        } else {
            notificationsContainer.innerHTML = '<div class="no-data">Brak powiadomień</div>';
        }
    } catch (error) {
        handleApiError('notifications');
        console.error('Błąd podczas pobierania powiadomień:', error);
    }
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

        // Ukryj komunikat po 5 sekundach
        setTimeout(() => {
            errorContainer.style.display = 'none';
        }, 5000);
    }
}

function setupEventListeners() {
    // Obsługa przycisków
    const startButton = document.getElementById('start-trading-btn');
    if (startButton) {
        startButton.addEventListener('click', startTrading);
    }

    const stopButton = document.getElementById('stop-trading-btn');
    if (stopButton) {
        stopButton.addEventListener('click', stopTrading);
    }

    const resetButton = document.getElementById('reset-system-btn');
    if (resetButton) {
        resetButton.addEventListener('click', resetSystem);
    }

    // Obsługa tabs/kart
    const tabButtons = document.querySelectorAll('.tab-button');
    if (tabButtons.length > 0) {
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                const tabId = this.getAttribute('data-tab');

                // Aktualizacja stanu aktywności dashboardu
                appState.activeDashboard = (tabId === 'dashboard-tab');

                // Usuń klasę active ze wszystkich przycisków
                tabButtons.forEach(btn => btn.classList.remove('active'));

                // Dodaj klasę active do klikniętego przycisku
                this.classList.add('active');

                // Ukryj wszystkie kontenery kart
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.style.display = 'none';
                });

                // Pokaż wybrany kontener
                const selectedTab = document.getElementById(tabId);
                if (selectedTab) {
                    selectedTab.style.display = 'block';
                }

                // Załaduj dane specyficzne dla wybranej karty
                if (tabId === 'dashboard-tab') {
                    updateDashboardData();
                } else if (tabId === 'trades-tab') {
                    updateRecentTrades();
                } else if (tabId === 'analytics-tab') {
                    updateChartData();
                } else if (tabId === 'notifications-tab') {
                    updateNotifications();
                }
            });
        });
    }
}

// Funkcje dla przycisków akcji
function startTrading() {
    if (confirm('Czy na pewno chcesz uruchomić trading automatyczny?')) {
        fetch('/api/trading/start', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Trading automatyczny uruchomiony!');
                updateComponentStatuses();
            } else {
                alert(`Błąd: ${data.error || 'Nie udało się uruchomić tradingu.'}`);
            }
        })
        .catch(error => {
            console.error('Błąd podczas uruchamiania tradingu:', error);
            alert('Wystąpił błąd podczas próby uruchomienia tradingu.');
        });
    }
}

function stopTrading() {
    if (confirm('Czy na pewno chcesz zatrzymać trading automatyczny?')) {
        fetch('/api/trading/stop', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Trading automatyczny zatrzymany!');
                updateComponentStatuses();
            } else {
                alert(`Błąd: ${data.error || 'Nie udało się zatrzymać tradingu.'}`);
            }
        })
        .catch(error => {
            console.error('Błąd podczas zatrzymywania tradingu:', error);
            alert('Wystąpił błąd podczas próby zatrzymania tradingu.');
        });
    }
}

function resetSystem() {
    if (confirm('Czy na pewno chcesz zresetować system? Wszystkie ustawienia zostaną przywrócone do wartości domyślnych.')) {
        fetch('/api/system/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('System został zresetowany.');
                // Przeładuj stronę, aby zmiany były widoczne
                window.location.reload();
            } else {
                alert(`Błąd: ${data.error || 'Nie udało się zresetować systemu.'}`);
            }
        })
        .catch(error => {
            console.error('Błąd podczas resetowania systemu:', error);
            alert('Wystąpił błąd podczas próby resetowania systemu.');
        });
    }
}

function initializeChart() {
    const ctx = document.getElementById('portfolio-chart') || document.getElementById('main-chart');
    if (ctx) {
        // Tworzymy tymczasowy wykres do czasu załadowania danych
        renderChart(ctx, {
            labels: ['Ładowanie...'],
            datasets: [{
                label: 'Wartość Portfela',
                data: [0],
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)'
            }]
        });
    }
}

function renderChart(canvas, data) {
    if (!canvas) return;

    // Sprawdź czy wykres już istnieje i go zniszcz
    if (canvas.chart) {
        canvas.chart.destroy();
    }

    // Utwórz nowy wykres
    canvas.chart = new Chart(canvas, {
        type: 'line',
        data: data,
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

// Bezpieczna aktualizacja elementu - sprawdza czy element istnieje
function safeUpdateElement(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    }
}

function getAccuracyClass(accuracy) {
    if (accuracy >= 70) return 'positive';
    if (accuracy >= 50) return 'neutral';
    return 'negative';
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}
