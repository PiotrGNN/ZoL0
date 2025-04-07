
// Dashboard.js - Główny skrypt dla dashboardu tradingowego
// Wersja zoptymalizowana z lepszym zarządzaniem zapytaniami API i obsługą błędów

// Konfiguracja globalna
const CONFIG = {
    // Częstotliwość odświeżania poszczególnych elementów (ms)
    refreshRates: {
        dashboard: 60000,     // Dashboard (podstawowe dane) - co 60 sekund
        portfolio: 120000,    // Portfolio (dane konta) - co 2 minuty
        charts: 180000,       // Wykresy (cięższe dane) - co 3 minuty
        trades: 150000,       // Transakcje - co 2.5 minuty
        components: 90000     // Statusy komponentów - co 1.5 minuty
    },
    // Maksymalna liczba błędów przed wyświetleniem ostrzeżenia
    maxErrors: 3,
    // Parametry retry dla zapytań API
    retry: {
        maxRetries: 2,
        delayMs: 15000,
        backoffMultiplier: 2.0
    },
    // Minimalne czasy między zapytaniami tego samego typu (ms)
    minTimeBetweenCalls: {
        dashboard: 30000,  // Minimum 30s między odświeżeniami dashboardu
        portfolio: 60000,  // Minimum 60s między zapytaniami o portfolio
        api: 5000          // Minimum 5s między jakimikolwiek zapytaniami API
    }
};

// Stan aplikacji
const appState = {
    errorCounts: {},    // Liczniki błędów dla różnych endpointów
    timers: {},         // Identyfikatory setTimeout dla różnych odświeżeń
    activeDashboard: true, // Czy dashboard jest aktywną zakładką
    lastApiCall: {},     // Czas ostatniego wywołania dla każdego API
    activeRequests: 0,   // Liczba aktywnych żądań
    retryTimeouts: {},   // Timeouty dla ponownych prób
    rateLimited: false,  // Czy API jest obecnie ograniczone
    rateLimitResetTime: 0 // Czas resetowania ograniczenia
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

// Funkcje API - z obsługą rate limiting i retry
async function fetchWithRateLimit(url, options = {}) {
    const endpoint = url.split('?')[0]; // Podstawowy endpoint bez parametrów

    // Sprawdzenie czy API jest obecnie ograniczone (rate limited)
    if (appState.rateLimited) {
        const now = Date.now();
        if (now < appState.rateLimitResetTime) {
            const waitTime = (appState.rateLimitResetTime - now) / 1000;
            console.log(`API jest obecnie ograniczone. Spróbuj ponownie za ${waitTime.toFixed(0)} sekund.`);
            showNotification('warning', `Limit API przekroczony. Odczekaj ${waitTime.toFixed(0)} sekund.`);
            throw new Error(`Rate limit - odczekaj ${waitTime.toFixed(0)}s`);
        } else {
            // Reset stanu ograniczenia
            appState.rateLimited = false;
        }
    }

    // Sprawdzenie czasu od ostatniego wywołania tego endpointu
    const now = Date.now();
    const lastCall = appState.lastApiCall[endpoint] || 0;
    const timeSinceLastCall = now - lastCall;
    const lastGlobalCall = appState.lastApiCall['global'] || 0;
    const timeSinceLastGlobalCall = now - lastGlobalCall;

    // Określenie minimalnego czasu między wywołaniami dla danego endpointu
    let minTimeBetweenCalls = CONFIG.minTimeBetweenCalls.api;
    if (endpoint.includes('portfolio')) {
        minTimeBetweenCalls = CONFIG.minTimeBetweenCalls.portfolio;
    } else if (endpoint.includes('dashboard')) {
        minTimeBetweenCalls = CONFIG.minTimeBetweenCalls.dashboard;
    }

    // Czekaj na minimum czasu między wywołaniami tego samego endpointu
    if (timeSinceLastCall < minTimeBetweenCalls) {
        const timeToWait = minTimeBetweenCalls - timeSinceLastCall;
        await new Promise(resolve => setTimeout(resolve, timeToWait));
    }

    // Minimum czasu między dowolnymi wywołaniami API
    if (timeSinceLastGlobalCall < CONFIG.minTimeBetweenCalls.api) {
        const timeToWait = CONFIG.minTimeBetweenCalls.api - timeSinceLastGlobalCall;
        await new Promise(resolve => setTimeout(resolve, timeToWait));
    }

    // Zwiększ licznik aktywnych żądań
    appState.activeRequests++;
    
    try {
        // Aktualizacja czasu ostatniego wywołania
        appState.lastApiCall[endpoint] = Date.now();
        appState.lastApiCall['global'] = Date.now();

        // Wykonanie zapytania
        const response = await fetch(url, options);
        
        // Sprawdź czy mamy problemy z limitami API
        if (response.status === 403 || response.status === 429) {
            // Ustawienie stanu ograniczenia
            appState.rateLimited = true;
            appState.rateLimitResetTime = Date.now() + (3 * 60 * 1000); // 3 minuty
            console.warn(`Przekroczono limit API (${response.status}). Czekaj 3 minuty.`);
            showNotification('error', 'Przekroczono limit zapytań API. Poczekaj 3 minuty.');
            throw new Error(`HTTP error ${response.status} - Rate limit exceeded`);
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        // Zwiększ licznik błędów dla endpointu
        appState.errorCounts[endpoint] = (appState.errorCounts[endpoint] || 0) + 1;
        console.error(`Błąd podczas pobierania danych z ${endpoint}:`, error);
        
        // Sprawdź, czy to problem z ograniczeniem API
        if (error.message && (error.message.includes('403') || error.message.includes('429') || error.message.includes('rate limit'))) {
            appState.rateLimited = true;
            appState.rateLimitResetTime = Date.now() + (3 * 60 * 1000); // 3 minuty
            showNotification('error', 'Przekroczono limit zapytań API. Poczekaj 3 minuty.');
        }
        
        throw error;
    } finally {
        // Zmniejsz licznik aktywnych żądań
        appState.activeRequests--;
    }
}

// Aktualizacja danych dashboardu z obsługą retry
async function updateDashboardData() {
    console.log("Aktualizacja danych dashboardu...");
    try {
        // Pobieranie danych dashboardu
        const dashboardData = await fetchWithRetry('/api/dashboard/data');

        // Aktualizacja każdej sekcji (ale z fallback, jeśli niektóre zapytania się nie powiodą)
        try { updateTradingStats(); } catch (e) { console.error("Nie udało się zaktualizować statystyk trading:", e); }
        try { updateAlerts(); } catch (e) { console.error("Nie udało się zaktualizować alertów:", e); }
        try { updateRecentTrades(); } catch (e) { console.error("Nie udało się zaktualizować transakcji:", e); }
        try { updateNotifications(); } catch (e) { console.error("Nie udało się zaktualizować powiadomień:", e); }
        try { updateChartData(); } catch (e) { console.error("Nie udało się zaktualizować wykresu:", e); }
        try { updateComponentStatuses(); } catch (e) { console.error("Nie udało się zaktualizować statusów komponentów:", e); }
        try { fetchPortfolioData(); } catch (e) { console.error("Nie udało się pobrać danych portfolio:", e); }

    } catch (error) {
        handleApiError('dashboard_data');
        console.error('Błąd podczas aktualizacji danych dashboardu:', error);
    }
}

// Funkcja do ponawiania prób z opóźnieniem wykładniczym
async function fetchWithRetry(url, options = {}, retryCount = 0) {
    try {
        return await fetchWithRateLimit(url, options);
    } catch (error) {
        // Sprawdź czy mamy problem z rate limit
        if (error.message && (error.message.includes('403') || error.message.includes('429') || error.message.includes('rate limit'))) {
            // Dla problemów z rate limit, czekamy dłużej
            if (retryCount < CONFIG.retry.maxRetries) {
                const delayTime = CONFIG.retry.delayMs * Math.pow(CONFIG.retry.backoffMultiplier, retryCount);
                console.log(`Limit API przekroczony, ponawiam próbę ${retryCount + 1}/${CONFIG.retry.maxRetries} za ${delayTime/1000}s`);
                await new Promise(resolve => setTimeout(resolve, delayTime));
                return fetchWithRetry(url, options, retryCount + 1);
            }
        } else if (retryCount < CONFIG.retry.maxRetries) {
            // Dla innych błędów, standardowy retry
            const delayTime = CONFIG.retry.delayMs * Math.pow(CONFIG.retry.backoffMultiplier, retryCount);
            console.log(`Błąd API, ponawiam próbę ${retryCount + 1}/${CONFIG.retry.maxRetries} za ${delayTime/1000}s`);
            await new Promise(resolve => setTimeout(resolve, delayTime));
            return fetchWithRetry(url, options, retryCount + 1);
        }
        
        // Jeśli osiągnęliśmy maksymalną liczbę prób, rzucamy wyjątek
        throw error;
    }
}

// Aktualizacja danych portfela z retry
async function fetchPortfolioData() {
    try {
        const data = await fetchWithRetry(`/api/portfolio?_=${Date.now()}`);
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

// Aktualizacja ostatnich transakcji z retry
async function updateRecentTrades() {
    try {
        const data = await fetchWithRetry('/api/recent-trades');
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

// Aktualizacja statystyk tradingowych z retry
async function updateTradingStats() {
    try {
        const data = await fetchWithRetry('/api/trading-stats');
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

// Aktualizacja alertów z retry
async function updateAlerts() {
    try {
        const data = await fetchWithRetry('/api/alerts');
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

// Aktualizacja powiadomień z retry
async function updateNotifications() {
    try {
        const data = await fetchWithRetry('/api/notifications');
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

// Aktualizacja statusów komponentów z retry
async function updateComponentStatuses() {
    console.log("Aktualizacja statusów komponentów...");
    try {
        const data = await fetchWithRetry('/api/component-status');
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
            
            // Dodanie informacji o statusie API, jeśli jest dostępny
            const apiComponent = document.getElementById('api-connector');
            if (apiComponent && appState.rateLimited) {
                apiComponent.classList.remove('status-online');
                apiComponent.classList.add('status-warning');
                const statusText = apiComponent.querySelector('.status-text');
                if (statusText) {
                    statusText.textContent = 'Rate Limited';
                }
            }
        }
    } catch (error) {
        handleApiError('components');
        console.error('Błąd podczas aktualizacji statusów komponentów:', error);
    }
}

// Aktualizacja wykresu głównego z retry
async function updateChartData() {
    try {
        const data = await fetchWithRetry('/api/chart-data');

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
// Z obsługą debounce i throttle dla lepszego zarządzania zapytaniami
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Throttled funkcje obsługi przycisków
const startTradingThrottled = throttle(function() {
    if (appState.rateLimited) {
        showNotification('warning', 'API jest obecnie ograniczone. Spróbuj ponownie później.');
        return;
    }
    
    fetch('/api/trading/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.status === 403 || response.status === 429) {
            appState.rateLimited = true;
            appState.rateLimitResetTime = Date.now() + (3 * 60 * 1000); // 3 minuty
            throw new Error('Przekroczono limit API');
        }
        return response.json();
    })
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
        if (error.message.includes('limit API')) {
            showNotification('error', 'Przekroczono limit API. Spróbuj ponownie za kilka minut.');
        } else {
            showNotification('error', 'Nie udało się uruchomić tradingu automatycznego');
        }
    });
}, 5000);

const stopTradingThrottled = throttle(function() {
    if (appState.rateLimited) {
        showNotification('warning', 'API jest obecnie ograniczone. Spróbuj ponownie później.');
        return;
    }
    
    fetch('/api/trading/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.status === 403 || response.status === 429) {
            appState.rateLimited = true;
            appState.rateLimitResetTime = Date.now() + (3 * 60 * 1000); // 3 minuty
            throw new Error('Przekroczono limit API');
        }
        return response.json();
    })
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
        if (error.message.includes('limit API')) {
            showNotification('error', 'Przekroczono limit API. Spróbuj ponownie za kilka minut.');
        } else {
            showNotification('error', 'Nie udało się zatrzymać tradingu automatycznego');
        }
    });
}, 5000);

const resetSystemThrottled = throttle(function() {
    if (appState.rateLimited) {
        showNotification('warning', 'API jest obecnie ograniczone. Spróbuj ponownie później.');
        return;
    }
    
    if (confirm('Czy na pewno chcesz zresetować system? Wszystkie aktywne operacje zostaną zakończone.')) {
        fetch('/api/system/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (response.status === 403 || response.status === 429) {
                appState.rateLimited = true;
                appState.rateLimitResetTime = Date.now() + (3 * 60 * 1000); // 3 minuty
                throw new Error('Przekroczono limit API');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showNotification('success', 'System został zresetowany');
                // Odświeżenie wszystkich danych z opóźnieniem aby dać czas na restart
                setTimeout(() => {
                    updateDashboardData();
                }, 2000);
            } else {
                showNotification('error', `Błąd: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Błąd podczas resetowania systemu:', error);
            if (error.message.includes('limit API')) {
                showNotification('error', 'Przekroczono limit API. Spróbuj ponownie za kilka minut.');
            } else {
                showNotification('error', 'Nie udało się zresetować systemu');
            }
        });
    }
}, 5000);

// Setup Event Listeners
function setupEventListeners() {
    // Trading controls
    const startTradingBtn = document.getElementById('start-trading-btn');
    if (startTradingBtn) {
        startTradingBtn.addEventListener('click', startTradingThrottled);
    }

    const stopTradingBtn = document.getElementById('stop-trading-btn');
    if (stopTradingBtn) {
        stopTradingBtn.addEventListener('click', stopTradingThrottled);
    }

    const resetSystemBtn = document.getElementById('reset-system-btn');
    if (resetSystemBtn) {
        resetSystemBtn.addEventListener('click', resetSystemThrottled);
    }

    // Visibility change (to pause updates when tab is not visible)
    document.addEventListener('visibilitychange', function() {
        appState.activeDashboard = !document.hidden;
        
        // Jeśli dashboard jest znowu widoczny, odśwież dane po krótkim czasie
        if (appState.activeDashboard) {
            setTimeout(() => {
                updateDashboardData();
            }, 1000);
        }
    });

    // Dodanie obsługi retry dla błędów fetch
    window.addEventListener('offline', function() {
        showNotification('error', 'Utracono połączenie z internetem');
    });

    window.addEventListener('online', function() {
        showNotification('success', 'Połączenie z internetem przywrócone');
        // Odśwież dane po przywróceniu połączenia
        setTimeout(() => {
            updateDashboardData();
        }, 1000);
    });

    // Event listener do resetowania stanu rate limitu po kliknięciu w przycisk
    const retryBtn = document.getElementById('retry-api-btn');
    if (retryBtn) {
        retryBtn.addEventListener('click', function() {
            if (appState.rateLimited) {
                const now = Date.now();
                if (now < appState.rateLimitResetTime) {
                    const waitTime = (appState.rateLimitResetTime - now) / 1000;
                    showNotification('warning', `Wciąż trzeba poczekać ${waitTime.toFixed(0)} sekund.`);
                } else {
                    appState.rateLimited = false;
                    showNotification('success', 'Reset stanu rate limit. Próbuję pobrać dane.');
                    updateDashboardData();
                }
            } else {
                showNotification('info', 'API nie jest ograniczone. Odświeżam dane.');
                updateDashboardData();
            }
        });
    }
}

// Obsługa błędów API
function handleApiError(endpoint) {
    // Zwiększ licznik błędów dla danego endpointu
    appState.errorCounts[endpoint] = (appState.errorCounts[endpoint] || 0) + 1;

    // Jeśli przekroczono limit błędów, pokaż komunikat
    if (appState.errorCounts[endpoint] >= CONFIG.maxErrors) {
        showErrorMessage(`Zbyt wiele błędów podczas komunikacji z API (${endpoint}). Sprawdź logi.`);
        
        // Jeśli mamy błędy 403/429, pokaż info o rate limit
        if (appState.rateLimited) {
            const now = Date.now();
            const waitTime = (appState.rateLimitResetTime - now) / 1000;
            if (waitTime > 0) {
                showErrorMessage(`Przekroczono limit zapytań API. Spróbuj ponownie za ${waitTime.toFixed(0)} sekund.`);
            }
        }
    }
}

// Wyświetlanie komunikatu o błędzie
function showErrorMessage(message) {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.innerHTML = `<div class="error-message">${message}</div>`;
        errorContainer.style.display = 'block';
        
        // Dodaj przycisk retry jeśli problem z rate limit
        if (message.includes('limitu zapytań API') || message.includes('limit zapytań')) {
            errorContainer.innerHTML += `<button id="retry-api-btn" class="retry-button">Spróbuj ponownie</button>`;
            const retryBtn = document.getElementById('retry-api-btn');
            if (retryBtn) {
                retryBtn.addEventListener('click', function() {
                    if (appState.rateLimited) {
                        const now = Date.now();
                        if (now < appState.rateLimitResetTime) {
                            const waitTime = (appState.rateLimitResetTime - now) / 1000;
                            showNotification('warning', `Wciąż trzeba poczekać ${waitTime.toFixed(0)} sekund.`);
                        } else {
                            appState.rateLimited = false;
                            errorContainer.style.display = 'none';
                            showNotification('success', 'Reset stanu rate limit. Próbuję pobrać dane.');
                            updateDashboardData();
                        }
                    } else {
                        errorContainer.style.display = 'none';
                        showNotification('info', 'Odświeżam dane.');
                        updateDashboardData();
                    }
                });
            }
        }
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
