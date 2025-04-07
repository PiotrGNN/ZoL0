// Funkcje dla dashboardu tradingowego
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard załadowany');

    // Inicjalizacja
    updateDashboardData();
    updateComponentStatuses();
    fetchNotifications();
    setupEventListeners();
    initializeChart();

    // Interwały aktualizacji
    setInterval(updateDashboardData, 10000); // Co 10 sekund
    setInterval(updateComponentStatuses, 15000); // Co 15 sekund
    setInterval(fetchNotifications, 60000); // Co minutę
});

// Liczniki błędów dla obsługi ponownych prób
let errorCounts = {
    chart: 0,
    status: 0,
    stats: 0,
    portfolio: 0
};

// Główne funkcje dashboardu
function updateDashboardData() {
    console.log('Aktualizacja danych dashboardu...');
    updatePortfolioData();
    updateCharts();
    updateTradingStats();
    updateRecentTrades();
    updateAIModelsStatus(); //Retained from original
    updateAlerts(); //Retained from original
}

function updatePortfolioData() {
    fetch('/api/bybit/account-balance')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Aktualizacja interfejsu z danymi portfela
            const portfolioContainer = document.getElementById('portfolio-container');
            if (portfolioContainer && data.balances) {
                let portfolioHTML = '<div class="card"><div class="card-header">Stan portfela</div><div class="card-body">';
                portfolioHTML += '<table class="table"><thead><tr><th>Waluta</th><th>Dostępne</th><th>Całkowite</th></tr></thead><tbody>';

                for (const [currency, balance] of Object.entries(data.balances)) {
                    portfolioHTML += `<tr>
                        <td>${currency}</td>
                        <td>${Number(balance.available_balance).toFixed(6)}</td>
                        <td>${Number(balance.wallet_balance).toFixed(6)}</td>
                    </tr>`;
                }

                portfolioHTML += '</tbody></table></div></div>';
                portfolioContainer.innerHTML = portfolioHTML;
            }

            // Resetuj licznik błędów po sukcesie
            errorCounts.portfolio = 0;
        })
        .catch(error => {
            errorCounts.portfolio++;
            console.warn("Błąd podczas pobierania danych portfela:", error);

            // Tylko wyświetl komunikat o błędzie jeśli jest to powtarzający się problem
            if (errorCounts.portfolio > 3) {
                const portfolioContainer = document.getElementById('portfolio-container');
                if (portfolioContainer) {
                    portfolioContainer.innerHTML = `<div class="card">
                        <div class="card-header">Stan portfela</div>
                        <div class="card-body">
                            <div class="alert alert-warning">
                                Nie udało się pobrać danych portfela. Spróbuj odświeżyć stronę.
                            </div>
                        </div>
                    </div>`;
                }
            }
        });
}

function updateCharts() {
    fetch('/api/chart-data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                const ctx = document.getElementById('portfolio-chart');
                if (ctx) {
                    renderChart(ctx, data.data);
                }
                errorCounts.chart = 0;
            } else {
                throw new Error(data.error || 'Błąd pobierania danych wykresu');
            }
        })
        .catch(error => {
            errorCounts.chart++;
            console.error('Błąd podczas aktualizacji wykresu:', error);
            if (errorCounts.chart > 3) {
                showErrorMessage('Nie udało się załadować wykresu. Spróbuj odświeżyć stronę.');
            }
        });
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

function initializeChart() {
    const ctx = document.getElementById('portfolio-chart');
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

function updateComponentStatuses() {
    console.log('Aktualizacja statusów komponentów...');
    fetch('/api/system/status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                updateStatusIndicators(data.components);
                errorCounts.status = 0;
            } else {
                throw new Error(data.error || 'Błąd pobierania statusu systemu');
            }
        })
        .catch(error => {
            errorCounts.status++;
            console.error('Błąd podczas aktualizacji statusów komponentów:', error);
            if (errorCounts.status > 3) {
                showErrorMessage('Nie udało się zaktualizować statusu systemu.');
            }
        });
}

function updateStatusIndicators(components) {
    for (const [component, status] of Object.entries(components)) {
        const indicator = document.getElementById(`${component}-status`);
        if (indicator) {
            // Usuń wszystkie klasy statusu
            indicator.classList.remove('status-online', 'status-warning', 'status-offline', 'status-maintenance');

            // Dodaj odpowiednią klasę
            if (status === 'active') {
                indicator.classList.add('status-online');
                indicator.title = 'Aktywny';
            } else if (status === 'warning') {
                indicator.classList.add('status-warning');
                indicator.title = 'Ostrzeżenie';
            } else if (status === 'inactive') {
                indicator.classList.add('status-offline');
                indicator.title = 'Nieaktywny';
            } else {
                indicator.classList.add('status-maintenance');
                indicator.title = 'Konserwacja';
            }
        }
    }
}

function updateTradingStats() {
    fetch('/api/trading-stats')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Aktualizuj statystyki
            document.getElementById('profit-value').textContent = data.profit || 'N/A';
            document.getElementById('trades-count').textContent = data.trades_count || 'N/A';
            document.getElementById('win-rate').textContent = data.win_rate || 'N/A';
            document.getElementById('max-drawdown').textContent = data.max_drawdown || 'N/A';

            errorCounts.stats = 0;
        })
        .catch(error => {
            errorCounts.stats++;
            console.error('Błąd podczas aktualizacji statystyk tradingowych:', error);
            if (errorCounts.stats > 3) {
                // Ustaw wartości domyślne
                document.getElementById('profit-value').textContent = 'N/A';
                document.getElementById('trades-count').textContent = 'N/A';
                document.getElementById('win-rate').textContent = 'N/A';
                document.getElementById('max-drawdown').textContent = 'N/A';
            }
        });
}

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
            if (tradesContainer) {
                if (data.trades && data.trades.length > 0) {
                    let tradesHTML = '<table class="table"><thead><tr><th>Symbol</th><th>Typ</th><th>Czas</th><th>Zysk</th></tr></thead><tbody>';

                    data.trades.forEach(trade => {
                        const profitClass = trade.profit >= 0 ? 'positive' : 'negative';
                        tradesHTML += `<tr>
                            <td>${trade.symbol}</td>
                            <td>${trade.type}</td>
                            <td>${trade.time}</td>
                            <td class="${profitClass}">${trade.profit}%</td>
                        </tr>`;
                    });

                    tradesHTML += '</tbody></table>';
                    tradesContainer.innerHTML = tradesHTML;
                } else {
                    tradesContainer.innerHTML = '<p class="text-center">Brak ostatnich transakcji</p>';
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania ostatnich transakcji:', error);
            const tradesContainer = document.getElementById('recent-trades-container');
            if (tradesContainer) {
                tradesContainer.innerHTML = '<p class="text-center">Nie udało się pobrać ostatnich transakcji</p>';
            }
        });
}

function fetchNotifications() {
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
                    if (data.notifications.length > 0) {
                        let notificationsHTML = '';

                        data.notifications.forEach(notification => {
                            let typeClass = 'info';
                            if (notification.type === 'warning') typeClass = 'warning';
                            if (notification.type === 'error') typeClass = 'danger';
                            if (notification.type === 'success') typeClass = 'success';

                            notificationsHTML += `<div class="alert alert-${typeClass}">
                                <span class="time">${notification.timestamp}</span>
                                <span class="message">${notification.message}</span>
                            </div>`;
                        });

                        notificationsContainer.innerHTML = notificationsHTML;
                    } else {
                        notificationsContainer.innerHTML = '<p class="text-center">Brak nowych powiadomień</p>';
                    }
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania powiadomień:', error);
        });
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
                const tabId = this.dataset.tab;
                switchTab(tabId);
            });
        });
    }
}

function switchTab(tabId) {
    // Ukryj wszystkie taby
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });

    // Usuń aktywną klasę z przycisków
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });

    // Pokaż wybrany tab
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.style.display = 'block';
    }

    // Dodaj aktywną klasę do przycisku
    const activeButton = document.querySelector(`.tab-button[data-tab="${tabId}"]`);
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

function showErrorMessage(message) {
    // Pokaż komunikat o błędzie
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.innerHTML = `<div class="error-message">${message}</div>`;
        errorContainer.style.display = 'block';
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
    if (confirm('Czy na pewno chcesz zresetować system? Ta operacja zatrzyma wszystkie aktywne procesy.')) {
        fetch('/api/system/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('System został zresetowany!');
                updateComponentStatuses();
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

function updateAIModelsStatus() {
    fetch('/api/ai-models-status')
        .then(response => {
            if (!response.ok) {
                return { models: [] }; // Zwracamy pusty obiekt jeśli API nie istnieje
            }
            return response.json();
        })
        .then(data => {
            const aiModelsContainer = document.getElementById('ai-models-container');
            if (!aiModelsContainer) return;

            if (data && Array.isArray(data.models) && data.models.length > 0) {
                let modelsHTML = '';

                data.models.forEach(model => {
                    const accuracyClass = getAccuracyClass(model.accuracy);
                    modelsHTML += `
                    <div class="ai-model-card">
                        <h4>${model.name}</h4>
                        <div class="model-details">
                            <div>Typ: ${model.type}</div>
                            <div>Dokładność: <span class="${accuracyClass}">${model.accuracy}%</span></div>
                            <div>Status: <span class="status-${model.status.toLowerCase()}">${model.status}</span></div>
                            <div>Ostatnie użycie: ${model.last_used || 'Nigdy'}</div>
                        </div>
                    </div>`;
                });

                aiModelsContainer.innerHTML = modelsHTML;
                document.getElementById('ai-models-section').style.display = 'block';
            } else {
                // Brak modeli lub API nie zwraca poprawnych danych
                document.getElementById('ai-models-section').style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusów modeli AI:', error);
            document.getElementById('ai-models-section').style.display = 'none';
        });
}

function updateAlerts() {
    // Pobierz dane z API
    fetch('/api/alerts')
        .then(response => {
            if (!response.ok) {
                return { alerts: [] }; // Pusta lista
            }
            return response.json();
        })
        .then(data => {
            const alertsContainer = document.getElementById('alerts-list');
            if (!alertsContainer) return;

            if (data && Array.isArray(data.alerts) && data.alerts.length > 0) {
                let alertsHTML = '';

                data.alerts.forEach(alert => {
                    const alertClass = {
                        'critical': 'status-offline',
                        'warning': 'status-warning',
                        'info': 'status-online'
                    }[alert.level] || 'status-maintenance';

                    alertsHTML += `
                    <div class="alert-item ${alertClass}">
                        <div class="alert-time">${alert.time}</div>
                        <div class="alert-message">${alert.message}</div>
                    </div>`;
                });

                alertsContainer.innerHTML = alertsHTML;

                // Aktualizacja licznika alertów
                const alertBadge = document.getElementById('alerts-badge');
                if (alertBadge) {
                    alertBadge.textContent = data.alerts.length;
                    alertBadge.style.display = data.alerts.length > 0 ? 'inline-block' : 'none';
                }

            } else {
                alertsContainer.innerHTML = '<div class="no-data">Brak alertów</div>';

                // Ukryj badge jeśli nie ma alertów
                const alertBadge = document.getElementById('alerts-badge');
                if (alertBadge) {
                    alertBadge.style.display = 'none';
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania alertów:', error);
            const alertsContainer = document.getElementById('alerts-list');
            if (alertsContainer) {
                alertsContainer.innerHTML = '<div class="no-data">Brak alertów</div>';
            }
        });
}

function updateComponentStatus(componentId, status) {
    const component = document.getElementById(componentId);
    if (component) {
        // Usuń poprzednie klasy statusu
        component.classList.remove('status-online', 'status-warning', 'status-offline', 'status-maintenance');

        // Dodaj nową klasę statusu
        component.classList.add(`status-${status}`);

        // Aktualizuj tekst statusu
        const statusElement = component.querySelector('.status-text');
        if (statusElement) {
            statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }
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