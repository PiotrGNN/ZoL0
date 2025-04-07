// Funkcje dla dashboardu tradingowego
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard załadowany');

    // Inicjalizacja
    updateDashboardData();
    updateComponentStatuses();
    fetchNotifications();
    setupEventListeners();
    initializeChart(); // Added chart initialization

    // Interwały aktualizacji - poprawione interwały
    setInterval(updateDashboardData, 5000); // Co 5 sekund
    setInterval(updateComponentStatuses, 10000); // Co 10 sekund
    setInterval(fetchNotifications, 60000); // Co minutę
});

// Liczniki błędów dla obsługi ponownych prób
let chartErrorCount = 0;
let statusErrorCount = 0;
let statsErrorCount = 0;

// Główne funkcje dashboardu
function updateDashboardData() {
    console.log('Aktualizacja danych dashboardu...');
    updateCharts();
    updateTradingStats();

function updatePortfolio() {
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            const portfolioContainer = document.getElementById('portfolio-container');
            if (!portfolioContainer) return;
            
            if (data.error) {
                portfolioContainer.innerHTML = `<div class="no-data">Błąd: ${data.error}</div>`;
                return;
            }
            
            if (!data.coins || data.coins.length === 0) {
                portfolioContainer.innerHTML = '<div class="no-data">Brak danych portfela.</div>';
                return;
            }
            
            let html = '';
            data.coins.forEach(coin => {
                html += `
                <div class="portfolio-item">
                    <div class="coin-name">${coin.coin}</div>
                    <div class="coin-balance">Balans: ${coin.walletBalance}</div>
                    <div class="coin-value">Wartość: ${coin.usdValue || 'N/A'}</div>
                </div>`;
            });
            
            portfolioContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych portfela:', error);
            const portfolioContainer = document.getElementById('portfolio-container');
            if (portfolioContainer) {
                portfolioContainer.innerHTML = `<div class="no-data">Błąd połączenia. Spróbuj ponownie.</div>`;
            }
        });
}

    updateRecentTrades();
    updateAlerts();
    updatePortfolio();
    updateAIModelsStatus();
}

function updateComponentStatuses() {
    console.log('Aktualizacja statusów komponentów...');

    fetch('/api/component-status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania statusów komponentów');
            }
            return response.json();
        })
        .then(data => {
            statusErrorCount = 0; // Reset licznika błędów

            if (data && Array.isArray(data.components)) {
                data.components.forEach(component => {
                    updateComponentStatus(component.id, component.status);
                });
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusów komponentów:', error);
            statusErrorCount++;

            if (statusErrorCount <= 3) {
                // Spróbuj ponownie po 5 sekundach, ale tylko 3 razy
                setTimeout(updateComponentStatuses, 5000);
            } else {
                // Po 3 próbach pokaż błąd na interfejsie
                showErrorMessage('Nie można połączyć się z serwerem. Sprawdź połączenie sieciowe.');
            }
        });
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

// Funkcje pomocnicze
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

function updateCharts() {
    // Pobierz dane do wykresów
    fetch('/api/dashboard/data')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania danych wykresu');
            }
            return response.json();
        })
        .then(data => {
            chartErrorCount = 0; // Reset licznika błędów
            renderMainChart(data);
            hideChartError();
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych wykresu:', error);
            chartErrorCount++;

            if (chartErrorCount > 3) {
                // Po 3 próbach pokaż błąd na wykresie
                showChartError();
            }
        });
}

function renderMainChart(data) {
    const ctx = document.getElementById('main-chart');
    if (!ctx) return;

    // Jeżeli wykres już istnieje, zniszcz go
    if (window.mainChart) {
        window.mainChart.destroy();
    }

    // Stwórz nowy wykres
    window.mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: data.datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

function showChartError() {
    const chartContainer = document.querySelector('.chart-container');
    if (!chartContainer) return;

    // Dodaj komunikat o błędzie
    if (!document.querySelector('.chart-error-message')) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message chart-error-message';
        errorDiv.innerHTML = `
            <p>Nie można załadować danych wykresu. Spróbuj ponownie później.</p>
            <button class="btn btn-primary retry-button" onclick="retryLoadChart()">Spróbuj ponownie</button>
        `;
        chartContainer.appendChild(errorDiv);
    }
}

function hideChartError() {
    const errorMessage = document.querySelector('.chart-error-message');
    if (errorMessage) {
        errorMessage.remove();
    }
}

function retryLoadChart() {
    hideChartError();
    initializeChart(); // Call initializeChart instead of updateCharts
}

function initializeChart() {
    updateCharts(); // Initialize the chart by calling updateCharts
}


function updateTradingStats() {
    fetch('/api/trading-stats')
        .then(response => {
            if (!response.ok) {
                // API może nie istnieć jeszcze, więc używamy symulowanych danych
                return {
                    profit: '$1,234.56',
                    trades_count: 42,
                    win_rate: '68%',
                    max_drawdown: '12.3%'
                };
            }
            return response.json();
        })
        .then(data => {
            statsErrorCount = 0;

            // Aktualizuj statystyki
            if (document.getElementById('profit-value')) {
                document.getElementById('profit-value').textContent = data.profit || '$0.00';
            }

            if (document.getElementById('trades-value')) {
                document.getElementById('trades-value').textContent = data.trades_count || '0';
            }

            if (document.getElementById('win-rate-value')) {
                document.getElementById('win-rate-value').textContent = data.win_rate || 'N/A';
            }

            if (document.getElementById('drawdown-value')) {
                document.getElementById('drawdown-value').textContent = data.max_drawdown || '0%';
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statystyk tradingowych:', error);
            statsErrorCount++;

            // Używamy symulowanych danych w przypadku błędu
            if (document.getElementById('profit-value')) {
                document.getElementById('profit-value').textContent = '$0.00';
            }

            if (document.getElementById('trades-value')) {
                document.getElementById('trades-value').textContent = '0';
            }

            if (document.getElementById('win-rate-value')) {
                document.getElementById('win-rate-value').textContent = 'N/A';
            }

            if (document.getElementById('drawdown-value')) {
                document.getElementById('drawdown-value').textContent = '0%';
            }
        });
}

function updateRecentTrades() {
    // Pobierz dane z API
    fetch('/api/recent-trades')
        .then(response => {
            if (!response.ok) {
                return { trades: [] }; // Pusta lista
            }
            return response.json();
        })
        .then(data => {
            const tradesContainer = document.getElementById('recent-trades-list');
            if (!tradesContainer) return;

            if (data && Array.isArray(data.trades) && data.trades.length > 0) {
                let tradesHTML = '';

                data.trades.forEach(trade => {
                    const profitClass = trade.profit >= 0 ? 'positive' : 'negative';

                    tradesHTML += `
                    <div class="trade-item">
                        <div class="trade-symbol">${trade.symbol}</div>
                        <div class="trade-type">${trade.type}</div>
                        <div class="trade-time">${trade.time}</div>
                        <div class="trade-profit ${profitClass}">${trade.profit >= 0 ? '+' : ''}${trade.profit}%</div>
                    </div>`;
                });

                tradesContainer.innerHTML = tradesHTML;
            } else {
                tradesContainer.innerHTML = '<div class="no-data">Brak transakcji</div>';
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania ostatnich transakcji:', error);
            const tradesContainer = document.getElementById('recent-trades-list');
            if (tradesContainer) {
                tradesContainer.innerHTML = '<div class="no-data">Brak transakcji</div>';
            }
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

function fetchNotifications() {
    // Pobieranie powiadomień z API
    fetch('/api/notifications')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const notifications = data.notifications;
                if (notifications.length > 0) {
                    const notificationsList = document.getElementById('notifications-list');
                    if (notificationsList) {
                        notificationsList.innerHTML = '';
                        let notificationsHTML = '';

                        notifications.forEach(notification => {
                            const notificationClass = {
                                'critical': 'status-offline',
                                'alert': 'status-warning',
                                'warning': 'status-warning',
                                'info': 'status-online'
                            }[notification.level] || '';

                            notificationsHTML += `
                            <div class="notification-item ${notificationClass}" data-id="${notification.id}">
                                <div class="notification-time">${formatTimestamp(notification.timestamp)}</div>
                                <div class="notification-content">
                                    <div class="notification-title">${notification.title}</div>
                                    <div class="notification-message">${notification.message}</div>
                                </div>
                                ${!notification.read ? '<div class="unread-indicator"></div>' : ''}
                            </div>`;
                        });

                        notificationsList.innerHTML = notificationsHTML;

                        // Aktualizacja licznika
                        const unreadCount = notifications.filter(n => !n.read).length;
                        const notificationBadge = document.getElementById('notifications-badge');
                        if (notificationBadge) {
                            notificationBadge.textContent = unreadCount;
                            notificationBadge.style.display = unreadCount > 0 ? 'inline-block' : 'none';
                        }

                        // Dodaj obsługę kliknięć
                        document.querySelectorAll('.notification-item').forEach(item => {
                            item.addEventListener('click', function() {
                                markNotificationAsRead(this.dataset.id);
                            });
                        });

                    }
                } else {
                    const notificationsList = document.getElementById('notifications-list');
                    if (notificationsList) {
                        notificationsList.innerHTML = '<div class="no-data">Brak powiadomień</div>';

                        // Ukryj badge
                        const notificationBadge = document.getElementById('notifications-badge');
                        if (notificationBadge) {
                            notificationBadge.style.display = 'none';
                        }
                    }
                }

                // Aktualizacja licznika powiadomień
                const badge = document.getElementById('notifications-badge');
                if (badge) {
                    badge.textContent = notifications.length;
                    badge.style.display = 'inline-block';
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania powiadomień:', error);
        });
}

function markNotificationAsRead(id) {
    fetch(`/api/notifications/${id}/read`, {
        method: 'POST'
    })
    .then(response => {
        if (response.ok) {
            // Odśwież listę powiadomień
            fetchNotifications();
        }
    })
    .catch(error => {
        console.error('Błąd oznaczania powiadomienia jako przeczytane:', error);
    });
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function getAccuracyClass(accuracy) {
    if (accuracy >= 70) return 'positive';
    if (accuracy >= 50) return 'neutral';
    return 'negative';
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
    if (confirm('Czy na pewno chcesz zresetować system? Wszystkie otwarte pozycje zostaną zamknięte.')) {
        fetch('/api/system/reset', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('System został zresetowany!');
                updateDashboardData();
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