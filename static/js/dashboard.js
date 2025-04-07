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

// Licznik błędów dla każdego typu danych
const errorCounts = {
    'chart': 0,
    'balance': 0,
    'tradingStats': 0,
    'recentTrades': 0
};

// Główne funkcje dashboardu
function updateDashboardData() {
    console.log('Aktualizacja danych dashboardu...');
    updatePortfolio(); // Zmienione na nową funkcję
    updateCharts();
    updateTradingStats();
    updateRecentTrades();
    updateAIModelsStatus(); 
    updateAlerts(); 
}

// Licznik błędów dla śledzenia problemów z API
let apiErrorCount = 0;
const MAX_ERROR_COUNT = 5;

function updatePortfolio() {
    fetch('/api/bybit/account-balance')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Resetujemy licznik błędów, gdy żądanie zakończy się sukcesem
            apiErrorCount = 0;

            const portfolioContainer = document.getElementById('portfolio-data');
            if (!portfolioContainer) return;

            // Wyczyszczenie obecnej zawartości
            portfolioContainer.innerHTML = '';

            // Sprawdzanie czy mamy błąd API
            if (data.error) {
                console.error("Błąd podczas pobierania danych portfela:", data.error);
                portfolioContainer.innerHTML = `
                    <div class="error-message">
                        <h4>Problem z API ByBit</h4>
                        <p>${data.error}</p>
                        <p>Sprawdź swoje klucze API i uprawnienia.</p>
                    </div>`;
                return;
            }

            if (data.balances && Object.keys(data.balances).length > 0) {
                let totalUSDT = 0;

                // Tworzenie tabeli z danymi portfela
                const table = document.createElement('table');
                table.className = 'portfolio-table';

                // Nagłówek tabeli
                const thead = document.createElement('thead');
                thead.innerHTML = `
                    <tr>
                        <th>Waluta</th>
                        <th>Wartość</th>
                        <th>Dostępne</th>
                        <th>W portfelu</th>
                    </tr>
                `;
                table.appendChild(thead);

                // Ciało tabeli
                const tbody = document.createElement('tbody');

                let hasAssets = false;

                for (const [coin, balance] of Object.entries(data.balances)) {
                    if (balance.equity > 0) {
                        hasAssets = true;
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${coin}</td>
                            <td>${parseFloat(balance.equity).toFixed(6)}</td>
                            <td>${parseFloat(balance.available_balance).toFixed(6)}</td>
                            <td>${parseFloat(balance.wallet_balance).toFixed(6)}</td>
                        `;
                        tbody.appendChild(row);

                        // Dodawanie do sumy USDT (wartość szacunkowa)
                        if (coin === 'USDT') {
                            totalUSDT += parseFloat(balance.equity);
                        } else {
                            // Tutaj można dodać przeliczanie innych walut na USDT
                        }
                    }
                }

                if (!hasAssets) {
                    portfolioContainer.innerHTML = `
                        <div class="warning-message">
                            <h4>Brak aktywów na koncie</h4>
                            <p>Twoje konto ByBit nie zawiera żadnych środków lub dane nie mogły zostać pobrane.</p>
                        </div>`;

                    // Dodaj notatkę jeśli dane są symulowane
                    if (data.note) {
                        const noteElement = document.createElement('div');
                        noteElement.className = 'data-note';
                        noteElement.textContent = data.note;
                        portfolioContainer.appendChild(noteElement);
                    }

                    return;
                }

                table.appendChild(tbody);
                portfolioContainer.appendChild(table);

                // Dodawanie informacji o szacunkowej całkowitej wartości
                const totalValueInfo = document.createElement('div');
                totalValueInfo.className = 'total-value-info';
                totalValueInfo.innerHTML = `Szacunkowa wartość (USDT): <strong>${totalUSDT.toFixed(2)}</strong>`;
                portfolioContainer.appendChild(totalValueInfo);

                // Status API
                const apiStatus = document.createElement('div');
                apiStatus.className = data.success ? 'api-status success' : 'api-status warning';
                apiStatus.innerHTML = `<strong>Status API:</strong> ${data.success ? 'Połączono' : 'Problem z połączeniem'}`;
                if (data.source) {
                    apiStatus.innerHTML += ` (źródło: ${data.source})`;
                }
                portfolioContainer.appendChild(apiStatus);

                // Dodaj notatkę jeśli dane są symulowane
                if (data.note) {
                    const noteElement = document.createElement('div');
                    noteElement.className = 'data-note';
                    noteElement.textContent = data.note;
                    portfolioContainer.appendChild(noteElement);
                }

                // Dodaj ostrzeżenie jeśli jest
                if (data.warning) {
                    const warningElement = document.createElement('div');
                    warningElement.className = 'warning-message';
                    warningElement.textContent = data.warning;
                    portfolioContainer.appendChild(warningElement);
                }
            } else {
                portfolioContainer.innerHTML = `
                    <div class="error-message">
                        <h4>Brak danych portfela</h4>
                        <p>Problem z połączeniem z ByBit lub konto nie zawiera żadnych środków.</p>
                        <p>Upewnij się, że klucze API są poprawne i mają wymagane uprawnienia.</p>
                    </div>`;

                // Dodaj notatkę jeśli dane są symulowane
                if (data.note) {
                    const noteElement = document.createElement('div');
                    noteElement.className = 'data-note';
                    noteElement.textContent = data.note;
                    portfolioContainer.appendChild(noteElement);
                }
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania danych portfela:", error);

            // Zwiększamy licznik błędów
            apiErrorCount++;

            const portfolioContainer = document.getElementById('portfolio-data');
            if (portfolioContainer) {
                if (apiErrorCount >= MAX_ERROR_COUNT) {
                    // Gdy osiągniemy limit błędów, wyświetlamy bardziej szczegółową informację
                    portfolioContainer.innerHTML = `
                        <div class="error-message">
                            <h4>Poważny problem z połączeniem API</h4>
                            <p>Nie można połączyć się z API ByBit po ${apiErrorCount} próbach.</p>
                            <p>Powody:</p>
                            <ul>
                                <li>Nieprawidłowe klucze API</li>
                                <li>Wygasłe klucze API</li>
                                <li>Brak wymaganych uprawnień API</li>
                                <li>Problem z połączeniem internetowym</li>
                                <li>API ByBit może być chwilowo niedostępne</li>
                            </ul>
                            <p>Błąd: ${error.message}</p>
                        </div>`;
                } else {
                    portfolioContainer.innerHTML = `
                        <div class="error-message">
                            <h4>Błąd podczas pobierania danych portfela</h4>
                            <p>Sprawdź połączenie z internetem i API ByBit.</p>
                            <p>Próba ${apiErrorCount} z ${MAX_ERROR_COUNT}</p>
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

            errorCounts.tradingStats = 0;
        })
        .catch(error => {
            errorCounts.tradingStats++;
            console.error('Błąd podczas aktualizacji statystyk tradingowych:', error);
            if (errorCounts.tradingStats > 3) {
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