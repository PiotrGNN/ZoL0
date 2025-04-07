// Funkcje dla dashboardu tradingowego
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard załadowany');

    // Inicjalizacja
    setupTabNavigation();
    updateDashboardData();
    updateComponentStatuses();
    fetchNotifications();
    setupEventListeners();
    initializeChart();
    updatePortfolio(); // Initialize portfolio data on load

    // Interwały aktualizacji
    setInterval(updateDashboardData, 5000); // Co 5 sekund
    setInterval(updateComponentStatuses, 10000); // Co 10 sekund
    setInterval(fetchNotifications, 60000); // Co minutę
    setInterval(updatePortfolio, 5000); // Update portfolio every 5 seconds

});

// Liczniki błędów dla obsługi ponownych prób
let chartErrorCount = 0;
let statusErrorCount = 0;
let statsErrorCount = 0;
let portfolioErrorCount = 0;


// Ustawienie nawigacji tabów
function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Usuń klasę active ze wszystkich przycisków
            tabButtons.forEach(btn => btn.classList.remove('active'));

            // Ukryj wszystkie zawartości tabów
            tabContents.forEach(content => content.style.display = 'none');

            // Dodaj klasę active do klikniętego przycisku
            button.classList.add('active');

            // Pokaż odpowiednią zawartość taba
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).style.display = 'block';
        });
    });
}

// Główne funkcje dashboardu
function updateDashboardData() {
    console.log('Aktualizacja danych dashboardu...');
    updateCharts();
    updateTradingStats();
    updateRecentTrades();
    updateAlerts();
    updateAIModels();
    updatePortfolio(); // Include portfolio update in main data refresh
}

function updateComponentStatuses() {
    console.log('Aktualizacja statusów komponentów...');

    fetch('/api/component-status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania statusu komponentów');
            }
            return response.json();
        })
        .then(data => {
            statusErrorCount = 0;

            if (data.success && data.components) {
                data.components.forEach(component => {
                    updateComponentStatus(component.id, component.status);
                });
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusów komponentów:', error);
            statusErrorCount++;

            if (statusErrorCount > 3) {
                // Po 3 próbach pokaż błąd na dashboardzie
                showStatusError('Nie można pobrać statusu komponentów.');
            }
        });
}

function updateTradingStats() {
    fetch('/api/trading-stats')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania statystyk handlowych');
            }
            return response.json();
        })
        .then(data => {
            statsErrorCount = 0;

            if (data.success && data.data) {
                const stats = data.data;
                document.getElementById('trades-value').textContent = stats.total_trades;
                document.getElementById('win-rate-value').textContent = `${Math.round(stats.win_rate * 100)}%`;
                document.getElementById('profit-value').textContent = `$${stats.daily_pnl.toFixed(2)}`;
                document.getElementById('drawdown-value').textContent = '5.2%'; // Przykładowa wartość
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statystyk handlowych:', error);
            statsErrorCount++;
        });
}

function updateRecentTrades() {
    fetch('/api/recent-trades')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania ostatnich transakcji');
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.data) {
                const tradesContainer = document.getElementById('recent-trades-list');
                tradesContainer.innerHTML = '';

                if (data.data.length === 0) {
                    tradesContainer.innerHTML = '<div class="no-data">Brak transakcji</div>';
                    return;
                }

                data.data.forEach(trade => {
                    const tradeItem = document.createElement('div');
                    tradeItem.className = 'trade-item';

                    // Tutaj możesz dostosować format danych odpowiednio do twojego API
                    tradeItem.innerHTML = `
                        <div class="trade-symbol">${trade.symbol}</div>
                        <div class="trade-type">${trade.side}</div>
                        <div class="trade-time">${trade.time}</div>
                        <div class="trade-profit positive">+1.2%</div>
                    `;

                    tradesContainer.appendChild(tradeItem);
                });
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania ostatnich transakcji:', error);
            document.getElementById('recent-trades-list').innerHTML = 
                '<div class="error-message">Nie można załadować transakcji</div>';
        });
}

function updateAlerts() {
    fetch('/api/alerts')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania alertów');
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.data) {
                const alertsContainer = document.getElementById('alerts-list');
                alertsContainer.innerHTML = '';

                if (data.data.length === 0) {
                    alertsContainer.innerHTML = '<div class="no-data">Brak alertów</div>';
                    document.getElementById('alerts-badge').textContent = '0';
                    return;
                }

                document.getElementById('alerts-badge').textContent = data.data.length;

                data.data.forEach(alert => {
                    const alertItem = document.createElement('div');
                    alertItem.className = 'alert-item';

                    let levelClass = 'status-online';
                    if (alert.type === 'WARNING') {
                        levelClass = 'status-warning';
                    } else if (alert.type === 'ERROR') {
                        levelClass = 'status-offline';
                    }

                    alertItem.classList.add(levelClass);

                    alertItem.innerHTML = `
                        <div class="alert-time">${alert.time}</div>
                        <div class="alert-message">${alert.message}</div>
                    `;

                    alertsContainer.appendChild(alertItem);
                });
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania alertów:', error);
            document.getElementById('alerts-list').innerHTML = 
                '<div class="error-message">Nie można załadować alertów</div>';
        });
}

function updateAIModels() {
    fetch('/api/ai-models-status')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania statusu modeli AI');
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.data) {
                const modelsContainer = document.getElementById('ai-models-container');

                if (!modelsContainer) {
                    return; // Element nie istnieje, pomiń aktualizację
                }

                modelsContainer.innerHTML = '';

                if (data.data.length === 0) {
                    modelsContainer.innerHTML = '<div class="no-data">Brak modeli AI</div>';
                    return;
                }

                data.data.forEach(model => {
                    const modelCard = document.createElement('div');
                    modelCard.className = 'ai-model-card';

                    const accuracyClass = model.accuracy >= 0.7 ? 'positive' : 
                                        model.accuracy >= 0.5 ? 'neutral' : 'negative';

                    modelCard.innerHTML = `
                        <h4>${model.name}</h4>
                        <div class="model-details">
                            <div>Status: <span class="status-${model.status}">${model.status}</span></div>
                            <div>Dokładność: <span class="${accuracyClass}">${(model.accuracy * 100).toFixed(0)}%</span></div>
                            <div>Ostatnia aktualizacja: ${model.last_update}</div>
                        </div>
                    `;

                    modelsContainer.appendChild(modelCard);
                });
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusu modeli AI:', error);
            const modelsContainer = document.getElementById('ai-models-container');
            if (modelsContainer) {
                modelsContainer.innerHTML = '<div class="error-message">Nie można załadować statusu modeli AI</div>';
            }
        });
}

function initializeChart() {
    const ctx = document.getElementById('main-chart');

    if (!ctx) {
        console.error('Nie znaleziono elementu wykresu');
        return;
    }

    // Przykładowe dane dla wykresu
    const labels = ['Styczeń', 'Luty', 'Marzec', 'Kwiecień', 'Maj', 'Czerwiec', 'Lipiec'];
    const data = {
        labels: labels,
        datasets: [{
            label: 'Portfolio Value',
            data: [10000, 10250, 10800, 10600, 11200, 11800, 12300],
            fill: false,
            borderColor: '#3498db',
            tension: 0.1
        }]
    };

    try {
        new Chart(ctx, {
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
    } catch (error) {
        console.error('Błąd podczas inicjalizacji wykresu:', error);
        // Pokaż alternatywne informacje zamiast wykresu
        ctx.parentNode.innerHTML = '<div class="error-message">Nie można załadować wykresu</div>';
    }
}

function fetchNotifications() {
    fetch('/api/notifications')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania powiadomień');
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.data) {
                const notificationsList = document.getElementById('notifications-list');

                if (!notificationsList) {
                    return; // Element nie istnieje, pomiń aktualizację
                }

                notificationsList.innerHTML = '';

                if (data.data.length === 0) {
                    notificationsList.innerHTML = '<div class="no-data">Brak powiadomień</div>';
                    document.getElementById('notifications-badge').style.display = 'none';
                    return;
                }

                document.getElementById('notifications-badge').style.display = 'inline-block';
                document.getElementById('notifications-badge').textContent = data.data.length;

                data.data.forEach(notification => {
                    const notificationItem = document.createElement('div');
                    notificationItem.className = `notification-item ${notification.type}`;

                    notificationItem.innerHTML = `
                        <div class="notification-time">${notification.time}</div>
                        <div class="notification-message">${notification.message}</div>
                    `;

                    notificationsList.appendChild(notificationItem);
                });
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania powiadomień:', error);
            const notificationsList = document.getElementById('notifications-list');
            if (notificationsList) {
                notificationsList.innerHTML = '<div class="error-message">Nie można załadować powiadomień</div>';
            }
        });
}

function setupEventListeners() {
    // Obsługa przycisku Start Trading
    const startTradingBtn = document.getElementById('start-trading-btn');
    if (startTradingBtn) {
        startTradingBtn.addEventListener('click', function() {
            fetch('/api/trading/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Trading uruchomiony pomyślnie!');
                        updateComponentStatuses();
                    } else {
                        alert('Błąd uruchamiania tradingu: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Błąd podczas uruchamiania tradingu:', error);
                    alert('Nie można uruchomić tradingu. Sprawdź konsolę po więcej informacji.');
                });
        });
    }

    // Obsługa przycisku Stop Trading
    const stopTradingBtn = document.getElementById('stop-trading-btn');
    if (stopTradingBtn) {
        stopTradingBtn.addEventListener('click', function() {
            fetch('/api/trading/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Trading zatrzymany pomyślnie!');
                        updateComponentStatuses();
                    } else {
                        alert('Błąd zatrzymywania tradingu: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Błąd podczas zatrzymywania tradingu:', error);
                    alert('Nie można zatrzymać tradingu. Sprawdź konsolę po więcej informacji.');
                });
        });
    }

    // Obsługa przycisku Reset System
    const resetSystemBtn = document.getElementById('reset-system-btn');
    if (resetSystemBtn) {
        resetSystemBtn.addEventListener('click', function() {
            if (confirm('Czy na pewno chcesz zresetować system? Spowoduje to utratę wszystkich niezapisanych danych.')) {
                fetch('/api/system/reset', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('System zresetowany pomyślnie!');
                            window.location.reload();
                        } else {
                            alert('Błąd resetowania systemu: ' + data.message);
                        }
                    })
                    .catch(error => {
                        console.error('Błąd podczas resetowania systemu:', error);
                        alert('Nie można zresetować systemu. Sprawdź konsolę po więcej informacji.');
                    });
            }
        });
    }

    // Obsługa formularza ustawień systemu
    const systemSettingsForm = document.getElementById('system-settings-form');
    if (systemSettingsForm) {
        systemSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(systemSettingsForm);
            const settings = {
                risk_level: formData.get('risk_level'),
                max_position_size: parseFloat(formData.get('max_position_size')),
                enable_auto_trading: formData.get('enable_auto_trading') === 'on'
            };

            fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Ustawienia zapisane pomyślnie!');
                    } else {
                        alert('Błąd zapisywania ustawień: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Błąd podczas zapisywania ustawień:', error);
                    alert('Nie można zapisać ustawień. Sprawdź konsolę po więcej informacji.');
                });
        });
    }
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
    // Tutaj można dodać kod do aktualizacji wykresów jeśli są dynamiczne
}

function showStatusError(message) {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.innerHTML = `<div class="error-message">${message}</div>`;
        errorContainer.style.display = 'block';
    }
}

function hideStatusError() {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.style.display = 'none';
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

function getAccuracyClass(accuracy) {
    if (accuracy >= 70) return 'positive';
    if (accuracy >= 50) return 'neutral';
    return 'negative';
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function retryLoadChart() {
    hideChartError();
    initializeChart(); // Call initializeChart instead of updateCharts
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

// Funkcja aktualizująca dane portfela
function updatePortfolio() {
    fetch('/api/portfolio')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd pobierania danych portfela');
            }
            return response.json();
        })
        .then(data => {
            portfolioErrorCount = 0;
            if (data.success) {
                displayPortfolioData(data.data);
                renderPortfolioChart(data.data.assets);
            } else {
                console.error('Błąd pobierania danych portfela:', data.error);
                portfolioErrorCount++;
                if (portfolioErrorCount > 3) {
                    showErrorMessage('Nie można pobrać danych portfela.');
                }
            }
        })
        .catch(error => {
            console.error('Błąd pobierania danych portfela:', error);
            portfolioErrorCount++;
            if (portfolioErrorCount > 3) {
                showErrorMessage('Nie można pobrać danych portfela.');
            }
        });
}

// Wyświetlanie danych portfela
function displayPortfolioData(portfolio) {
    // Aktualizacja wartości portfela
    document.getElementById('total-portfolio-value').textContent = portfolio.total_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    // Aktualizacja PnL
    const pnlElement = document.getElementById('portfolio-pnl');
    const pnlValue = portfolio.pnl_percentage;
    pnlElement.textContent = (pnlValue > 0 ? '+' : '') + pnlValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + '%';
    pnlElement.className = 'pnl ' + (pnlValue >= 0 ? 'positive' : 'negative');

    // Aktualizacja tabeli aktywów
    const tableBody = document.getElementById('portfolio-assets');
    tableBody.innerHTML = '';

    portfolio.assets.forEach(asset => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${asset.symbol}</td>
            <td>${asset.amount.toLocaleString('en-US', { minimumFractionDigits: asset.symbol === 'USDT' ? 2 : 8, maximumFractionDigits: asset.symbol === 'USDT' ? 2 : 8 })}</td>
            <td>$${asset.value_usd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td>${asset.allocation.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%</td>
            <td class="${asset.pnl_24h >= 0 ? 'positive' : 'negative'}">${(asset.pnl_24h > 0 ? '+' : '') + asset.pnl_24h.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%</td>
        `;
        tableBody.appendChild(row);
    });
}

// Renderowanie wykresu alokacji portfela
function renderPortfolioChart(assets) {
    const ctx = document.getElementById('portfolio-allocation-chart').getContext('2d');

    // Dane dla wykresu
    const labels = assets.map(asset => asset.symbol);
    const data = assets.map(asset => asset.value_usd);
    const backgroundColors = [
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 99, 132, 0.8)',
        'rgba(255, 206, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)',
        'rgba(153, 102, 255, 0.8)',
        'rgba(255, 159, 64, 0.8)',
        'rgba(199, 199, 199, 0.8)'
    ];

    // Zniszcz istniejący wykres jeśli istnieje
    if (window.portfolioChart) {
        window.portfolioChart.destroy();
    }

    // Utwórz nowy wykres
    window.portfolioChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors.slice(0, assets.length),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                },
                title: {
                    display: true,
                    text: 'Alokacja portfela'
                }
            }
        }
    });
}


// Funkcje do obsługi modalu trenowania modeli
function setupModelTraining() {
    const trainBtn = document.getElementById('train-model-btn');
    const modal = document.getElementById('train-model-modal');
    const closeBtn = modal.querySelector('.close');
    const cancelBtn = document.getElementById('cancel-training-btn');
    const startTrainingBtn = document.getElementById('start-training-btn');
    const trainingStatus = document.getElementById('training-status');

    // Otwieranie modalu
    trainBtn.addEventListener('click', function() {
        modal.style.display = 'block';
    });

    // Zamykanie modalu
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    cancelBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    // Kliknięcie poza modalem
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Rozpoczęcie trenowania
    startTrainingBtn.addEventListener('click', function() {
        const symbol = document.getElementById('model-symbol').value;
        const interval = document.getElementById('model-interval').value;
        const lookback = document.getElementById('model-lookback').value;

        // Pokazujemy status trenowania
        trainingStatus.style.display = 'block';
        startTrainingBtn.disabled = true;
        cancelBtn.disabled = true;

        // Wysyłamy zapytanie do API
        fetch('/api/train-ai-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: symbol,
                interval: interval,
                lookback_days: parseInt(lookback)
            })
        })
        .then(response => response.json())
        .then(data => {
            trainingStatus.style.display = 'none';
            startTrainingBtn.disabled = false;
            cancelBtn.disabled = false;

            if (data.success) {
                alert(`Model dla ${symbol} został wytrenowany pomyślnie!`);
                modal.style.display = 'none';

                // Odświeżamy status modeli
                fetch('/api/ai-models-status')
                    .then(response => response.json())
                    .then(data => {
                        // ... Tutaj można zaktualizować tabelę modeli
                        updateDashboardData();
                    });
            } else {
                alert(`Błąd podczas trenowania modelu: ${data.error || 'Nieznany błąd'}`);
            }
        })
        .catch(error => {
            console.error('Error training model:', error);
            trainingStatus.style.display = 'none';
            startTrainingBtn.disabled = false;
            cancelBtn.disabled = false;
            alert('Wystąpił błąd podczas komunikacji z serwerem.');
        });
    });
}

// Inicjalizacja strony
document.addEventListener('DOMContentLoaded', function() {
    // Pierwszy load
    updateDashboardData();

    // Aktualizacja co 5 sekund
    setInterval(updateDashboardData, 5000);

    // Inicjalizacja trenowania modeli
    if (document.getElementById('train-model-btn')) {
        setupModelTraining();
    }

    console.log("Dashboard załadowany");
});