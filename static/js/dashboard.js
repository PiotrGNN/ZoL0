// Konfiguracja aplikacji
const CONFIG = {
    updateInterval: 30000, // 30 sekund
    chartUpdateInterval: 60000, // 1 minuta
    maxErrors: 3,
    apiEndpoints: {
        portfolio: '/api/portfolio',
        dashboard: '/api/dashboard/data',
        componentStatus: '/api/component-status',
        chartData: '/api/chart/data',
        aiModelsStatus: '/api/ai-models-status',
        simulationResults: '/api/simulation-results'
    }
};

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
    updateAIModelsStatus();
}

// Rozpoczęcie automatycznych aktualizacji danych
function startDataUpdates() {
    // Regularne aktualizacje danych dashboardu
    setInterval(function() {
        if (appState.activeDashboard) {
            updateDashboardData();
            updateComponentStatus();
            updateAIModelsStatus();
            updatePortfolioData(); // Added portfolio update
        }
    }, CONFIG.updateInterval);

    // Regularne aktualizacje wykresu (rzadziej)
    setInterval(function() {
        if (appState.activeDashboard) {
            updateChartData();
        }
    }, CONFIG.chartUpdateInterval);
}

// Aktualizacja statusu modeli AI
function updateAIModelsStatus() {
    fetch('/api/ai-models-status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const aiModelsContainer = document.getElementById('ai-models-container');
            if (!aiModelsContainer) {
                console.error("Element 'ai-models-container' nie istnieje");
                return;
            }

            if (data.models && data.models.length > 0) {
                let modelsHtml = '';

                data.models.forEach(model => {
                    let statusClass = model.status === 'Active' ? 'positive' : 
                                     (model.status === 'Inactive' ? 'neutral' : 
                                     (model.status === 'Error' ? 'negative' : 'neutral'));

                    let accuracyClass = model.accuracy >= 70 ? 'positive' : 
                                      (model.accuracy >= 50 ? 'neutral' : 'negative');

                    let cardStatusClass = model.status.toLowerCase();

                    let testResultHtml = '';
                    if (model.test_result) {
                        let testClass = model.test_result === 'Passed' ? 'positive' : 
                                      (model.test_result === 'Failed' ? 'negative' : 'neutral');
                        testResultHtml = `
                            <div>Test: <span class="${testClass}">${model.test_result}</span></div>`;
                    }

                    let moduleHtml = '';
                    if (model.module) {
                        moduleHtml = `<div>Moduł: <span>${model.module}</span></div>`;
                    }

                    let errorHtml = '';
                    if (model.error) {
                        errorHtml = `<div class="error-message">${model.error}</div>`;
                    }

                    modelsHtml += `
                    <div class="ai-model-card ${cardStatusClass}">
                        <h4>${model.name}</h4>
                        <div class="model-details">
                            <div>Typ: <span>${model.type}</span></div>
                            <div>Dokładność: <span class="${accuracyClass}">${model.accuracy.toFixed(1)}%</span></div>
                            <div>Status: <span class="${statusClass}">${model.status}</span></div>
                            <div>Ostatnie użycie: <span>${model.last_used || 'Nieznane'}</span></div>
                            <div>Metody: 
                                <div>
                                    <span class="${model.has_predict ? 'positive' : 'negative'}">predict ${model.has_predict ? '✓' : '✗'}</span>, 
                                    <span class="${model.has_fit ? 'positive' : 'negative'}">fit ${model.has_fit ? '✓' : '✗'}</span>
                                </div>
                            </div>
                            ${testResultHtml}
                            ${moduleHtml}
                            ${errorHtml}
                        </div>
                    </div>`;
                });

                aiModelsContainer.innerHTML = modelsHtml;

                // Dodaj podsumowanie
                const aiModelsSection = document.getElementById('ai-models-section');
                if (aiModelsSection) {
                    const activeModels = data.models.filter(m => m.status === 'Active').length;
                    const totalModels = data.models.length;

                    // Zaktualizuj nagłówek
                    const header = aiModelsSection.querySelector('h2');
                    if (header) {
                        header.innerHTML = `Modele AI <span class="models-count">(${activeModels}/${totalModels} aktywnych)</span>`;
                    }
                }
            } else {
                aiModelsContainer.innerHTML = '<div class="no-data">Brak dostępnych modeli AI</div>';
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania statusu modeli AI:", error);
            const aiModelsContainer = document.getElementById('ai-models-container');
            if (aiModelsContainer) {
                aiModelsContainer.innerHTML = '<div class="error-message">Błąd podczas pobierania statusu modeli AI</div>';
            }
            // Próbujemy ponownie po 15 sekundach zamiast 5 dla zmniejszenia obciążenia serwera
            setTimeout(updateAIModelsStatus, 15000);
        });
}

// Inicjalizacja wykresu portfela
function initializePortfolioChart() {
    const ctx = document.getElementById('main-chart');
    if (!ctx) {
        console.log("Element 'main-chart' nie został znaleziony");
        return;
    }

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

    fetch('/api/chart-data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
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
    fetch(CONFIG.apiEndpoints.portfolio)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const portfolioContainer = document.getElementById('portfolio-data');
            if (!portfolioContainer) {
                console.error("Element 'portfolio-data' nie istnieje");
                return;
            }

            if (data && data.success) {
                let portfolioHtml = `
                    <div class="portfolio-summary-data">
                        <div class="portfolio-item">
                            <span class="label">Początkowy balans:</span>
                            <span class="value">${data.portfolio.initial_balance.toFixed(2)} ${data.portfolio.currency}</span>
                        </div>
                        <div class="portfolio-item">
                            <span class="label">Aktualny balans:</span>
                            <span class="value">${data.portfolio.current_balance.toFixed(2)} ${data.portfolio.currency}</span>
                        </div>
                        <div class="portfolio-item">
                            <span class="label">Wartość pozycji:</span>
                            <span class="value">${data.portfolio.total_positions_value.toFixed(2)} ${data.portfolio.currency}</span>
                        </div>
                        <div class="portfolio-item">
                            <span class="label">Całkowita wartość:</span>
                            <span class="value">${data.portfolio.total_value.toFixed(2)} ${data.portfolio.currency}</span>
                        </div>
                        <div class="portfolio-item">
                            <span class="label">Zmiana (%):</span>
                            <span class="value ${data.portfolio.pct_change >= 0 ? 'positive' : 'negative'}">${data.portfolio.pct_change.toFixed(2)}%</span>
                        </div>
                        <div class="portfolio-item">
                            <span class="label">Zrealizowany P/L:</span>
                            <span class="value ${data.portfolio.realized_pnl >= 0 ? 'positive' : 'negative'}">${data.portfolio.realized_pnl.toFixed(2)} ${data.portfolio.currency}</span>
                        </div>
                        <div class="portfolio-item">
                            <span class="label">Niezrealizowany P/L:</span>
                            <span class="value ${data.portfolio.unrealized_pnl >= 0 ? 'positive' : 'negative'}">${data.portfolio.unrealized_pnl.toFixed(2)} ${data.portfolio.currency}</span>
                        </div>
                        <div class="portfolio-item">
                            <span class="label">Ostatnia aktualizacja:</span>
                            <span class="value">${data.portfolio.last_update}</span>
                        </div>
                    </div>
                `;
                portfolioContainer.innerHTML = portfolioHtml;


                // Aktualizacja otwartych pozycji
                let positionsHtml = '';
                const positions = data.positions;
                const positionKeys = Object.keys(positions);

                if (positionKeys.length > 0) {
                    positionsHtml = '<table class="table table-striped table-sm"><thead><tr>' +
                        '<th>Symbol</th><th>Ilość</th><th>Cena wejścia</th><th>Aktualna cena</th><th>Wartość</th><th>P/L</th>' +
                        '</tr></thead><tbody>';

                    positionKeys.forEach(symbol => {
                        const position = positions[symbol];
                        const pnl = position.unrealized_pnl || 0;

                        positionsHtml += `<tr>
                            <td>${symbol}</td>
                            <td>${position.quantity}</td>
                            <td>${position.entry_price.toFixed(2)}</td>
                            <td>${position.current_price.toFixed(2)}</td>
                            <td>${position.market_value.toFixed(2)} ${data.portfolio.currency}</td>
                            <td class="${pnl >= 0 ? 'positive' : 'negative'}">${pnl.toFixed(2)} ${data.portfolio.currency}</td>
                        </tr>`;
                    });

                    positionsHtml += '</tbody></table>';
                } else {
                    positionsHtml = '<p>Brak otwartych pozycji</p>';
                }

                const positionsContainer = document.getElementById('portfolio-positions');
                if(positionsContainer) positionsContainer.innerHTML = positionsHtml;
            } else {
                portfolioContainer.innerHTML = '<div class="error-message">Błąd podczas pobierania danych portfela</div>';
            }
        })
        .catch(err => {
            console.error("Błąd podczas pobierania danych portfela:", err);
            const portfolioContainer = document.getElementById('portfolio-data');
            if (portfolioContainer) {
                portfolioContainer.innerHTML = '<div class="error-message">Błąd podczas pobierania danych portfela</div>';
            }
        });
}

// Aktualizacja głównych danych dashboardu
function updateDashboardData() {
    console.log("Aktualizacja danych dashboardu...");
    fetch('/api/dashboard/data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Bezpieczne aktualizowanie elementów z kontrolą istnienia
                updateElementById('profit-value', `${data.balance.toFixed(2)} USDT`);
                updateElementById('trades-value', data.total_trades);
                updateElementById('win-rate-value', `${data.win_rate.toFixed(1)}%`);
                updateElementById('drawdown-value', `${data.max_drawdown.toFixed(1)}%`);

                // Aktualizacja sekcji sentymentu w zakładce analityki
                if (data.sentiment_data) {
                    updateSentimentSection(data.sentiment_data);
                }

                // Aktualizacja czasu
                updateLastRefreshed();
            } else {
                console.error("Błąd podczas pobierania danych dashboardu:", data.error);
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania danych dashboardu:", error);
        });

    // Pobierz wyniki symulacji tradingu
    fetch('/api/simulation-results')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd HTTP: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            updateSimulationResults(data);
        })
        .catch(error => {
            console.error("Błąd podczas pobierania wyników symulacji:", error);
        });
}

// Funkcja do aktualizacji wyników symulacji
function updateSimulationResults(data) {
    if (data.status !== 'success') {
        document.getElementById('simulationSummary').innerHTML = `<p>Brak danych symulacji: ${data.message}</p>`;
        return;
    }

    const summary = data.summary;
    const trades = data.trades;

    // Aktualizacja podsumowania
    const summaryHTML = `
        <div class="table-responsive">
            <table class="table table-sm">
                <tbody>
                    <tr>
                        <td>Kapitał początkowy:</td>
                        <td>${summary.initial_capital.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Kapitał końcowy:</td>
                        <td>${summary.final_capital.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Zysk/strata:</td>
                        <td class="${summary.profit >= 0 ? 'text-success' : 'text-danger'}">${summary.profit.toFixed(2)} (${summary.profit_percentage.toFixed(2)}%)</td>
                    </tr>
                    <tr>
                        <td>Liczba transakcji:</td>
                        <td>${summary.trades}</td>
                    </tr>
                    <tr>
                        <td>Win rate:</td>
                        <td>${summary.win_rate.toFixed(2)}% (${summary.winning_trades}/${summary.closes})</td>
                    </tr>
                    <tr>
                        <td>Max drawdown:</td>
                        <td>${summary.max_drawdown.toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Prowizje:</td>
                        <td>${summary.total_commission.toFixed(2)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    document.getElementById('simulationSummary').innerHTML = summaryHTML;

    // Aktualizacja wykresu
    const chartHTML = `
        <img src="${data.chart_path}" alt="Wykres symulacji" class="img-fluid" />
    `;

    document.getElementById('simulationChart').innerHTML = chartHTML;

    // Aktualizacja historii transakcji
    const tradeHistoryTable = document.getElementById('tradeHistoryTable').getElementsByTagName('tbody')[0];
    tradeHistoryTable.innerHTML = '';

    if (trades && trades.length > 0) {
        trades.forEach(trade => {
            const row = tradeHistoryTable.insertRow();

            // Format daty/czasu
            const timestamp = new Date(trade.timestamp * 1000).toLocaleString();

            // Dodaj komórki
            let cell = row.insertCell();
            cell.textContent = timestamp;

            cell = row.insertCell();
            cell.textContent = trade.action;

            cell = row.insertCell();
            cell.textContent = trade.price.toFixed(2);

            cell = row.insertCell();
            cell.textContent = trade.size.toFixed(4);

            cell = row.insertCell();
            if (trade.pnl !== undefined) {
                cell.textContent = trade.pnl.toFixed(2);
                if (trade.pnl > 0) {
                    cell.classList.add('text-success');
                } else if (trade.pnl < 0) {
                    cell.classList.add('text-danger');
                }
            } else {
                cell.textContent = '-';
            }

            cell = row.insertCell();
            cell.textContent = trade.commission.toFixed(2);

            cell = row.insertCell();
            cell.textContent = trade.capital.toFixed(2);
        });
    } else {
        const row = tradeHistoryTable.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 7;
        cell.textContent = 'Brak historii transakcji';
        cell.style.textAlign = 'center';
    }
}

// Bezpieczna aktualizacja elementu po ID
function updateElementById(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    } else {
        console.log(`Element o ID '${elementId}' nie istnieje`);
    }
}

function updateSentimentSection(sentimentData) {
    const sentimentContainer = document.getElementById('sentiment-container');
    if (!sentimentContainer) {
        console.log("Element 'sentiment-container' nie istnieje");
        return;
    }

    if (!sentimentData) {
        sentimentContainer.innerHTML = '<div class="no-data">Brak danych o sentymencie rynkowym</div>';
        return;
    }

    let sentimentClass = 'neutral';
    if (sentimentData.overall_score > 0.1) {
        sentimentClass = 'positive';
    } else if (sentimentData.overall_score < -0.1) {
        sentimentClass = 'negative';
    }

    let sourcesHtml = '';
    for (const [source, data] of Object.entries(sentimentData.sources)) {
        let sourceClass = 'neutral';
        if (data.score > 0.1) {
            sourceClass = 'positive';
        } else if (data.score < -0.1) {
            sourceClass = 'negative';
        }

        sourcesHtml += `
        <li>
            <strong>${source}</strong>: 
            <span class="${sourceClass}">
                ${data.score.toFixed(2)}
            </span>
            (${data.volume} wzmianek)
        </li>`;
    }

    sentimentContainer.innerHTML = `
        <div class="sentiment-score">
            <div class="sentiment-label">Ogólny sentyment:</div>
            <div class="sentiment-value ${sentimentClass}">
                ${sentimentData.analysis}
            </div>
        </div>

        <div class="sentiment-details">
            <h4>Źródła danych:</h4>
            <ul>
                ${sourcesHtml}
            </ul>
        </div>

        <div class="sentiment-footer">
            <div>Zakres czasowy: ${sentimentData.time_range}</div>
            <div>Ostatnia aktualizacja: ${sentimentData.timestamp}</div>
        </div>
    `;
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
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.components) {
                let hasWarnings = false;
                let hasErrors = false;

                data.components.forEach(component => {
                    const componentElement = document.getElementById(component.id);
                    if (componentElement) {
                        // Usunięcie poprzednich klas statusu
                        componentElement.classList.remove('online', 'warning', 'offline');
                        // Dodanie nowej klasy statusu
                        componentElement.classList.add(component.status);

                        // Aktualizacja tekstu statusu
                        const statusTextElement = componentElement.querySelector('.status-text');
                        if (statusTextElement) {
                            statusTextElement.textContent = component.status.charAt(0).toUpperCase() + component.status.slice(1);
                        }

                        // Sprawdź, czy są ostrzeżenia lub błędy
                        if (component.status === 'warning') {
                            hasWarnings = true;
                        } else if (component.status === 'offline') {
                            hasErrors = true;
                        }
                    }
                });

                // Pokaż powiadomienie, jeśli są problemy
                if (hasErrors) {
                    showNotification('error', 'Co najmniej jeden komponent jest offline. Sprawdź status systemu.');
                } else if (hasWarnings) {
                    showNotification('warning', 'Niektóre komponenty mają status ostrzeżenia. Sprawdź logi systemu.');
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas aktualizacji statusu komponentów:', error);
            showNotification('error', 'Nie udało się pobrać statusu komponentów');
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
    switch(severity?.toLowerCase()) {
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
    // Sprawdź, czy kontener istnieje
    let container = document.getElementById('notifications-container');

    // Jeśli nie istnieje, utwórz go
    if (!container) {
        container = document.createElement('div');
        container.id = 'notifications-container';
        container.className = 'notifications-container';
        document.body.appendChild(container);
    }

    // Stwórz element powiadomienia
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    // Dodaj do kontenera powiadomień
    container.appendChild(notification);

    // Usuń po 5 sekundach
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            if (container.contains(notification)) {
                container.removeChild(notification);
            }
        }, 500);
    }, 5000);
}

// Funkcje zarządzania stanem tradingu
function startTrading() {
    fetch('/api/trading/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Błąd HTTP: ${response.status}`);
        }
        return response.json();
    })
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
    .then(response => {
        if (!response.ok) {
            throw new Error(`Błąd HTTP: ${response.status}`);
        }
        return response.json();
    })
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
    .then(response => {
        if (!response.ok) {
            throw new Error(`Błąd HTTP: ${response.status}`);
        }
        return response.json();
    })
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