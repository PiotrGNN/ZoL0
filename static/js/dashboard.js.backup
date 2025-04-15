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
        simulationResults: '/api/simulation-results',
        sentiment: '/api/sentiment', // Added endpoint for sentiment data
        systemStatus: '/api/system/status' // Added endpoint for system status
    }
};

// Funkcja do pobierania statusu komponentów
function fetchComponentStatus() {
    fetch(CONFIG.apiEndpoints.systemStatus)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Aktualizacja statusu komponentów
                Object.keys(data.components).forEach(component => {
                    const element = document.getElementById(component);
                    if (element) {
                        element.className = `status-item status-${data.components[component].status.toLowerCase()}`;
                        element.querySelector('.status-text').innerText = data.components[component].status;
                    }
                });
            } else {
                console.error('Błąd podczas pobierania statusu komponentów:', data.error);
            }
        })
        .catch(error => console.error('Błąd podczas pobierania statusu komponentów:', error));
}

// Funkcja do pobierania statusu modeli AI
function fetchAIStatus() {
    fetch('/api/ai-models-status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateAIModelsStatus(data);
            updateLastRefreshed();
        })
        .catch(error => {
            console.log("Błąd podczas pobierania statusu modeli AI:", error);
            showNotification('error', 'Nie udało się pobrać statusu modeli AI');
        });
}

// Funkcja do aktualizacji UI na podstawie statusu komponentów
function updateComponentStatusUI(data) {
    // Aktualizacja statusu API Connector
    const apiConnector = document.getElementById('api-connector');
    if (apiConnector) {
        apiConnector.className = `status-item status-${data.api || 'offline'}`;
        apiConnector.querySelector('.status-text').textContent = capitalizeFirstLetter(data.api || 'offline');
    }

    // Aktualizacja statusu Trading Engine
    const tradingEngine = document.getElementById('trading-engine');
    if (tradingEngine) {
        tradingEngine.className = `status-item status-${data.trading_engine || 'offline'}`;
        tradingEngine.querySelector('.status-text').textContent = capitalizeFirstLetter(data.trading_engine || 'offline');
    }

    // Aktualizacja statusu Data Processor
    const dataProcessor = document.getElementById('data-processor');
    if (dataProcessor) {
        dataProcessor.className = `status-item status-${data.api || 'offline'}`;
        dataProcessor.querySelector('.status-text').textContent = capitalizeFirstLetter(data.api || 'offline');
    }

    // Aktualizacja statusu Risk Manager
    const riskManager = document.getElementById('risk-manager');
    if (riskManager) {
        riskManager.className = `status-item status-${data.trading_engine || 'offline'}`;
        riskManager.querySelector('.status-text').textContent = capitalizeFirstLetter(data.trading_engine || 'offline');
    }
}

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
    setupTabNavigation(); // Dodano jawne wywołanie
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
        }
    }, CONFIG.updateInterval);

    // Regularne aktualizacje wykresu (rzadziej)
    setInterval(function() {
        if (appState.activeDashboard) {
            updateChartData();
        }
    }, CONFIG.chartUpdateInterval);

    // Inicjalne pobranie statusu komponentów
    fetchComponentStatus();

    // Ustawienie interwału do aktualizacji statusu
    setInterval(fetchComponentStatus, 30000); // co 30 sekund
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
            // Aktualizacja kart modeli AI
            const aiModelsContainer = document.getElementById('ai-models-container');
            if (aiModelsContainer) {
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
            }

            // Aktualizacja tabeli modeli AI
            const aiModelTable = document.getElementById('aiModelTable');
            if (aiModelTable) {
                const tbody = aiModelTable.querySelector('tbody');
                if (tbody && data.models && data.models.length > 0) {
                    tbody.innerHTML = '';

                    data.models.forEach(model => {
                        const row = document.createElement('tr');

                        // Komórka z nazwą modelu
                        const nameCell = document.createElement('td');
                        nameCell.textContent = model.name;
                        row.appendChild(nameCell);

                        // Komórka ze statusem
                        const statusCell = document.createElement('td');
                        const statusSpan = document.createElement('span');
                        statusSpan.className = `status-${model.status.toLowerCase()}`;
                        statusSpan.textContent = model.status;
                        statusCell.appendChild(statusSpan);
                        row.appendChild(statusCell);

                        // Komórka z typem
                        const typeCell = document.createElement('td');
                        typeCell.textContent = model.type;
                        row.appendChild(typeCell);

                        // Komórka z metrykami
                        const metricsCell = document.createElement('td');
                        if (model.accuracy !== undefined) {
                            const accuracySpan = document.createElement('span');
                            let accuracyClass = '';
                            if (model.accuracy >= 70) accuracyClass = 'positive';
                            else if (model.accuracy >= 50) accuracyClass = 'neutral';
                            else accuracyClass = 'negative';

                            accuracySpan.className = accuracyClass;
                            accuracySpan.textContent = `Dokładność: ${model.accuracy.toFixed(1)}%`;
                            metricsCell.appendChild(accuracySpan);
                        }
                        row.appendChild(metricsCell);

                        tbody.appendChild(row);
                    });
                } else if (tbody) {
                    tbody.innerHTML = '<tr><td colspan="4" class="text-center">Brak dostępnych modeli AI</td></tr>';
                }
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania statusu modeli AI:", error);
            const aiModelsContainer = document.getElementById('ai-models-container');
            if (aiModelsContainer) {
                aiModelsContainer.innerHTML = '<div class="error-message">Błąd podczas pobierania statusu modeli AI</div>';
            }

            // Aktualizuj tabelę modeli w przypadku błędu
            const aiModelTable = document.getElementById('aiModelTable');
            if (aiModelTable) {
                const tbody = aiModelTable.querySelector('tbody');
                if (tbody) {
                    tbody.innerHTML = '<tr><td colspan="4" class="text-center">Błąd podczas pobierania modeli AI</td></tr>';
                }
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
    fetch('/api/portfolio')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Próbujemy znaleźć portfolio-container zamiast portfolio-data
            let portfolioContainer = document.getElementById('portfolio-container');

            // Sprawdźmy również portfolio-data jako alternatywę
            if (!portfolioContainer) {
                portfolioContainer = document.getElementById('portfolio-data');
            }

            // Jeśli nadal nie znaleziono, spróbuj znaleźć dowolny element z klasą portfolio-data
            if (!portfolioContainer) {
                const possibleContainers = document.getElementsByClassName('portfolio-data');
                if (possibleContainers.length > 0) {
                    portfolioContainer = possibleContainers[0];
                }
            }

            if (!portfolioContainer) {
                console.error("Element portfolio-container ani portfolio-data nie istnieje");
                
                // Stwórz element portfolio-container, jeśli nie istnieje
                const mainContent = document.querySelector('.dashboard-grid');
                if (mainContent) {
                    const newContainer = document.createElement('div');
                    newContainer.id = 'portfolio-container';
                    newContainer.className = 'portfolio-data';
                    
                    const firstCard = mainContent.querySelector('.card');
                    if (firstCard) {
                        firstCard.appendChild(newContainer);
                        portfolioContainer = newContainer;
                    }
                }
                
                if (!portfolioContainer) return;
            }

            if (data && data.success === true && data.balances && Object.keys(data.balances).length > 0) {
                portfolioContainer.innerHTML = ''; // Wyczyść kontener

                // Przetwarzanie danych portfela
                for (const [currency, details] of Object.entries(data.balances)) {
                    if (details) {
                        const balanceItem = document.createElement('div');
                        balanceItem.className = 'balance-item';

                        // Bezpieczne wyświetlanie wartości z obsługą wartości null/undefined
                        const equity = typeof details.equity === 'number' ? details.equity.toFixed(4) : '0.0000';
                        const available = typeof details.available_balance === 'number' ? details.available_balance.toFixed(4) : '0.0000';
                        const wallet = typeof details.wallet_balance === 'number' ? details.wallet_balance.toFixed(4) : '0.0000';

                        balanceItem.innerHTML = `
                            <div class="currency">${currency}</div>
                            <div class="balance-details">
                                <div class="balance-row"><span>Equity:</span> <span>${equity}</span></div>
                                <div class="balance-row"><span>Available:</span> <span>${available}</span></div>
                                <div class="balance-row"><span>Wallet:</span> <span>${wallet}</span></div>
                            </div>
                        `;
                        portfolioContainer.appendChild(balanceItem);
                    }
                }
            } else {
                portfolioContainer.innerHTML = '<div class="error-message">Brak danych portfela lub problem z połączeniem z ByBit. Sprawdź klucze API w ustawieniach.</div>';

                // Wyświetl dane diagnostyczne w logach konsoli
                console.log("Otrzymane dane portfela:", data);
                if (data && data.error) {
                    console.log("Błąd API:", data.error);
                }
            }
        })
        .catch(err => {
            console.error("Błąd podczas pobierania danych portfela:", err);
            const possibleContainers = [
                document.getElementById('portfolio-container'),
                document.getElementById('portfolio-data'),
                ...Array.from(document.getElementsByClassName('portfolio-data'))
            ].filter(Boolean);

            if (possibleContainers.length > 0) {
                possibleContainers[0].innerHTML = '<div class="error-message">Błąd podczas pobierania danych portfela. Sprawdź połączenie internetowe.</div>';
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
    updateSentimentData(); //Dodano aktualizację danych sentymentu
}

// Funkcja do aktualizacji danych sentymentu
function updateSentimentData() {
    fetch('/api/sentiment')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Aktualizuj kontener sentymentu z otrzymanymi danymi
            const sentimentContainer = document.getElementById('sentiment-container');
            if (sentimentContainer && data) {
                if (data.value !== undefined && data.analysis) {
                    // Aktualizacja głównej wartości sentymentu
                    const sentimentValue = document.getElementById('sentiment-value');
                    if (sentimentValue) {
                        sentimentValue.textContent = data.analysis;

                        // Usunięcie poprzednich klas
                        sentimentValue.classList.remove('positive', 'negative', 'neutral');

                        // Dodanie właściwej klasy na podstawie wartości
                        if (data.value > 0.1) {
                            sentimentValue.classList.add('positive');
                        } else if (data.value < -0.1) {
                            sentimentValue.classList.add('negative');
                        } else {
                            sentimentValue.classList.add('neutral');
                        }
                    }

                    // Aktualizacja danych źródłowych, jeśli są dostępne
                    if (data.sources) {
                        const sourcesContainer = document.getElementById('sentiment-sources');
                        if (sourcesContainer) {
                            let sourcesHtml = '';
                            for (const [source, info] of Object.entries(data.sources)) {
                                const scoreClass = info.score > 0 ? 'positive' : info.score < 0 ? 'negative' : 'neutral';
                                sourcesHtml += `
                                    <div class="source-item">
                                        <div class="source-name">${source}</div>
                                        <div class="source-score ${scoreClass}">${info.score.toFixed(2)}</div>
                                        <div class="source-volume">${info.volume} wzmianek</div>
                                    </div>
                                `;
                            }
                            sourcesContainer.innerHTML = sourcesHtml || '<div class="no-data">Brak danych źródłowych</div>';
                        }
                    }
                } else {
                    sentimentContainer.innerHTML = '<div class="no-data">Brak danych sentymentu rynkowego</div>';
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych sentymentu:', error);
            const sentimentContainer = document.getElementById('sentiment-container');
            if (sentimentContainer) {
                sentimentContainer.innerHTML = '<div class="error-message">Błąd podczas pobierania danych sentymentu</div>';
                
                // Wyświetl dane zastępcze
                setTimeout(() => {
                    sentimentContainer.innerHTML = `
                        <div class="sentiment-score">
                            <div class="sentiment-label">Ogólny sentyment (dane zastępcze):</div>
                            <div class="sentiment-value neutral">Neutralny</div>
                        </div>
                        <div class="sentiment-details">
                            <p>Serwer sentymentu jest niedostępny. Wyświetlam dane testowe.</p>
                        </div>
                    `;
                }, 3000);
            }
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
    .then(response=> {
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

// Obsługa zakładek w dashboardzie
function setupTabNavigation() {
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
                    console.log(`Przełączono na zakładkę: ${tabId}`);
                } else {
                    console.error(`Nie znaleziono elementu o ID: ${tabId}`);
                }

                // Specjalne działania dla poszczególnych zakładek
                if (tabId === 'trades-tab') {
                    fetchTradesHistory();
                } else if (tabId === 'analytics-tab') {
                    updateSentimentData();
                } else if (tabId === 'ai-monitor-tab') {
                    fetchAIStatus();
                    fetchAIThoughts();
                } else if (tabId === 'settings-tab') {
                    fetchSystemSettings();
                } else if (tabId === 'notifications-tab') {
                    fetchNotifications();
                }
            });
        });
    } else {
        console.error("Nie znaleziono przycisków zakładek (.tab-button)");
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

    // Inicjalizacja nawigacji zakładek
    setupTabNavigation();

    // Obsługa formularza symulacji
    const simulationForm = document.getElementById('simulation-form');
    if (simulationForm) {
        simulationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            runSimulation();
        });
    }

    // Widoczność (aby wstrzymać aktualizacje, gdy karta nie jest widoczna)
    document.addEventListener('visibilitychange', function() {
        appState.activeDashboard = !document.hidden;
    });
}

// Funkcje obsługujące poszczególne zakładki
function fetchTradesHistory() {
    fetch('/api/trades/history')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Update the trades table with the received data
            const tradesTableBody = document.getElementById('trades-table-body');
            if (tradesTableBody && data && data.trades) {
                if (data.trades.length > 0) {
                    let html = '';
                    data.trades.forEach(trade => {
                        html += `<tr>
                            <td>${trade.id || '-'}</td>
                            <td>${trade.timestamp || '-'}</td>
                            <td>${trade.symbol || '-'}</td>
                            <td>${trade.type || '-'}</td>
                            <td>${trade.price ? trade.price.toFixed(2) : '-'}</td>
                            <td>${trade.size ? trade.size.toFixed(4) : '-'}</td>
                            <td>${trade.value ? trade.value.toFixed(2) : '-'}</td>
                            <td class="${trade.profit > 0 ? 'positive' : trade.profit < 0 ? 'negative' : ''}">${trade.profit ? trade.profit.toFixed(2) : '-'}</td>
                            <td>${trade.status || '-'}</td>
                        </tr>`;
                    });
                    tradesTableBody.innerHTML = html;
                } else {
                    tradesTableBody.innerHTML = '<tr><td colspan="9" class="no-data">Brak historii transakcji</td></tr>';
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania historii transakcji:', error);
            const tradesTableBody = document.getElementById('trades-table-body');
            if (tradesTableBody) {
                tradesTableBody.innerHTML = '<tr><td colspan="9" class="no-data">Nie udało się pobrać historii transakcji</td></tr>';
            }
        });
}

function fetchSystemSettings() {
    // Funkcja pobierająca ustawienia systemu
    console.log("Ładowanie ustawień systemu...");
    // Tutaj można dodać rzeczywiste pobieranie ustawień z API
}

function fetchNotifications() {
    // Funkcja pobierająca powiadomienia
    console.log("Ładowanie powiadomień...");
    fetch('/api/notifications')
        .then(response => response.json())
        .catch(error => {
            console.error('Błąd podczas pobierania powiadomień:', error);
            const notificationsList = document.getElementById('notifications-list');
            if (notificationsList) {
                notificationsList.innerHTML = '<div class="no-data">Nie udało się pobrać powiadomień</div>';
            }
        });
}

// Funkcja uruchamiająca symulację
function runSimulation() {
    const initialCapital = document.getElementById('initial-capital').value;
    const duration = document.getElementById('duration').value;
    const withLearning = document.getElementById('with-learning').checked;
    const iterations = document.getElementById('iterations').value;

    const simulationData = {
        initial_capital: initialCapital,
        duration: duration,
        with_learning: withLearning,
        iterations: withLearning ? iterations : 1
    };

    fetch('/api/simulation/run', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(simulationData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('success', 'Symulacja uruchomiona pomyślnie');
            setTimeout(() => fetchSimulationResults(), 2000);
        } else {
            showNotification('error', data.error || 'Błąd podczas uruchamiania symulacji');
        }
    })
    .catch(error => {
        console.error('Błąd podczas uruchamiania symulacji:', error);
        showNotification('error', 'Błąd podczas uruchamiania symulacji');
    });
}

// Funkcja do pobierania danych portfela
function fetchPortfolioData() {
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            // Pobierz element, który ma być aktualizowany
            const portfolioElement = document.getElementById('portfolio-data');
            if (!portfolioElement) {
                console.error("Element 'portfolio-data' nie istnieje");
                return;
            }

            // Sprawdź czy dane są dostępne
            if (data && data.balances) {
                // Wyczyść obecną zawartość
                portfolioElement.innerHTML = '';

                // Utwórz nagłówek tabeli
                const table = document.createElement('table');
                table.className = 'portfolio-table';
                table.innerHTML = `
                    <thead>
                        <tr>
                            <th>Waluta</th>
                            <th>Kapitał</th>
                            <th>Dostępne</th>
                            <th>Akcje</th>
                        </tr>
                    </thead>
                    <tbody id="portfolio-body"></tbody>
                `;
                portfolioElement.appendChild(table);

                const tbody = document.getElementById('portfolio-body');

                // Dodaj wiersze dla każdej waluty
                Object.keys(data.balances).forEach(currency => {
                    const balance = data.balances[currency];
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${currency}</td>
                        <td>${parseFloat(balance.equity).toFixed(6)}</td>
                        <td>${parseFloat(balance.available_balance).toFixed(6)}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary" 
                                    onclick="setBalance(${parseFloat(balance.equity).toFixed(6)}, '${currency}')">
                                Ustaw
                            </button>
                        </td>
                    `;
                    tbody.appendChild(row);
                });

                // Dodaj formularz do ustawienia nowego salda
                const formRow = document.createElement('div');
                formRow.className = 'mt-3';
                formRow.innerHTML = `
                    <form id="set-balance-form" class="row g-3 align-items-center">
                        <div class="col-auto">
                            <input type="number" step="0.01" min="0" class="form-control" id="balance-amount" placeholder="Kwota" required>
                        </div>
                        <div class="col-auto">
                            <input type="text" class="form-control" id="balance-currency" placeholder="USDT" value="USDT" required>
                        </div>
                        <div class="col-auto">
                            <button type="submit" class="btn btn-primary">Ustaw nowe saldo</button>
                        </div>
                    </form>
                `;
                portfolioElement.appendChild(formRow);

                // Dodaj obsługę formularza
                document.getElementById('set-balance-form').addEventListener('submit', function(e) {
                    e.preventDefault();
                    const amount = document.getElementById('balance-amount').value;
                    const currency = document.getElementById('balance-currency').value;
                    setBalance(amount, currency);
                });

            } else {
                portfolioElement.innerHTML = '<p>Nie można pobrać danych portfela.</p>';
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych portfela:', error);
            const portfolioElement = document.getElementById('portfolio-data');
            if (portfolioElement) {
                portfolioElement.innerHTML = '<p>Wystąpił błąd podczas pobierania danych portfela.</p>';
            }
        });
}

// Funkcja do ustawiania nowego saldo
function setBalance(amount, currency) {
    fetch('/api/portfolio/set-balance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            amount: amount,
            currency: currency
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`Saldo zostało zaktualizowane na ${amount} ${currency}`);
            fetchPortfolioData(); // Odśwież dane portfela
        } else {
            alert(`Błąd: ${data.error || 'Nie udało się zaktualizować salda'}`);
        }
    })
    .catch(error => {
        console.error('Błąd podczas ustawiania salda:', error);
        alert('Wystąpił błąd podczas ustawiania salda');
    });
}

// Funkcja do pobierania danych sentymentu
function fetchSentimentData() {
    fetch('/api/sentiment')
        .then(response => response.json())
        .then(data => {
            // Pobierz element, który ma być aktualizowany
            const sentimentElement = document.getElementById('sentiment-data');
            if (!sentimentElement) {
                return; // Element może nie istnieć w niektórych widokach
            }

            // Wyczyść obecną zawartość
            sentimentElement.innerHTML = '';

            if (data) {
                // Dodaj nagłówek
                const header = document.createElement('h4');
                header.textContent = 'Sentyment rynkowy';
                header.className = 'mb-3';
                sentimentElement.appendChild(header);

                // Dodaj wartość sentymentu
                const sentimentValue = document.createElement('div');
                sentimentValue.className = 'sentiment-value mb-2';

                // Określ klasę na podstawie wartości
                let sentimentClass = 'neutral';
                if (data.value > 0.2) sentimentClass = 'positive';
                if (data.value < -0.2) sentimentClass = 'negative';

                sentimentValue.innerHTML = `
                    <span class="sentiment-label">Wartość sentymentu:</span>
                    <span class="sentiment-score ${sentimentClass}">${data.value.toFixed(2)}</span>
                    <span class="sentiment-analysis">(${data.analysis})</span>
                `;
                sentimentElement.appendChild(sentimentValue);

                // Dodaj źródła sentymentu
                if (data.sources) {
                    const sourcesContainer = document.createElement('div');
                    sourcesContainer.className = 'sources-container';

                    const sourcesHeader = document.createElement('h5');
                    sourcesHeader.textContent = 'Źródła danych';
                    sourcesHeader.className = 'mt-3 mb-2';
                    sourcesContainer.appendChild(sourcesHeader);

                    const sourcesList = document.createElement('ul');
                    sourcesList.className = 'sources-list';

                    Object.keys(data.sources).forEach(source => {
                        const value = data.sources[source];
                        const sourceItem = document.createElement('li');

                        // Określ klasę na podstawie wartości
                        let sourceClass = 'neutral';
                        if (value > 0.2) sourceClass = 'positive';
                        if (value < -0.2) sourceClass = 'negative';

                        sourceItem.innerHTML = `
                            <span class="source-name">${source}:</span>
                            <span class="source-value ${sourceClass}">${value.toFixed(2)}</span>
                        `;
                        sourcesList.appendChild(sourceItem);
                    });

                    sourcesContainer.appendChild(sourcesList);
                    sentimentElement.appendChild(sourcesContainer);
                }

                // Dodaj datę aktualizacji
                if (data.timestamp) {
                    const timestampElem = document.createElement('div');
                    timestampElem.className = 'timestamp mt-3 text-muted';
                    timestampElem.textContent = `Ostatnia aktualizacja: ${data.timestamp}`;
                    sentimentElement.appendChild(timestampElem);
                }

                // Dodaj przycisk odświeżania
                const refreshButton = document.createElement('button');
                refreshButton.className = 'btn btn-sm btn-outline-secondary mt-3';
                refreshButton.textContent = 'Odśwież dane sentymentu';
                refreshButton.onclick = fetchSentimentData;
                sentimentElement.appendChild(refreshButton);
            } else {
                sentimentElement.innerHTML = '<p>Nie można pobrać danych sentymentu.</p>';
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych sentymentu:', error);
            const sentimentElement = document.getElementById('sentiment-data');
            if (sentimentElement) {
                sentimentElement.innerHTML = '<p>Wystąpił błąd podczas pobierania danych sentymentu.</p>';
            }
        });
}

// Pobieranie statusu komponentów
function fetchComponentStatus() {
    fetch('/api/component-status')
        .then(response => response.json())
        .then(data => {
            // Aktualizacja statusu API
            updateComponentStatus('api-connector', data.api);
            // Aktualizacja statusu silnika handlowego
            updateComponentStatus('trading-engine', data.trading_engine);
            // Aktualizacja statusu managera ryzyka
            updateComponentStatus('risk-manager', data.portfolio || 'unknown');
            // Aktualizacja statusu procesora danych
            updateComponentStatus('data-processor', data.api === 'online' ? 'online' : 'warning');
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusu komponentów:', error);
            // Ustaw wszystkie komponenty jako offline w przypadku błędu
            updateComponentStatus('api-connector', 'offline');
            updateComponentStatus('trading-engine', 'offline');
            updateComponentStatus('risk-manager', 'offline');
            updateComponentStatus('data-processor', 'offline');
        });
}

// Pobieranie danych portfela
function fetchPortfolioData() {
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('portfolio-container');
            if (!container) {
                console.warn("Element 'portfolio-data' nie istnieje");
                return;
            }

            if (data.success && data.balances) {
                let html = '';
                for (const [currency, balance] of Object.entries(data.balances)) {
                    html += `
                    <div class="portfolio-item">
                        <div class="coin-name">${currency}</div>
                        <div class="coin-balance">Balans: ${balance.wallet_balance}</div>
                        <div class="coin-value">Dostępne: ${balance.available_balance}</div>
                    </div>`;
                }
                container.innerHTML = html;
            } else {
                container.innerHTML = `
                <div class="no-data">Brak danych portfela lub problem z połączeniem z ByBit.</div>
                <div class="error-details">${data.error || ''}</div>`;
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych portfela:', error);
            const container = document.getElementById('portfolio-container');
            if (container) {
                container.innerHTML = `
                <div class="no-data">Błąd podczas ładowania danych portfela.</div>
                <div class="error-details">${error.message}</div>`;
            }
        });
}

// Funkcja pomocnicza do aktualizacji statusu komponentu
function updateComponentStatus(elementId, status) {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Usuń wszystkie klasy statusu
    element.classList.remove('status-online', 'status-offline', 'status-warning');

    // Dodaj odpowiednią klasę na podstawie statusu
    if (status === 'online') {
        element.classList.add('status-online');
        element.querySelector('.status-text').textContent = 'Online';
    } else if (status === 'offline') {
        element.classList.add('status-offline');
        element.querySelector('.status-text').textContent = 'Offline';
    } else {
        element.classList.add('status-warning');
        element.querySelector('.status-text').textContent = 'Warning';
    }
}

// Funkcja pobierająca status modeli AI
function fetchAIStatus() {
    fetch('/api/ai-models-status')
        .then(response => response.json())
        .then(data => {
            updateAIModelsStatus(data);
            updateLastRefreshed();
        })
        .catch(error => {
            console.log("Błąd podczas pobierania statusu modeli AI:", error);
            showNotification('error', 'Nie udało się pobrać statusu modeli AI');
        });
}

// Aktualizacja statusu modeli AI
function updateAIModelsStatus(data) {
    const container = document.getElementById('ai-models-status');
    if (!container) return;

    // Wyczyść kontener
    container.innerHTML = '';

    if (!data || !data.models || data.models.length === 0) {
        container.innerHTML = '<div class="no-data">Brak dostępnych modeli AI</div>';
        return;
    }

    // Dla każdego modelu
    data.models.forEach(model => {
        const modelItem = document.createElement('div');
        modelItem.className = `model-item ${model.status === 'active' ? 'model-active' : 'model-inactive'}`;

        modelItem.innerHTML = `
            <div class="model-name">${model.name}</div>
            <div class="model-type">${model.type || 'Nieznany'}</div>
            <div class="model-status">${model.status === 'active' ? 'Aktywny' : 'Nieaktywny'}</div>
            <div class="model-accuracy">${model.accuracy ? model.accuracy + '%' : 'N/A'}</div>
        `;

        container.appendChild(modelItem);
    });
}

// Flaga do śledzenia, czy funkcja już jest wykonywana
let isInitializing = false;
let isInitialized = false;

// Funkcja inicjalizująca, wywoływana po załadowaniu strony
function initDashboard() {
    // Zapobiegamy wielokrotnemu uruchomieniu
    if (isInitializing || isInitialized) {
        console.log("Dashboard już jest inicjalizowany lub zainicjalizowany");
        return;
    }

    isInitializing = true;
    console.log("Aktualizacja danych dashboardu...");

    // Pobierz dane startowe (po jednym, z opóźnieniem, aby uniknąć zawieszenia)
    setTimeout(() => fetchComponentStatus(), 500);
    setTimeout(() => fetchPortfolioData(), 1000);
    setTimeout(() => fetchSentimentData(), 1500);
    setTimeout(() => fetchAIStatus(), 2000);
    setTimeout(() => fetchSimulationResults(), 2500);
    setTimeout(() => fetchAIThoughts(), 3000);
    setTimeout(() => updateChartData(), 3500);

    // Ustaw interwały dla automatycznego odświeżania (różne czasy, aby rozłożyć obciążenie)
    setTimeout(() => {
        // Sprawdzamy przed ustawieniem interwału, czy strona jest aktywna
        if (appState.activeDashboard) {
            setInterval(() => {
                if (appState.activeDashboard) fetchComponentStatus();
            }, 11000); // Co 11 sekund

            setInterval(() => {
                if (appState.activeDashboard) fetchPortfolioData();
            }, 31000);   // Co 31 sekund

            setInterval(() => {
                if (appState.activeDashboard) fetchSentimentData();
            }, 61000);   // Co 61 sekund

            setInterval(() => {
                if (appState.activeDashboard) fetchAIStatus();
            }, 33000);   // Co 33 sekundy

            setInterval(() => {
                if (appState.activeDashboard) fetchSimulationResults();
            }, 63000);   // Co 63 sekundy

            setInterval(() => {
                if (appState.activeDashboard) updateChartData();
            }, 65000);   // Co 65 sekund
        }

        isInitializing = false;
        isInitialized = true;
        console.log("Dashboard załadowany");
    }, 4000);
}

// Wywołaj funkcję inicjalizującą po załadowaniu strony
document.addEventListener('DOMContentLoaded', initDashboard);

// Funkcja do pobierania wyników symulacji
function fetchSimulationResults() {
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

// Funkcja do pobierania myśli AI
function fetchAIThoughts() {
    fetch('/api/ai/thoughts')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Jeśli istnieje specyficzna obsługa w dashboardzie, dodaj ją tutaj
            console.log("Pobrano myśli AI:", data);
        })
        .catch(error => {
            console.error("Błąd podczas pobierania myśli AI:", error);
        });
}

// Funkcja do pobierania statusu AI
function fetchAIStatus() {
    fetch('/api/ai-models-status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateAIModelsStatus(data);
            updateLastRefreshed();
        })
        .catch(error => {
            console.log("Błąd podczas pobierania statusu modeli AI:", error);
            showNotification('error', 'Nie udało się pobrać statusu modeli AI');
        });
}