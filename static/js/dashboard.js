// Konfiguracja aplikacji
const CONFIG = {
    // Ogólne ustawienia
    updateInterval: 30000,              // 30 sekund - podstawowy interwał aktualizacji
    chartUpdateInterval: 60000,         // 1 minuta - interwał aktualizacji wykresów
    maxErrors: 3,                       // Maksymalna liczba błędów przed pokazaniem powiadomienia
    debugMode: false,                   // Tryb debugowania (więcej logów)

    // Wszystkie endpointy API w jednym miejscu (dla łatwej zmiany i konsystencji)
    apiEndpoints: {
        // Autoryzacja
        auth: {
            login: '/api/auth/login',
            verify: '/api/auth/verify',
            register: '/api/auth/register',
            logout: '/api/auth/logout'
        },
        
        // Główne dane dashboardu
        dashboard: '/api/dashboard/data',
        componentStatus: '/api/component-status',
        
        // Portfolio i wykresy
        portfolio: '/api/portfolio',
        setBalance: '/api/portfolio/set-balance',
        chartData: '/api/chart/data',
        
        // Handel
        trading: {
            start: '/api/trading/start',
            stop: '/api/trading/stop',
            status: '/api/trading/status',
            history: '/api/trades/history'
        },
        
        // Zarządzanie systemem
        system: {
            reset: '/api/system/reset',
            settings: '/api/system/settings',
            notifications: '/api/notifications',
            status: '/api/system/status'
        },
        
        // Modele AI i symulacje
        ai: {
            modelsStatus: '/api/ai-models-status',
            thoughts: '/api/ai/thoughts',
            learningStatus: '/api/ai/learning-status',
            train: '/api/ai/train'
        },
        
        // Symulacje i wyniki
        simulation: {
            run: '/api/simulation/run',
            results: '/api/simulation-results',
            learn: '/api/simulation/learn'
        },
        
        // Analiza sentymentu
        sentiment: {
            main: '/api/sentiment',           // Główny endpoint dla sentymentu
            latest: '/api/sentiment/latest',   // Endpoint dla najnowszych danych sentymentu
            history: '/api/sentiment/history'  // Historia sentymentu
        }
    }
};

// Konfiguracja retry mechanizmu
const RETRY_CONFIG = {
    maxRetries: 3,
    initialDelay: 1000, // 1 sekunda
    maxDelay: 5000, // 5 sekund
    backoffFactor: 2,
    retryableStatuses: [502, 503, 504]
};

// Funkcja pomocnicza do wykonywania fetch z automatycznymi ponownymi próbami
async function fetchWithRetry(url, options = {}) {
    let attempt = 0;
    let delay = RETRY_CONFIG.initialDelay;
    
    // Wyodrębnij nazwę endpointu z URL do celów diagnostycznych
    const endpointName = url.split('/').pop().replace(/[^a-zA-Z]/g, '') || 'unknown';
    
    // Sprawdź, czy żądanie jest już w toku - zapobiegaj duplikatom
    if (appState.isRequestInProgress(endpointName)) {
        console.log(`Żądanie do ${endpointName} jest już w toku, pomijam duplikat`);
        return new Promise((resolve) => {
            // Poczekaj chwilę i sprawdź, czy dane są dostępne
            setTimeout(() => {
                // Możemy zwrócić success response z pustym body
                resolve(new Response('{}', { 
                    status: 200, 
                    headers: { 'Content-Type': 'application/json' }
                }));
            }, 500);
        });
    }
    
    // Oznacz żądanie jako w toku
    appState.startRequest(endpointName);
    
    try {
        while (attempt < RETRY_CONFIG.maxRetries) {
            try {
                const response = await fetch(url, options);
                
                // Oznacz żądanie jako zakończone
                appState.endRequest(endpointName);
                
                // Jeśli status nie jest retryable, zwróć odpowiedź
                if (!RETRY_CONFIG.retryableStatuses.includes(response.status)) {
                    return response;
                }
                
                // Pobierz dane o błędzie jeśli to możliwe
                let errorData = {};
                try {
                    errorData = await response.clone().json();
                } catch (e) {
                    // Ignoruj błędy parsowania JSON
                }
                
                // Jeśli to błąd 502, 503, 504, spróbuj ponownie
                attempt++;
                if (attempt < RETRY_CONFIG.maxRetries) {
                    // Pobierz czas oczekiwania z nagłówka odpowiedzi lub użyj domyślnego
                    const retryAfter = response.headers.get('Retry-After');
                    const waitTime = retryAfter ? parseInt(retryAfter) * 1000 : delay;
                    
                    // Pokaż informację o ponownej próbie
                    const componentName = errorData.component || 'serwera';
                    showNotification('warning', 
                        `Problem z połączeniem do ${componentName}. ` +
                        `Ponowna próba za ${Math.round(waitTime/1000)} sekund... (${attempt}/${RETRY_CONFIG.maxRetries})`
                    );
                    
                    await new Promise(resolve => setTimeout(resolve, waitTime));
                    delay = Math.min(delay * RETRY_CONFIG.backoffFactor, RETRY_CONFIG.maxDelay);
                    continue;
                }
                
                // Jeśli wszystkie próby się nie powiodły, wywołaj handleApiError i zwróć odpowiedź
                handleApiError(endpointName, new Error(`Wszystkie próby (${RETRY_CONFIG.maxRetries}) nie powiodły się`), response);
                return response;
                
            } catch (error) {
                // Obsługa błędów sieciowych (np. brak połączenia)
                if (attempt === RETRY_CONFIG.maxRetries - 1) {
                    // Oznacz żądanie jako zakończone w przypadku błędu
                    appState.endRequest(endpointName);
                    
                    console.error(`Ostateczny błąd po ${RETRY_CONFIG.maxRetries} próbach połączenia z ${url}:`, error);
                    handleApiError(endpointName, error);
                    throw error;
                }
                
                attempt++;
                
                // Pokaż powiadomienie o problemie z siecią
                showNotification('warning',
                    `Problem z połączeniem sieciowym. ` +
                    `Ponowna próba za ${delay/1000} sekund... (${attempt}/${RETRY_CONFIG.maxRetries})`
                );
                
                await new Promise(resolve => setTimeout(resolve, delay));
                delay = Math.min(delay * RETRY_CONFIG.backoffFactor, RETRY_CONFIG.maxDelay);
            }
        }
    } finally {
        // Zawsze oznacz żądanie jako zakończone, nawet w przypadku wyjątku
        appState.endRequest(endpointName);
    }
    
    // Zabezpieczenie na wypadek wyjścia z pętli bez zwrócenia odpowiedzi
    throw new Error(`Nie udało się wykonać zapytania do ${url} po ${RETRY_CONFIG.maxRetries} próbach.`);
}

// Funkcja do pobierania statusu komponentów
function fetchComponentStatus() {
    fetchWithRetry('/api/component-status', {
        headers: getAuthHeaders()
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
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
                
                // Aktualizacja ostatniego czasu odświeżenia danych
                updateLastRefreshed();
            } else {
                console.error('Błąd podczas pobierania statusu komponentów:', data.error);
                showNotification('warning', 'Nie udało się pobrać statusu komponentów');
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania statusu komponentów:', error);
            showNotification('error', 'Problem z połączeniem do serwera');
            handleApiError('componentStatus');
        });
}

// Funkcja do pobierania statusu modeli AI
function fetchAIStatus() {
    fetchWithRetry(CONFIG.apiEndpoints.ai.modelsStatus, {
        headers: getAuthHeaders()
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Zapisz dane w stanie aplikacji
            appState.updateData('aiModelsData', data);
            
            // Zaktualizuj interfejs
            updateAIModelsStatus(data);
            
            // Zaaktualizuj również status odświeżenia
            updateLastRefreshed();
        })
        .catch(error => {
            console.error("Błąd podczas pobierania statusu modeli AI:", error);
            
            // Pobierz kontener modeli AI
            const aiModelsContainer = findElement(['#ai-models-container', '#ai-models-status'], 'ai-models-container');
            
            if (aiModelsContainer) {
                aiModelsContainer.innerHTML = `
                    <div class="error-message">
                        <h4>Błąd podczas pobierania statusu modeli AI</h4>
                        <p>${error.message || 'Problem z połączeniem z serwerem'}</p>
                        <button class="retry-button" onclick="fetchAIStatus()">Spróbuj ponownie</button>
                    </div>
                `;
            }
            
            showNotification('error', 'Nie udało się pobrać statusu modeli AI');
            
            // Zarejestruj błąd
            handleApiError('aiModels', error);
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
    // Ustawienia ogólne
    activeDashboard: true,  // Flaga aktywności dashboardu (gdy karta jest widoczna)
    isAuthenticated: false, // Status uwierzytelnienia
    token: null,            // Token uwierzytelniający
    
    // Liczniki i dane diagnostyczne
    errorCounts: {},       // Liczniki błędów dla różnych endpointów
    retryDelays: {},       // Opóźnienia dla mechanizmu ponownych prób
    requestsInProgress: {}, // Flagi dla równoległych żądań, aby uniknąć duplikacji
    
    // Dane aplikacji
    portfolioData: null,   // Dane portfela
    dashboardData: null,   // Główne dane dashboardu
    sentimentData: null,   // Dane sentymentu
    aiModelsData: null,    // Dane modeli AI
    simulationData: null,  // Dane symulacji
    
    // Znaczniki czasu ostatnich aktualizacji
    lastUpdated: {
        portfolio: 0,
        dashboard: 0,
        chart: 0,
        components: 0,
        sentiment: 0,
        aiModels: 0,
        simulation: 0
    },
    
    // Referencje do obiektów
    portfolioChart: null,  // Referencja do wykresu portfela
    
    // Metody pomocnicze do zarządzania stanem
    updateData: function(key, data) {
        this[key] = data;
        this.lastUpdated[key] = Date.now();
    },
    
    isStale: function(key, maxAge = 30000) {
        return Date.now() - (this.lastUpdated[key] || 0) > maxAge;
    },
    
    startRequest: function(endpointName) {
        this.requestsInProgress[endpointName] = true;
    },
    
    endRequest: function(endpointName) {
        this.requestsInProgress[endpointName] = false;
    },
    
    isRequestInProgress: function(endpointName) {
        return this.requestsInProgress[endpointName] === true;
    }
};

// Inicjalizacja po załadowaniu dokumentu
document.addEventListener('DOMContentLoaded', function() {
    console.log("Dashboard załadowany");
    // Wymuś pokazanie formularza logowania przy starcie
    const token = localStorage.getItem('token');
    if (!token) {
        showLoginForm();
        return;
    }

    // Sprawdź ważność tokenu tylko jeśli istnieje
    fetch(CONFIG.apiEndpoints.auth.verify, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Token invalid');
        }
        return response.json();
    })
    .then(data => {
        if (data.valid) {
            appState.token = token;
            appState.isAuthenticated = true;
            initializeUI();
        } else {
            localStorage.removeItem('token');
            showLoginForm();
        }
    })
    .catch(() => {
        localStorage.removeItem('token');
        showLoginForm();
    });
});

// Inicjalizacja interfejsu użytkownika
function initializeUI() {
    // Pokaż główną treść dashboardu
    const dashboardContent = document.getElementById('dashboard-content');
    if (dashboardContent) {
        dashboardContent.style.display = 'block';
    }

    // Ukryj kontener autoryzacji
    const authContainer = document.getElementById('auth-container');
    if (authContainer) {
        authContainer.style.display = 'none';
    }

    // Inicjalizacja nawigacji zakładek
    setupTabNavigation();
    
    // Dodaj przycisk wylogowania
    const header = document.querySelector('header') || document.body;
    const logoutBtn = document.getElementById('logout-btn');
    if (!logoutBtn) {
        const newLogoutBtn = document.createElement('button');
        newLogoutBtn.textContent = 'Wyloguj';
        newLogoutBtn.className = 'btn btn-outline-secondary';
        newLogoutBtn.id = 'logout-btn';
        newLogoutBtn.onclick = logout;
        header.appendChild(newLogoutBtn);
    } else {
        logoutBtn.onclick = logout;
    }

    // Inicjalizacja wykresu portfela
    initializePortfolioChart();

    // Pobierz początkowe dane
    updateDashboardData();
    updateComponentStatus();
    updatePortfolioData();
    updateAIModelsStatus();
    
    // Uruchom aktualizacje danych
    startDataUpdates();
    
    // Dodaj obsługę eventów dla przycisków
    setupEventListeners();
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
    fetch('/api/ai-models-status', {
        headers: getAuthHeaders()
    })
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

    fetch('/api/chart-data', {
        headers: getAuthHeaders()
    })
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
    // Użyjemy naszej nowej funkcji fetchWithRetry dla lepszej obsługi błędów
    fetchWithRetry(CONFIG.apiEndpoints.portfolio, {
        headers: getAuthHeaders()
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Zapamiętaj dane w stanie aplikacji
            appState.updateData('portfolioData', data);
            
            // Znajdź kontener portfolio za pomocą naszej nowej funkcji pomocniczej
            // Szukamy elementu za pomocą różnych selektorów dla kompatybilności
            const portfolioContainer = findElement([
                '#portfolio-container', 
                '#portfolio-data', 
                '.portfolio-data'
            ], 'portfolio-container');

            if (!portfolioContainer) {
                console.error("Nie udało się znaleźć ani utworzyć kontenera portfolio");
                return;
            }

            // Aktualizacja kontenera danymi
            if (data && data.success === true && data.balances && Object.keys(data.balances).length > 0) {
                // Przygotuj zawartość HTML
                let portfolioHTML = '<div class="portfolio-summary">';
                
                // Dodaj nagłówek z podsumowaniem
                const total = Object.values(data.balances).reduce(
                    (sum, bal) => sum + (parseFloat(bal.equity) || 0), 0
                ).toFixed(2);
                
                portfolioHTML += `<div class="portfolio-total">Całkowita wartość: <span class="value">${total} USD</span></div>`;
                portfolioHTML += '</div><div class="portfolio-assets">';

                // Przetwarzanie danych portfela
                for (const [currency, details] of Object.entries(data.balances)) {
                    if (details) {
                        // Bezpieczne wyświetlanie wartości z obsługą wartości null/undefined
                        const equity = typeof details.equity === 'number' ? details.equity.toFixed(4) : '0.0000';
                        const available = typeof details.available_balance === 'number' ? details.available_balance.toFixed(4) : '0.0000';
                        const wallet = typeof details.wallet_balance === 'number' ? details.wallet_balance.toFixed(4) : '0.0000';
                        
                        // Oblicz zmianę procentową jeśli dostępna
                        let changeClass = '';
                        let changeHTML = '';
                        if (details.change_24h) {
                            changeClass = details.change_24h > 0 ? 'positive' : 'negative';
                            changeHTML = `<div class="balance-change ${changeClass}">${details.change_24h > 0 ? '+' : ''}${details.change_24h.toFixed(2)}%</div>`;
                        }

                        portfolioHTML += `
                            <div class="balance-item">
                                <div class="currency-header">
                                    <div class="currency">${currency}</div>
                                    ${changeHTML}
                                </div>
                                <div class="balance-details">
                                    <div class="balance-row"><span>Equity:</span> <span>${equity}</span></div>
                                    <div class="balance-row"><span>Available:</span> <span>${available}</span></div>
                                    <div class="balance-row"><span>Wallet:</span> <span>${wallet}</span></div>
                                </div>
                            </div>
                        `;
                    }
                }
                portfolioHTML += '</div>';
                
                // Aktualizuj kontener
                portfolioContainer.innerHTML = portfolioHTML;
                
                // Zaaktualizuj również status odświeżenia
                updateLastRefreshed();
            } else {
                // Pokaż komunikat o błędzie
                portfolioContainer.innerHTML = `
                    <div class="error-message">
                        <h4>Brak danych portfela</h4>
                        <p>Problem z połączeniem z ByBit. Sprawdź klucze API w ustawieniach.</p>
                        <div class="error-details">${data?.error || 'Nieznany błąd'}</div>
                        <button class="retry-button" onclick="updatePortfolioData()">Spróbuj ponownie</button>
                    </div>
                `;

                // Wyświetl dane diagnostyczne w logach konsoli
                console.log("Otrzymane dane portfela:", data);
            }
        })
        .catch(err => {
            console.error("Błąd podczas pobierania danych portfela:", err);
            
            // Znajdź kontener
            const portfolioContainer = findElement([
                '#portfolio-container', 
                '#portfolio-data', 
                '.portfolio-data'
            ], 'portfolio-container');
            
            if (portfolioContainer) {
                portfolioContainer.innerHTML = `
                    <div class="error-message">
                        <h4>Błąd podczas pobierania danych portfela</h4>
                        <p>${err.message || 'Problem z połączeniem internetowym'}</p>
                        <button class="retry-button" onclick="updatePortfolioData()">Spróbuj ponownie</button>
                    </div>
                `;
            }
            
            // Zarejestruj błąd w systemie obsługi błędów
            handleApiError('portfolio', err);
        });
}

// Aktualizacja głównych danych dashboardu
async function updateDashboardData() {
    try {
        const response = await fetchWithRetry('/api/dashboard/data', {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            if (response.status === 502) {
                const errorData = await response.json();
                showErrorMessage(
                    `Błąd połączenia z komponentem ${errorData.component}. ` +
                    `ID żądania: ${errorData.request_id}. ` +
                    `Spróbuj odświeżyć stronę lub sprawdź status serwera.`
                );
                // Dodaj próbę ponownego połączenia
                setTimeout(() => {
                    updateDashboardData();
                }, 5000); // Próba ponownego połączenia po 5 sekundach
                return;
            }
            throw new Error(`Błąd HTTP: ${response.status}`);
        }

        const data = await response.json();
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
    } catch (error) {
        console.error('Błąd podczas aktualizacji danych dashboardu:', error);
        handleApiError('dashboardData');
    }

    // Pobierz wyniki symulacji tradingu
    fetch('/api/simulation-results', {
        headers: getAuthHeaders()
    })
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
    // Użyj nowej struktury CONFIG.apiEndpoints
    fetchWithRetry(CONFIG.apiEndpoints.sentiment.main, {
        headers: getAuthHeaders()
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Zapisz dane w stanie aplikacji
            appState.updateData('sentimentData', data);
            
            // Aktualizuj kontener sentymentu z otrzymanymi danymi
            const sentimentContainer = findElement(['#sentiment-container'], 'sentiment-container');
            
            if (sentimentContainer && data) {
                if (data.value !== undefined && data.analysis) {
                    // Przygotuj HTML z danymi sentymentu
                    let sentimentHTML = `
                        <div class="sentiment-header">
                            <h3>Analiza Sentymentu Rynkowego</h3>
                            <span class="last-updated">Zaktualizowano: ${new Date().toLocaleTimeString()}</span>
                        </div>
                        <div class="sentiment-content">
                    `;
                    
                    // Określ klasę sentymentu na podstawie wartości
                    let sentimentClass = 'neutral';
                    if (data.value > 0.1) sentimentClass = 'positive';
                    if (data.value < -0.1) sentimentClass = 'negative';
                    
                    // Dodaj główną wartość sentymentu
                    sentimentHTML += `
                        <div class="sentiment-value-container">
                            <div class="sentiment-label">Ogólny sentyment:</div>
                            <div class="sentiment-value ${sentimentClass}">
                                ${data.analysis || 'Neutralny'} (${data.value.toFixed(2)})
                            </div>
                        </div>
                    `;
                    
                    // Dodaj źródła danych, jeśli są dostępne
                    if (data.sources) {
                        sentimentHTML += `<div class="sentiment-sources">
                            <h4>Źródła danych</h4>
                            <div class="sources-list">`;
                            
                        for (const [source, info] of Object.entries(data.sources)) {
                            const scoreClass = info.score > 0 ? 'positive' : info.score < 0 ? 'negative' : 'neutral';
                            sentimentHTML += `
                                <div class="source-item">
                                    <div class="source-name">${source}</div>
                                    <div class="source-score ${scoreClass}">${info.score.toFixed(2)}</div>
                                    <div class="source-volume">${info.volume} wzmianek</div>
                                </div>
                            `;
                        }
                        
                        sentimentHTML += `</div></div>`;
                    }
                    
                    // Dodaj przycisk odświeżania
                    sentimentHTML += `
                        </div>
                        <div class="sentiment-footer">
                            <button onclick="updateSentimentData()" class="btn btn-sm">Odśwież dane sentymentu</button>
                        </div>
                    `;
                    
                    // Aktualizuj kontener
                    sentimentContainer.innerHTML = sentimentHTML;
                } else {
                    sentimentContainer.innerHTML = '<div class="no-data">Brak danych sentymentu rynkowego</div>';
                }
            }
        })
        .catch(error => {
            console.error('Błąd podczas pobierania danych sentymentu:', error);
            
            // Użyj findElement zamiast bezpośredniego getElementById
            const sentimentContainer = findElement(['#sentiment-container'], 'sentiment-container');
            
            if (sentimentContainer) {
                sentimentContainer.innerHTML = '<div class="error-message">Błąd podczas pobierania danych sentymentu</div>';
                
                // Spróbuj użyć endpoint sentimentLatest jako zapasowy
                fetchWithRetry(CONFIG.apiEndpoints.sentiment.latest, {
                    headers: getAuthHeaders()
                })
                    .then(response => {
                        if (!response.ok) throw new Error('Backup endpoint failed');
                        return response.json();
                    })
                    .then(data => {
                        if (data && data.value !== undefined) {
                            sentimentContainer.innerHTML = `
                                <div class="sentiment-score">
                                    <div class="sentiment-label">Ogólny sentyment (dane zapasowe):</div>
                                    <div class="sentiment-value ${data.value > 0.1 ? 'positive' : data.value < -0.1 ? 'negative' : 'neutral'}">
                                        ${data.analysis || (data.value > 0.1 ? 'Pozytywny' : data.value < -0.1 ? 'Negatywny' : 'Neutralny')}
                                    </div>
                                </div>
                                <div class="sentiment-footer">
                                    <button onclick="updateSentimentData()" class="btn btn-sm">Odśwież</button>
                                </div>
                            `;
                            showNotification('info', 'Wyświetlono zapasowe dane sentymentu');
                        }
                    })
                    .catch(backupError => {
                        console.error('Błąd zapasowego endpointu sentymentu:', backupError);
                        
                        // Wyświetl dane zastępcze po krótkim opóźnieniu
                        setTimeout(() => {
                            sentimentContainer.innerHTML = `
                                <div class="sentiment-score">
                                    <div class="sentiment-label">Ogólny sentyment (dane testowe):</div>
                                    <div class="sentiment-value neutral">Neutralny</div>
                                </div>
                                <div class="sentiment-details">
                                    <p>Serwer sentymentu jest niedostępny. Wyświetlam dane testowe.</p>
                                    <button onclick="updateSentimentData()" class="btn btn-sm">Odśwież</button>
                                </div>
                            `;
                        }, 2000);
                    });
            }
            
            // Zgłoś błąd do systemu obsługi błędów
            handleApiError('sentiment', error);
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
    fetch('/api/component-status', {
        headers: getAuthHeaders()
    })
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

// Uniwersalny system powiadomień
function showNotification(type, message, duration = 5000) {
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
    
    // Dodaj ikonę zależnie od typu powiadomienia
    const icon = document.createElement('span');
    icon.className = 'notification-icon';
    
    switch(type) {
        case 'success':
            icon.innerHTML = '✓';
            break;
        case 'error':
            icon.innerHTML = '✗';
            break;
        case 'warning':
            icon.innerHTML = '⚠';
            break;
        case 'info':
        default:
            icon.innerHTML = 'ℹ';
            break;
    }
    
    notification.appendChild(icon);
    
    // Dodaj tekst powiadomienia
    const text = document.createElement('span');
    text.className = 'notification-text';
    text.textContent = message;
    notification.appendChild(text);
    
    // Dodaj przycisk zamknięcia
    const closeBtn = document.createElement('span');
    closeBtn.className = 'notification-close';
    closeBtn.innerHTML = '×';
    closeBtn.addEventListener('click', () => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            if (container.contains(notification)) {
                container.removeChild(notification);
            }
        }, 300);
    });
    notification.appendChild(closeBtn);

    // Dodaj do kontenera powiadomień
    container.appendChild(notification);

    // Usuń po określonym czasie
    setTimeout(() => {
        if (container.contains(notification)) {
            notification.classList.add('fade-out');
            setTimeout(() => {
                if (container.contains(notification)) {
                    container.removeChild(notification);
                }
            }, 300);
        }
    }, duration);
    
    // Zapisz informację o powiadomieniu w logu
    if (type === 'error') {
        console.error(message);
    } else if (type === 'warning') {
        console.warn(message);
    } else {
        console.log(message);
    }
    
    return notification;
}

// Jednolita obsługa błędów API z retry mechanizmem
function handleApiError(endpoint, error, response) {
    // Zwiększ licznik błędów dla danego endpointu
    appState.errorCounts[endpoint] = (appState.errorCounts[endpoint] || 0) + 1;
    
    // Zapisz informację o błędzie w konsoli
    console.error(`Błąd API (${endpoint}):`, error);
    
    // Ustal typ błędu i odpowiednią wiadomość
    let errorMsg = 'Wystąpił nieoczekiwany błąd';
    let shouldRetry = false;
    let retryDelay = 5000; // Domyślne opóźnienie ponownej próby
    
    // Analizuj odpowiedź HTTP, jeśli istnieje
    if (response) {
        if (response.status === 401 || response.status === 403) {
            errorMsg = 'Brak autoryzacji lub sesja wygasła. Zaloguj się ponownie.';
            // Możemy tutaj wylogować użytkownika
            logout();
            return;
        } 
        else if (response.status === 404) {
            errorMsg = `Zasób nie został znaleziony (${endpoint})`;
        }
        else if (response.status === 429) {
            errorMsg = 'Przekroczony limit zapytań. Spróbuj ponownie za chwilę.';
            shouldRetry = true;
            retryDelay = 10000; // Większe opóźnienie dla rate limit
        }
        else if (response.status >= 500) {
            errorMsg = `Problem z serwerem (${response.status}). Spróbuj ponownie za chwilę.`;
            shouldRetry = true;
            
            // Sprawdź czy jest informacja o opóźnieniu w nagłówku
            const retryAfter = response.headers?.get('Retry-After');
            if (retryAfter) {
                retryDelay = parseInt(retryAfter) * 1000;
            }
        }
    }
    
    // Jeśli przekroczono limit błędów, pokaż bardziej szczegółowy komunikat
    if (appState.errorCounts[endpoint] >= CONFIG.maxErrors) {
        errorMsg = `Powtarzający się problem z ${endpoint}. Odśwież stronę lub skontaktuj się z administratorem.`;
        showErrorMessage(errorMsg, false); // Nie ukrywaj automatycznie poważnych błędów
    } else {
        // Dla mniej krytycznych błędów pokaż zwykłe powiadomienie
        showNotification('error', errorMsg);
    }
    
    // Automatyczna ponowna próba dla niektórych typów błędów
    if (shouldRetry) {
        // Zapisz informację o zaplanowanej ponownej próbie
        appState.retryDelays[endpoint] = retryDelay;
        
        // Pokaż informację o ponownej próbie
        showNotification('info', `Zaplanowano ponowną próbę połączenia za ${retryDelay/1000} sekund`);
        
        // Zaplanuj ponowną próbę
        setTimeout(() => {
            // Sprawdź, czy dashboard jest nadal aktywny
            if (appState.activeDashboard) {
                showNotification('info', `Ponowne połączenie z ${endpoint}...`);
                
                // Wywołaj odpowiednią funkcję zależnie od endpointu
                switch (endpoint) {
                    case 'componentStatus':
                        fetchComponentStatus();
                        break;
                    case 'portfolio':
                        fetchPortfolioData();
                        break;
                    case 'aiModelsStatus':
                        fetchAIStatus();
                        break;
                    case 'chartData':
                        updateChartData();
                        break;
                    case 'dashboardData':
                        updateDashboardData();
                        break;
                    case 'simulationResults':
                        fetchSimulationResults();
                        break;
                    case 'sentiment':
                        updateSentimentData();
                        break;
                    default:
                        // Dla nieznanych endpointów nie robimy nic
                        console.log(`Brak obsługi automatycznego retry dla endpointu: ${endpoint}`);
                }
            }
        }, retryDelay);
    }
}

// Ujednolicona funkcja wyświetlania błędu
function showErrorMessage(message, autoHide = true, duration = 10000) {
    // Pokaż w kontenerze błędów, jeśli istnieje
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');
    
    if (errorContainer && errorText) {
        errorText.textContent = message;
        errorContainer.style.display = 'flex';
        
        // Dodaj przycisk ponowienia próby
        const retryButton = errorContainer.querySelector('.retry-button');
        if (!retryButton && autoHide) {
            const newRetryButton = document.createElement('button');
            newRetryButton.className = 'retry-button';
            newRetryButton.textContent = 'Odśwież dane';
            newRetryButton.onclick = () => {
                hideError();
                
                // Pobierz dane ponownie
                fetchComponentStatus();
                updateDashboardData();
                
                showNotification('info', 'Odświeżanie danych...');
            };
            
            // Znajdź element, do którego możemy dodać przycisk
            const buttonContainer = errorContainer.querySelector('.error-message') || errorContainer;
            buttonContainer.appendChild(newRetryButton);
        }
        
        if (autoHide) {
            setTimeout(hideError, duration);
        }
    } else {
        // Jeśli nie ma kontenera błędów, użyj systemu powiadomień
        showNotification('error', message, duration);
    }
}

// Funkcje zarządzania stanem tradingu
function startTrading() {
    fetch('/api/trading/start', {
        method: 'POST',
        headers: getAuthHeaders()
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
        headers: getAuthHeaders()
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
        headers: getAuthHeaders()
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
    fetch('/api/trades/history', {
        headers: getAuthHeaders()
    })
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
    fetch('/api/notifications', {
        headers: getAuthHeaders()
    })
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
        headers: getAuthHeaders(),
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
    fetch('/api/portfolio', {
        headers: getAuthHeaders()
    })
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
    fetch(CONFIG.apiEndpoints.portfolio.setBalance, {
        method: 'POST',
        headers: getAuthHeaders(),
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
    fetch('/api/sentiment', {
        headers: getAuthHeaders()
    })
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
    fetchWithRetry(CONFIG.apiEndpoints.componentStatus, {
        headers: getAuthHeaders()
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
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
    fetch(CONFIG.apiEndpoints.portfolio, {
        headers: getAuthHeaders()
    })
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
    fetchWithRetry(CONFIG.apiEndpoints.ai.modelsStatus, {
        headers: getAuthHeaders()
    })
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
    console.log("Inicjalizacja dashboardu...");

    // Pobierz początkowe dane wstępne z różnymi odstępami, aby nie przeciążać serwera
    const initialLoads = [
        { delay: 500, fn: fetchComponentStatus, name: 'status komponentów' },
        { delay: 1000, fn: fetchPortfolioData, name: 'dane portfela' },
        { delay: 1500, fn: fetchAIStatus, name: 'status modeli AI' },
        { delay: 2000, fn: updateSentimentData, name: 'dane sentymentu' },
        { delay: 2500, fn: fetchSimulationResults, name: 'wyniki symulacji' },
        { delay: 3000, fn: updateChartData, name: 'dane wykresu' }
    ];

    // Funkcja do pobrania danych początkowych z pokazywaniem postępu
    let loadedCount = 0;
    initialLoads.forEach(item => {
        setTimeout(() => {
            try {
                console.log(`Ładowanie: ${item.name}...`);
                item.fn();
                
                // Aktualizuj procent załadowania
                loadedCount++;
                const percent = Math.round((loadedCount / initialLoads.length) * 100);
                
                // Pokaż postęp ładowania, jeśli istnieje element do tego
                const loadingElement = document.getElementById('loading-progress');
                if (loadingElement) {
                    loadingElement.textContent = `Ładowanie danych: ${percent}%`;
                    loadingElement.style.width = `${percent}%`;
                }
                
                // Po załadowaniu wszystkiego, ukryj pasek postępu
                if (loadedCount === initialLoads.length) {
                    setTimeout(() => {
                        const loadingContainer = document.getElementById('loading-container');
                        if (loadingContainer) {
                            loadingContainer.style.display = 'none';
                        }
                    }, 500);
                }
            } catch (e) {
                console.error(`Błąd podczas ładowania ${item.name}:`, e);
            }
        }, item.delay);
    });

    // Ustaw interwały dla automatycznego odświeżania
    setTimeout(() => {
        setupRefreshIntervals();
        isInitializing = false;
        isInitialized = true;
        console.log("Dashboard w pełni załadowany");
    }, 4000);
}

// Ustawienie inteligentnych interwałów odświeżania z przesunięciami czasowymi
function setupRefreshIntervals() {
    // Funkcja do sprawdzania, czy element jest widoczny na ekranie
    function isElementVisible(el) {
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    // Konfiguracja odświeżania - przesunięte czasowo, aby uniknąć nakładania się żądań
    const refreshConfig = [
        { 
            fn: fetchComponentStatus, 
            interval: 15000,  // Co 15s
            selector: '#component-status',
            key: 'components',
            criticalData: true // Dane krytyczne - odświeżaj nawet gdy niewidoczne
        },
        { 
            fn: updatePortfolioData, 
            interval: 25000,  // Co 25s
            selector: '#portfolio-container',
            key: 'portfolio',
            criticalData: true // Dane krytyczne - odświeżaj nawet gdy niewidoczne  
        },
        { 
            fn: fetchAIStatus, 
            interval: 35000,  // Co 35s
            selector: '#ai-models-container',
            key: 'aiModels',
            criticalData: false  
        },
        { 
            fn: updateSentimentData, 
            interval: 65000,  // Co 65s 
            selector: '#sentiment-container',
            key: 'sentiment',
            criticalData: false  
        },
        { 
            fn: updateChartData, 
            interval: 75000,  // Co 75s
            selector: '#main-chart',
            key: 'chart',
            criticalData: false  
        },
        { 
            fn: fetchSimulationResults, 
            interval: 90000,  // Co 90s
            selector: '#simulation-results',
            key: 'simulation',
            criticalData: false  
        }
    ];

    // Utwórz interwały odświeżania z dynamicznym sprawdzaniem widoczności
    refreshConfig.forEach((config, index) => {
        // Dodaj małe losowe opóźnienie do każdego interwału, aby uniknąć nakładania się
        const randomOffset = Math.floor(Math.random() * 1000); // 0-1000ms
        const initialDelay = 5000 + (index * 2000) + randomOffset;

        // Ustaw timeout początkowy, aby uniknąć nakładania się pierwszych odświeżeń
        setTimeout(() => {
            // Potem ustaw regularny interwał
            setInterval(() => {
                // Sprawdź, czy dokument jest aktywny
                if (!appState.activeDashboard && !config.criticalData) {
                    console.log(`Pomijam odświeżanie ${config.key} - karta nieaktywna`);
                    return;
                }

                // Sprawdź, czy element jest widoczny na ekranie
                const el = document.querySelector(config.selector);
                const isVisible = config.criticalData || isElementVisible(el);
                
                // Sprawdź, czy dane są nieaktualne i wymagają odświeżenia
                const isStale = appState.isStale(config.key, config.interval * 0.8);
                
                // Sprawdź, czy żądanie jest już w toku
                const isInProgress = appState.isRequestInProgress(config.key);
                
                if (isStale && !isInProgress && (isVisible || config.criticalData)) {
                    console.log(`Odświeżanie ${config.key}...`);
                    config.fn();
                }
            }, config.interval);
        }, initialDelay);
    });

    // Obsługa widoczności karty (document.visibilityState)
    document.addEventListener('visibilitychange', function() {
        appState.activeDashboard = document.visibilityState === 'visible';
        
        if (appState.activeDashboard) {
            console.log("Karta aktywna - wznawiam odświeżanie danych");
            // Jeśli karta została przywrócona, od razu odśwież krytyczne dane
            fetchComponentStatus();
            updatePortfolioData();
        } else {
            console.log("Karta nieaktywna - wstrzymuję odświeżanie nieistotnych danych");
        }
    });
}

// Wywołaj funkcję inicjalizującą po załadowaniu strony
document.addEventListener('DOMContentLoaded', initDashboard);

// Funkcja do pobierania wyników symulacji
function fetchSimulationResults() {
    fetchWithRetry(CONFIG.apiEndpoints.simulation.results, {
        headers: getAuthHeaders()
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Błąd HTTP: ' + response.status);
        }
        return response.json();
    })
    .then(data => {
        // Zapisz w stanie aplikacji
        appState.updateData('simulationData', data);
        
        // Zaktualizuj interfejs
        updateSimulationResults(data);
    })
    .catch(error => {
        console.error("Błąd podczas pobierania wyników symulacji:", error);
        
        // Znajdź kontener wyników symulacji
        const simulationContainer = findElement(['#simulationResults', '#simulation-results'], 'simulation-results');
        if (simulationContainer) {
            simulationContainer.innerHTML = `
                <div class="error-message">
                    <h4>Błąd podczas pobierania wyników symulacji</h4>
                    <p>${error.message || 'Problem z połączeniem z serwerem'}</p>
                    <button class="retry-button" onclick="fetchSimulationResults()">Spróbuj ponownie</button>
                </div>
            `;
        }
        
        // Rejestruj błąd
        handleApiError('simulationResults', error);
    });
}

// Funkcja do pobierania myśli AI
function fetchAIThoughts() {
    fetchWithRetry(CONFIG.apiEndpoints.ai.thoughts, {
        headers: getAuthHeaders()
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Zapisz w stanie aplikacji
            appState.updateData('aiThoughts', data);
            
            // Znajdź kontener myśli AI
            const thoughtsContainer = findElement(['#ai-thoughts-list', '.ai-thoughts-container'], 'ai-thoughts-list');
            
            if (thoughtsContainer && data && data.thoughts) {
                // Przygotuj HTML z myślami AI
                let thoughtsHTML = '<h3>Przemyślenia AI</h3>';
                
                if (data.thoughts.length > 0) {
                    thoughtsHTML += '<div class="thoughts-list">';
                    
                    data.thoughts.forEach(thought => {
                        // Określ klasę na podstawie pewności
                        let confidenceClass = 'low';
                        if (thought.confidence > 70) confidenceClass = 'high';
                        else if (thought.confidence > 40) confidenceClass = 'medium';
                        
                        thoughtsHTML += `
                            <div class="thought-card">
                                <div class="thought-header">
                                    <div class="thought-model">${thought.model}</div>
                                    <div class="thought-confidence ${confidenceClass}">Pewność: ${thought.confidence.toFixed(1)}%</div>
                                </div>
                                <div class="thought-content">${thought.thought}</div>
                                <div class="thought-timestamp">${thought.timestamp}</div>
                            </div>
                        `;
                    });
                    
                    thoughtsHTML += '</div>';
                } else {
                    thoughtsHTML += '<p>Brak dostępnych przemyśleń AI</p>';
                }
                
                // Zaktualizuj kontener
                thoughtsContainer.innerHTML = thoughtsHTML;
            }
            
            console.log("Pobrano myśli AI:", data);
        })
        .catch(error => {
            console.error("Błąd podczas pobierania myśli AI:", error);
            
            // Znajdź kontener myśli AI
            const thoughtsContainer = findElement(['#ai-thoughts-list', '.ai-thoughts-container'], 'ai-thoughts-list');
            
            if (thoughtsContainer) {
                thoughtsContainer.innerHTML = `
                    <div class="error-message">
                        <h4>Błąd podczas pobierania przemyśleń AI</h4>
                        <p>${error.message || 'Problem z połączeniem z serwerem'}</p>
                        <button class="retry-button" onclick="fetchAIThoughts()">Spróbuj ponownie</button>
                    </div>
                `;
            }
            
            // Zarejestruj błąd
            handleApiError('aiThoughts', error);
        });
}

// Funkcja do pobierania statusu AI
function fetchAIStatus() {
    fetchWithRetry(CONFIG.apiEndpoints.ai.modelsStatus, {
        headers: getAuthHeaders()
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Zapisz dane w stanie aplikacji
            appState.updateData('aiModelsData', data);
            
            // Zaktualizuj interfejs
            updateAIModelsStatus(data);
            
            // Zaaktualizuj również status odświeżenia
            updateLastRefreshed();
        })
        .catch(error => {
            console.error("Błąd podczas pobierania statusu modeli AI:", error);
            
            // Pobierz kontener modeli AI
            const aiModelsContainer = findElement(['#ai-models-container', '#ai-models-status'], 'ai-models-container');
            
            if (aiModelsContainer) {
                aiModelsContainer.innerHTML = `
                    <div class="error-message">
                        <h4>Błąd podczas pobierania statusu modeli AI</h4>
                        <p>${error.message || 'Problem z połączeniem z serwerem'}</p>
                        <button class="retry-button" onclick="fetchAIStatus()">Spróbuj ponownie</button>
                    </div>
                `;
            }
            
            showNotification('error', 'Nie udało się pobrać statusu modeli AI');
            
            // Zarejestruj błąd
            handleApiError('aiModels', error);
        });
}

// Funkcje uwierzytelniania
function login(username, password) {
    fetch(CONFIG.apiEndpoints.auth.login, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.token) {
            appState.token = data.token;
            appState.isAuthenticated = true;
            localStorage.setItem('token', data.token);
            initializeUI();
            showNotification('success', 'Zalogowano pomyślnie');
        } else {
            showNotification('error', data.message || 'Błąd logowania');
        }
    })
    .catch(error => {
        console.error('Błąd logowania:', error);
        showNotification('error', 'Błąd logowania');
    });
}

function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        showLoginForm();
        return false;
    }

    fetch(CONFIG.apiEndpoints.auth.verify, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.valid) {
            appState.token = token;
            appState.isAuthenticated = true;
            initializeUI();
        } else {
            showLoginForm();
        }
    })
    .catch(() => {
        showLoginForm();
    });
}

function logout() {
    appState.token = null;
    appState.isAuthenticated = false;
    localStorage.removeItem('token');
    showLoginForm();
}

function showLoginForm() {
    const mainContent = document.querySelector('.container');
    if (mainContent) {
        mainContent.innerHTML = `
            <div class="login-container">
                <h2>Logowanie</h2>
                <form id="login-form" class="login-form">
                    <div class="form-group">
                        <label for="username">Użytkownik:</label>
                        <input type="text" id="username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Hasło:</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Zaloguj</button>
                    <div class="form-links">
                        <a href="#" id="show-register">Nie masz konta? Zarejestruj się</a>
                    </div>
                </form>
            </div>
        `;

        document.getElementById('login-form').addEventListener('submit', (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            login(username, password);
        });

        document.getElementById('show-register').addEventListener('click', (e) => {
            e.preventDefault();
            showRegisterForm();
        });
    }
}

function showRegisterForm() {
    const mainContent = document.querySelector('.container');
    if (mainContent) {
        mainContent.innerHTML = `
            <div class="login-container">
                <h2>Rejestracja</h2>
                <form id="register-form" class="login-form">
                    <div class="form-group">
                        <label for="reg-username">Użytkownik:</label>
                        <input type="text" id="reg-username" name="username" required>
                    </div>
                    <div class="form-group">
                        <label for="reg-email">Email:</label>
                        <input type="email" id="reg-email" name="email" required>
                    </div>
                    <div class="form-group">
                        <label for="reg-password">Hasło:</label>
                        <input type="password" id="reg-password" name="password" required>
                    </div>
                    <div class="form-group">
                        <label for="reg-password-confirm">Potwierdź hasło:</label>
                        <input type="password" id="reg-password-confirm" name="password-confirm" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Zarejestruj</button>
                    <div class="form-links">
                        <a href="#" id="show-login">Masz już konto? Zaloguj się</a>
                    </div>
                </form>
            </div>
        `;

        document.getElementById('register-form').addEventListener('submit', (e) => {
            e.preventDefault();
            const username = document.getElementById('reg-username').value;
            const email = document.getElementById('reg-email').value;
            const password = document.getElementById('reg-password').value;
            const passwordConfirm = document.getElementById('reg-password-confirm').value;

            if (password !== passwordConfirm) {
                showNotification('error', 'Hasła nie są identyczne');
                return;
            }

            register(username, email, password);
        });

        document.getElementById('show-login').addEventListener('click', (e) => {
            e.preventDefault();
            showLoginForm();
        });
    }
}

function register(username, email, password) {
    fetch(CONFIG.apiEndpoints.auth.register, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, email, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.token) {
            appState.token = data.token;
            appState.isAuthenticated = true;
            localStorage.setItem('token', data.token);
            initializeUI();
            showNotification('success', 'Zarejestrowano pomyślnie');
        } else {
            showNotification('error', data.message || 'Błąd rejestracji');
        }
    })
    .catch(error => {
        console.error('Błąd rejestracji:', error);
        showNotification('error', 'Błąd rejestracji');
    });
}

// Modyfikacja funkcji wysyłających żądania API, aby dodawały token do nagłówków
function getAuthHeaders() {
    return appState.token ? {
        'Authorization': `Bearer ${appState.token}`,
        'Content-Type': 'application/json'
    } : {
        'Content-Type': 'application/json'
    };
}

// Funkcje obsługi błędów

// Pokaż błąd w kontenerze błędów
function showError(message) {
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');
    
    if (errorContainer && errorText) {
        errorText.textContent = message;
        errorContainer.style.display = 'flex';
    }
}

// Ukryj kontener błędów
function hideError() {
    const errorContainer = document.getElementById('error-container');
    if (errorContainer) {
        errorContainer.style.display = 'none';
    }
}

// Pokaż błąd w kontenerze błędów z opcjonalnym timeoutem
function showErrorMessage(message, autoHide = true) {
    showError(message);
    
    if (autoHide) {
        setTimeout(hideError, 5000); // Automatyczne ukrycie po 5 sekundach
    }
}

/**
 * Funkcja bezpiecznego dostępu do elementów DOM - wyszukuje element według różnych identyfikatorów
 * 
 * @param {string|string[]} selectors - Pojedynczy selektor lub tablica selektorów do wyszukania elementu
 * @param {string} fallbackId - ID elementu, który zostanie utworzony jeśli żaden element nie zostanie znaleziony
 * @param {HTMLElement} parentElement - Opcjonalny element nadrzędny do wyszukiwania (domyślnie document)
 * @returns {HTMLElement|null} - Znaleziony lub utworzony element, lub null jeśli nie znaleziono/nie utworzono
 */
function findElement(selectors, fallbackId = null, parentElement = document) {
    // Konwertuj pojedynczy selektor na tablicę, jeśli potrzeba
    const selectorsArray = Array.isArray(selectors) ? selectors : [selectors];
    
    // Przetwarzaj selektory w kolejności (ID mają pierwszeństwo)
    for (const selector of selectorsArray) {
        // Sprawdź, czy to jest ID (zaczyna się od #), klasa (zaczyna się od .) lub tag
        if (selector.startsWith('#')) {
            const element = document.getElementById(selector.substring(1));
            if (element) return element;
        } else if (selector.startsWith('.')) {
            const elements = parentElement.getElementsByClassName(selector.substring(1));
            if (elements.length > 0) return elements[0];
        } else {
            const elements = parentElement.getElementsByTagName(selector);
            if (elements.length > 0) return elements[0];
        }
    }
    
    // Jeśli nic nie znaleziono, a podano fallbackId, utwórz nowy element
    if (fallbackId) {
        console.log(`Tworzę nowy element z ID: ${fallbackId}`);
        
        // Znajdź odpowiednie miejsce na wstawienie elementu
        let container;
        
        // Próbuj znaleźć typowe kontenery
        const containers = [
            document.querySelector('.dashboard-content'),
            document.querySelector('.container'),
            document.querySelector('main'),
            document.body
        ].filter(Boolean);
        
        if (containers.length > 0) {
            container = containers[0];
            
            // Utwórz nowy element div
            const newElement = document.createElement('div');
            newElement.id = fallbackId;
            
            // Dodaj klasę z tym samym imieniem co ID dla kompatybilności
            newElement.className = fallbackId.replace(/[^a-zA-Z0-9-_]/g, '-');
            
            // Ustaw style, aby element był widoczny
            newElement.style.padding = '15px';
            newElement.style.margin = '15px 0';
            newElement.style.backgroundColor = '#f8f9fa';
            newElement.style.border = '1px solid #dee2e6';
            newElement.style.borderRadius = '4px';
            
            // Dodaj pusty nagłówek
            const header = document.createElement('h3');
            header.textContent = fallbackId.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            newElement.appendChild(header);
            
            // Dodaj do kontenera
            container.appendChild(newElement);
            
            return newElement;
        }
    }
    
    // Nic nie znaleziono i nie można utworzyć
    return null;
}

/**
 * Funkcja bezpiecznej aktualizacji elementu DOM
 * 
 * @param {string|string[]} selectors - Selektor lub tablica selektorów do znalezienia elementu
 * @param {string|Function} content - Zawartość lub funkcja zwracająca zawartość
 * @param {boolean} asHTML - Czy traktować zawartość jako HTML
 * @returns {HTMLElement|null} - Zaktualizowany element lub null
 */
function updateElement(selectors, content, asHTML = false) {
    const element = findElement(selectors);
    if (!element) {
        console.warn(`Nie znaleziono elementu dla selektorów: ${JSON.stringify(selectors)}`);
        return null;
    }
    
    // Określ zawartość (jeśli to funkcja, wykonaj ją)
    const finalContent = typeof content === 'function' ? content(element) : content;
    
    // Aktualizuj zawartość elementu
    if (asHTML) {
        element.innerHTML = finalContent;
    } else {
        element.textContent = finalContent;
    }
    
    return element;
}

/**
 * Funkcja do tworzenia elementu z szablonem HTML
 * 
 * @param {string} tagName - Nazwa znacznika HTML
 * @param {Object} attributes - Obiekt atrybutów
 * @param {string|HTMLElement|Array} children - Dzieci elementu (tekst, element lub tablica)
 * @returns {HTMLElement} - Utworzony element
 */
function createElement(tagName, attributes = {}, children = null) {
    const element = document.createElement(tagName);
    
    // Dodaj atrybuty
    Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'style' && typeof value === 'object') {
            Object.entries(value).forEach(([prop, val]) => {
                element.style[prop] = val;
            });
        } else {
            element.setAttribute(key, value);
        }
    });
    
    // Dodaj dzieci
    if (children !== null) {
        if (Array.isArray(children)) {
            children.forEach(child => {
                if (typeof child === 'string') {
                    element.appendChild(document.createTextNode(child));
                } else if (child instanceof HTMLElement) {
                    element.appendChild(child);
                }
            });
        } else if (typeof children === 'string') {
            element.textContent = children;
        } else if (children instanceof HTMLElement) {
            element.appendChild(children);
        }
    }
    
    return element;
}