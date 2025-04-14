// Konfiguracja
const AI_CONFIG = {
    updateInterval: 60000, // 60 sekund
    chartUpdateInterval: 120000, // 2 minuty
};

// Stan aplikacji AI Monitora
const aiMonitorState = {
    lastThoughts: [],
    lastLearningStatus: null,
    isTraining: false
};

// Inicjalizacja po załadowaniu dokumentu
document.addEventListener('DOMContentLoaded', function() {
    console.log("Inicjalizacja AI Monitor");
    initializeAIMonitor();
});

// Inicjalizacja komponentów AI Monitora
function initializeAIMonitor() {
    // Pobierz początkowe dane
    fetchAIThoughts();
    fetchAILearningStatus();

    // Ustaw regularną aktualizację danych
    setInterval(fetchAIThoughts, AI_CONFIG.updateInterval);
    setInterval(fetchAILearningStatus, AI_CONFIG.updateInterval);

    // Podłącz przyciski
    setupAIControlButtons();
}

// Pobieranie myśli AI
function fetchAIThoughts() {
    fetch('/api/ai/thoughts')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.thoughts && data.thoughts.length > 0) {
                updateAIThoughtsUI(data.thoughts);
                aiMonitorState.lastThoughts = data.thoughts;
            }
        })
        .catch(error => {
            console.error("Błąd podczas pobierania przemyśleń AI:", error);
            showNotification('error', 'Nie udało się pobrać przemyśleń AI');
        });
}

// Pobieranie statusu uczenia AI
function fetchAILearningStatus() {
    fetch('/api/ai/learning-status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateAILearningStatusUI(data);
            aiMonitorState.lastLearningStatus = data;
            aiMonitorState.isTraining = data.is_training || false;
        })
        .catch(error => {
            console.error("Błąd podczas pobierania statusu uczenia AI:", error);
            showNotification('error', 'Nie udało się pobrać statusu uczenia AI');
        });
}

// Aktualizacja interfejsu przemyśleń AI
function updateAIThoughtsUI(thoughts) {
    const container = document.getElementById('ai-thoughts-container');
    if (!container) return;

    let html = '';

    thoughts.forEach(thought => {
        const confidenceClass = getConfidenceClass(thought.confidence);

        html += `
        <div class="ai-thought">
            <div class="thought-header">
                <div class="thought-model">${thought.model}</div>
                <div class="thought-confidence ${confidenceClass}">${thought.confidence.toFixed(1)}%</div>
            </div>
            <div class="thought-content">${thought.thought}</div>
            <div class="thought-meta">
                <span class="thought-type">${thought.type}</span>
                <span class="thought-time">${thought.timestamp}</span>
            </div>
        </div>`;
    });

    if (html === '') {
        html = '<div class="no-data">Brak przemyśleń AI</div>';
    }

    container.innerHTML = html;
}

// Aktualizacja interfejsu statusu uczenia
function updateAILearningStatusUI(data) {
    const container = document.getElementById('learning-status-container');
    if (!container) return;

    if (!data) {
        container.innerHTML = '<div class="no-data">Brak danych o statusie uczenia</div>';
        return;
    }

    let html = '';

    // Status treningu
    html += `
    <div class="learning-header">
        <div class="learning-status ${data.is_training ? 'training-active' : 'training-inactive'}">
            ${data.is_training ? 'Trening w toku' : 'Brak aktywnego treningu'}
        </div>
        <div class="learning-timestamp">Ostatnia aktualizacja: ${data.timestamp}</div>
    </div>`;

    // Dane uczenia
    if (data.learning_data && data.learning_data.length > 0) {
        html += '<div class="learning-progress"><h4>Postęp uczenia</h4>';
        html += '<div class="learning-progress-table"><table>';
        html += '<thead><tr><th>Iteracja</th><th>Dokładność</th><th>Win Rate</th><th>Transakcje</th><th>Zysk</th></tr></thead>';
        html += '<tbody>';

        data.learning_data.forEach(item => {
            html += `<tr>
                <td>${item.iteration}</td>
                <td>${item.accuracy.toFixed(2)}%</td>
                <td>${item.win_rate.toFixed(2)}%</td>
                <td>${item.trades}</td>
                <td class="${item.profit >= 0 ? 'positive' : 'negative'}">${item.profit.toFixed(2)}</td>
            </tr>`;
        });

        html += '</tbody></table></div></div>';
    }

    // Modele w trakcie treningu
    if (data.models_training && data.models_training.length > 0) {
        html += '<div class="training-models"><h4>Modele w treningu</h4>';

        data.models_training.forEach(model => {
            const progress = Math.min(Math.max(model.progress, 0), 100);

            html += `
            <div class="training-model">
                <div class="training-model-header">
                    <span class="model-name">${model.name}</span>
                    <span class="model-type">${model.type}</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: ${progress}%"></div>
                    <div class="progress-text">${progress.toFixed(1)}%</div>
                </div>
                <div class="training-model-details">
                    <div>ETA: ${model.eta}</div>
                    <div>Dokładność: ${model.current_accuracy.toFixed(2)}%</div>
                </div>
            </div>`;
        });

        html += '</div>';
    }

    // Podsumowanie treningu
    if (data.is_training) {
        html += `
        <div class="training-summary">
            <div>Iteracja: ${data.current_iteration || 1}/${data.total_iterations || 5}</div>
        </div>`;
    }

    container.innerHTML = html;
}

// Konfiguracja przycisków kontrolnych AI
function setupAIControlButtons() {
    // Przycisk szybkiej symulacji
    const runSimulationBtn = document.getElementById('run-simulation-btn');
    if (runSimulationBtn) {
        runSimulationBtn.addEventListener('click', function() {
            runSimulation(false);
        });
    }

    // Przycisk symulacji z uczeniem
    const runLearningBtn = document.getElementById('run-learning-btn');
    if (runLearningBtn) {
        runLearningBtn.addEventListener('click', function() {
            runSimulation(true);
        });
    }

    // Formularz symulacji
    const simulationForm = document.getElementById('simulation-form');
    if (simulationForm) {
        simulationForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const initialCapital = parseFloat(document.getElementById('initial-capital').value) || 10000;
            const duration = parseInt(document.getElementById('duration').value) || 1000;
            const withLearning = document.getElementById('with-learning').checked;
            const iterations = parseInt(document.getElementById('iterations').value) || 5;

            runCustomSimulation(initialCapital, duration, withLearning, iterations);
        });
    }
}

// Uruchomienie symulacji
function runSimulation(withLearning) {
    let endpoint = withLearning ? '/api/simulation/learn' : '/api/simulation/run';

    // Dane do wysłania
    const data = {
        initial_capital: 10000,
        duration: 1000
    };

    if (withLearning) {
        data.iterations = 5;
    }

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Błąd HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            showNotification('success', `Symulacja ${withLearning ? 'z uczeniem' : ''} uruchomiona pomyślnie`);

            // Aktualizuj dane po 5 sekundach
            setTimeout(() => {
                fetchAILearningStatus();
                fetchSimulationResults();
            }, 5000);
        } else {
            showNotification('error', data.message || 'Błąd podczas uruchamiania symulacji');
        }
    })
    .catch(error => {
        console.error(`Błąd podczas uruchamiania symulacji ${withLearning ? 'z uczeniem' : ''}:`, error);
        showNotification('error', 'Nie udało się uruchomić symulacji');
    });
}

// Uruchomienie niestandardowej symulacji
function runCustomSimulation(initialCapital, duration, withLearning, iterations) {
    let endpoint = withLearning ? '/api/simulation/learn' : '/api/simulation/run';

    // Dane do wysłania
    const data = {
        initial_capital: initialCapital,
        duration: duration
    };

    if (withLearning) {
        data.iterations = iterations;
    }

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Błąd HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            showNotification('success', `Niestandardowa symulacja ${withLearning ? 'z uczeniem' : ''} uruchomiona pomyślnie`);

            // Aktualizuj dane po 5 sekundach
            setTimeout(() => {
                fetchAILearningStatus();
                fetchSimulationResults();
            }, 5000);
        } else {
            showNotification('error', data.message || 'Błąd podczas uruchamiania niestandardowej symulacji');
        }
    })
    .catch(error => {
        console.error(`Błąd podczas uruchamiania niestandardowej symulacji:`, error);
        showNotification('error', 'Nie udało się uruchomić niestandardowej symulacji');
    });
}

// Pobieranie wyników symulacji
function fetchSimulationResults() {
    fetch('/api/simulation-results')
        .then(response => {
            if (!response.ok) {
                throw new Error('Błąd HTTP: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            updateSimulationResultsUI(data);
        })
        .catch(error => {
            console.error("Błąd podczas pobierania wyników symulacji:", error);
            showNotification('error', 'Nie udało się pobrać wyników symulacji');
        });
}

// Aktualizacja interfejsu wyników symulacji
function updateSimulationResultsUI(data) {
    // Wyświetlenie wyników zostało zaimplementowane w dashboard.js
    // Ta funkcja może być rozszerzona o dodatkowe elementy specyficzne dla zakładki AI Monitor
}

// Funkcje pomocnicze
function getConfidenceClass(confidence) {
    if (confidence >= 80) return 'high-confidence';
    if (confidence >= 60) return 'medium-confidence';
    return 'low-confidence';
}

// Wyświetlanie powiadomień (jeśli nie istnieje w dashboardzie)
function showNotification(type, message) {
    // Sprawdź, czy funkcja istnieje w dashboard.js
    if (typeof window.showNotification === 'function') {
        window.showNotification(type, message);
        return;
    }

    // Implementacja własna, jeśli funkcja nie istnieje
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