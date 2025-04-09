
// Moduł do monitorowania myśli AI i statusu uczenia
const aiMonitor = {
    // Stan modułu
    state: {
        thoughts: [],
        learningStatus: null,
        lastUpdate: 0,
        isPolling: false,
        updateInterval: 30000, // 30 sekund
        isFirstLoad: true
    },

    // Inicjalizacja modułu
    init() {
        console.log("Inicjalizacja AI Monitor");
        this.setupEventListeners();
        this.startPolling();
    },

    // Ustawienie nasłuchiwania zdarzeń
    setupEventListeners() {
        // Nasłuchiwanie przycisków uruchamiających symulację
        document.getElementById('run-simulation-btn')?.addEventListener('click', () => this.runSimulation());
        document.getElementById('run-learning-btn')?.addEventListener('click', () => this.runLearningSimulation());
        
        // Nasłuchiwanie formularza konfiguracyjnego
        document.getElementById('simulation-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.runSimulationFromForm();
        });
    },

    // Rozpoczęcie regularnego odpytywania API
    startPolling() {
        if (this.state.isPolling) return;
        
        this.state.isPolling = true;
        this.updateAiThoughts();
        this.updateLearningStatus();
        
        // Ustawienie interwału dla regularnego odpytywania
        setInterval(() => {
            this.updateAiThoughts();
            this.updateLearningStatus();
        }, this.state.updateInterval);
    },

    // Pobranie myśli AI z API
    updateAiThoughts() {
        fetch('/api/ai/thoughts')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Błąd HTTP: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    this.state.thoughts = data.thoughts;
                    this.state.lastUpdate = Date.now();
                    this.renderThoughts();
                }
            })
            .catch(error => {
                console.error("Błąd podczas pobierania myśli AI:", error);
            });
    },

    // Pobranie statusu uczenia z API
    updateLearningStatus() {
        fetch('/api/ai/learning-status')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Błąd HTTP: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    this.state.learningStatus = data;
                    this.renderLearningStatus();
                }
            })
            .catch(error => {
                console.error("Błąd podczas pobierania statusu uczenia AI:", error);
            });
    },

    // Renderowanie myśli AI w interfejsie
    renderThoughts() {
        const container = document.getElementById('ai-thoughts-container');
        if (!container) return;

        if (this.state.thoughts.length === 0) {
            container.innerHTML = '<div class="no-data">Brak aktywnych przemyśleń AI</div>';
            return;
        }

        let html = '';
        
        // Sortowanie myśli wg typu i czasu
        const sortedThoughts = [...this.state.thoughts].sort((a, b) => {
            // Najpierw sortuj wg typu
            if (a.type !== b.type) {
                return a.type.localeCompare(b.type);
            }
            // Następnie wg czasu (od najnowszych)
            return new Date(b.timestamp) - new Date(a.timestamp);
        });

        sortedThoughts.forEach(thought => {
            // Określenie klasy CSS bazując na pewności
            let confidenceClass = 'neutral';
            if (thought.confidence >= 80) {
                confidenceClass = 'positive';
            } else if (thought.confidence < 60) {
                confidenceClass = 'negative';
            }

            // Ikona zależna od typu myśli
            let icon = '💭';
            if (thought.type === 'sentiment') {
                icon = '🔍';
            } else if (thought.type === 'anomaly') {
                icon = '⚠️';
            } else if (thought.type === 'strategy') {
                icon = '📊';
            } else if (thought.type === 'model_recognition') {
                icon = '🧠';
            }

            html += `
            <div class="ai-thought-card">
                <div class="thought-header">
                    <div class="thought-icon">${icon}</div>
                    <div class="thought-model">${thought.model}</div>
                    <div class="thought-time">${thought.timestamp}</div>
                </div>
                <div class="thought-content">
                    <p>${thought.thought}</p>
                </div>
                <div class="thought-footer">
                    <div class="thought-confidence ${confidenceClass}">
                        Pewność: ${thought.confidence.toFixed(1)}%
                    </div>
                </div>
            </div>`;
        });

        container.innerHTML = html;
    },

    // Renderowanie statusu uczenia w interfejsie
    renderLearningStatus() {
        const container = document.getElementById('learning-status-container');
        if (!container) return;

        const data = this.state.learningStatus;
        if (!data) {
            container.innerHTML = '<div class="no-data">Brak danych o uczeniu</div>';
            return;
        }

        let html = '';
        
        // Status aktywnego uczenia
        if (data.is_training) {
            html += `
            <div class="learning-active-status">
                <div class="status-header">
                    <div class="status-icon">🔄</div>
                    <div class="status-title">Uczenie w toku</div>
                </div>
                <div class="status-content">
                    <p>Iteracja: ${data.current_iteration}/${data.total_iterations}</p>
                </div>
            </div>`;
        }

        // Modele w treningu
        if (data.models_training && data.models_training.length > 0) {
            html += '<div class="models-training-section"><h4>Modele w treningu</h4>';
            
            data.models_training.forEach(model => {
                html += `
                <div class="model-training-item">
                    <div class="model-name">${model.name} (${model.type})</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: ${model.progress}%"></div>
                    </div>
                    <div class="training-details">
                        <div>Postęp: ${model.progress.toFixed(1)}%</div>
                        <div>Pozostało: ${model.eta}</div>
                        <div>Dokładność: ${model.current_accuracy.toFixed(1)}%</div>
                    </div>
                </div>`;
            });
            
            html += '</div>';
        }

        // Wyniki uczenia
        if (data.learning_data && data.learning_data.length > 0) {
            html += '<div class="learning-results-section"><h4>Wyniki uczenia</h4>';
            html += '<table class="learning-results-table">';
            html += `
            <thead>
                <tr>
                    <th>Iteracja</th>
                    <th>Dokładność</th>
                    <th>Win Rate</th>
                    <th>Transakcje</th>
                    <th>Zysk</th>
                </tr>
            </thead>
            <tbody>`;
            
            data.learning_data.forEach(result => {
                // Klasy CSS zależne od wyników
                const accuracyClass = result.accuracy >= 70 ? 'positive' : (result.accuracy >= 50 ? 'neutral' : 'negative');
                const winRateClass = result.win_rate >= 60 ? 'positive' : (result.win_rate >= 45 ? 'neutral' : 'negative');
                const profitClass = result.profit >= 0 ? 'positive' : 'negative';
                
                html += `
                <tr>
                    <td>${result.iteration}</td>
                    <td class="${accuracyClass}">${result.accuracy.toFixed(1)}%</td>
                    <td class="${winRateClass}">${result.win_rate.toFixed(1)}%</td>
                    <td>${result.trades}</td>
                    <td class="${profitClass}">${result.profit.toFixed(2)}</td>
                </tr>`;
            });
            
            html += '</tbody></table></div>';
        }

        // Jeśli brak danych
        if (html === '') {
            html = '<div class="no-data">Brak danych o uczeniu modeli AI</div>';
        }

        container.innerHTML = html;

        // Animacja na pierwsze załadowanie, jeśli są dane
        if (this.state.isFirstLoad && html !== '') {
            this.state.isFirstLoad = false;
            container.querySelectorAll('.learning-active-status, .model-training-item, .learning-results-section')
                    .forEach(el => el.classList.add('animate-in'));
        }
    },

    // Uruchomienie symulacji
    runSimulation() {
        const initialCapital = 10000.0; // Domyślna wartość
        const duration = 1000; // Domyślna wartość
        
        this.showLoader('Uruchamianie symulacji...');
        
        fetch('/api/simulation/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                initial_capital: initialCapital,
                duration: duration
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            this.hideLoader();
            if (data.status === 'success') {
                showNotification('success', 'Symulacja zakończona pomyślnie');
                // Odśwież dane wyników symulacji
                if (typeof updateSimulationResults === 'function') {
                    fetch('/api/simulation-results')
                        .then(response => response.json())
                        .then(data => {
                            updateSimulationResults(data);
                        });
                }
            } else {
                showNotification('error', `Błąd symulacji: ${data.message}`);
            }
        })
        .catch(error => {
            this.hideLoader();
            console.error("Błąd podczas uruchamiania symulacji:", error);
            showNotification('error', `Błąd podczas uruchamiania symulacji: ${error.message}`);
        });
    },

    // Uruchomienie symulacji z uczeniem
    runLearningSimulation() {
        const initialCapital = 10000.0; // Domyślna wartość
        const duration = 1000; // Domyślna wartość
        const iterations = 5; // Domyślna wartość
        
        this.showLoader('Uruchamianie symulacji z uczeniem...');
        
        fetch('/api/simulation/learn', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                initial_capital: initialCapital,
                duration: duration,
                iterations: iterations
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Błąd HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            this.hideLoader();
            if (data.status === 'success') {
                showNotification('success', 'Symulacja z uczeniem zakończona pomyślnie');
                // Odśwież dane
                this.updateLearningStatus();
                
                // Odśwież dane wyników symulacji
                if (typeof updateSimulationResults === 'function') {
                    fetch('/api/simulation-results')
                        .then(response => response.json())
                        .then(data => {
                            updateSimulationResults(data);
                        });
                }
            } else {
                showNotification('error', `Błąd symulacji z uczeniem: ${data.message}`);
            }
        })
        .catch(error => {
            this.hideLoader();
            console.error("Błąd podczas uruchamiania symulacji z uczeniem:", error);
            showNotification('error', `Błąd podczas uruchamiania symulacji z uczeniem: ${error.message}`);
        });
    },

    // Uruchomienie symulacji z wartościami z formularza
    runSimulationFromForm() {
        const form = document.getElementById('simulation-form');
        if (!form) return;
        
        const formData = new FormData(form);
        const initialCapital = parseFloat(formData.get('initial_capital') || 10000.0);
        const duration = parseInt(formData.get('duration') || 1000);
        const withLearning = formData.get('with_learning') === 'on';
        const iterations = parseInt(formData.get('iterations') || 5);
        
        if (withLearning) {
            this.showLoader(`Uruchamianie symulacji z uczeniem (${iterations} iteracji)...`);
            
            fetch('/api/simulation/learn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    initial_capital: initialCapital,
                    duration: duration,
                    iterations: iterations
                })
            })
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                if (data.status === 'success') {
                    showNotification('success', `Symulacja z uczeniem zakończona: ${data.learning_results?.length || 0} iteracji`);
                    this.updateLearningStatus();
                    
                    // Odśwież wyniki
                    if (typeof updateSimulationResults === 'function') {
                        fetch('/api/simulation-results')
                            .then(response => response.json())
                            .then(data => {
                                updateSimulationResults(data);
                            });
                    }
                } else {
                    showNotification('error', data.message || 'Błąd symulacji z uczeniem');
                }
            })
            .catch(error => {
                this.hideLoader();
                showNotification('error', error.message);
            });
        } else {
            this.showLoader('Uruchamianie symulacji...');
            
            fetch('/api/simulation/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    initial_capital: initialCapital,
                    duration: duration
                })
            })
            .then(response => response.json())
            .then(data => {
                this.hideLoader();
                if (data.status === 'success') {
                    showNotification('success', 'Symulacja zakończona pomyślnie');
                    
                    // Odśwież wyniki
                    if (typeof updateSimulationResults === 'function') {
                        fetch('/api/simulation-results')
                            .then(response => response.json())
                            .then(data => {
                                updateSimulationResults(data);
                            });
                    }
                } else {
                    showNotification('error', data.message || 'Błąd symulacji');
                }
            })
            .catch(error => {
                this.hideLoader();
                showNotification('error', error.message);
            });
        }
    },

    // Wyświetlenie loadera
    showLoader(message = 'Ładowanie...') {
        let loader = document.getElementById('ai-loader');
        
        // Utwórz loader jeśli nie istnieje
        if (!loader) {
            loader = document.createElement('div');
            loader.id = 'ai-loader';
            loader.innerHTML = `
                <div class="loader-content">
                    <div class="loader-spinner"></div>
                    <div class="loader-message">${message}</div>
                </div>
            `;
            document.body.appendChild(loader);
        } else {
            loader.querySelector('.loader-message').textContent = message;
            loader.style.display = 'flex';
        }
    },

    // Ukrycie loadera
    hideLoader() {
        const loader = document.getElementById('ai-loader');
        if (loader) {
            loader.style.display = 'none';
        }
    }
};

// Inicjalizacja modułu po załadowaniu dokumentu
document.addEventListener('DOMContentLoaded', function() {
    aiMonitor.init();
});
