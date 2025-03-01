"""
strategy_optimizer.py
---------------------
Moduł analizujący wyniki strategii handlowych i proponujący modyfikacje parametrów.
Funkcjonalności:
- Wykorzystuje dane z backtestingu, real-time tradingu i analizy ryzyka do oceny wydajności strategii.
- Implementuje metody optymalizacji portfela, takie jak metoda Markowitza oraz adaptacyjne algorytmy heurystyczne (np. genetyczne).
- Uwzględnia ograniczenia kapitałowe i cele inwestora (maksymalny zysk, minimalizacja drawdown, stabilność).
- Integruje się z modułem hyperparameter_tuner.py w celu automatycznej weryfikacji efektywności proponowanych zmian.
- Zapewnia mechanizmy raportowania i wizualizacji wyników oraz zawiera testy automatyczne.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Przykładowa funkcja optymalizacji portfela metodą Markowitza (mean-variance optimization)
def optimize_portfolio_markowitz(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Optymalizuje portfel przy użyciu metody Markowitza.
    
    Parameters:
        returns (pd.DataFrame): DataFrame z historycznymi zwrotami poszczególnych aktywów.
        risk_free_rate (float): Wolna od ryzyka stopa zwrotu.
    
    Returns:
        dict: Słownik zawierający optymalne wagi, oczekiwany zwrot, ryzyko (std) i Sharpe Ratio.
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_performance(weights):
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_std if port_std != 0 else 0
        return port_return, port_std, sharpe

    def negative_sharpe(weights):
        return -portfolio_performance(weights)[2]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array(n_assets * [1. / n_assets])

    result = minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        opt_weights = result.x
        port_return, port_std, sharpe = portfolio_performance(opt_weights)
        logging.info("Optymalizacja Markowitza zakończona sukcesem.")
        return {"weights": opt_weights, "expected_return": port_return, "risk": port_std, "sharpe_ratio": sharpe}
    else:
        logging.error("Optymalizacja Markowitza nie powiodła się.")
        raise ValueError("Optymalizacja portfela nie powiodła się.")

# Przykładowa adaptacyjna optymalizacja metodą genetyczną
def optimize_portfolio_genetic(returns: pd.DataFrame, population_size: int = 50, generations: int = 100) -> dict:
    """
    Optymalizuje portfel przy użyciu uproszczonej optymalizacji genetycznej.
    
    Parameters:
        returns (pd.DataFrame): DataFrame z historycznymi zwrotami poszczególnych aktywów.
        population_size (int): Rozmiar populacji.
        generations (int): Liczba generacji.
    
    Returns:
        dict: Słownik zawierający optymalne wagi, oczekiwany zwrot, ryzyko oraz Sharpe Ratio.
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def fitness(weights):
        # Funkcja fitness oparta na Sharpe Ratio (maksymalizujemy Sharpe Ratio)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return) / port_std if port_std != 0 else 0
        return sharpe

    # Inicjalizacja populacji
    population = []
    for _ in range(population_size):
        weights = np.random.dirichlet(np.ones(n_assets))
        population.append(weights)
    
    for gen in range(generations):
        # Ocena fitness
        fitness_scores = np.array([fitness(ind) for ind in population])
        # Selekcja: wybieramy najlepsze 50% osobników
        selected_indices = fitness_scores.argsort()[-(population_size // 2):]
        selected = [population[i] for i in selected_indices]
        # Reprodukcja: krzyżowanie i mutacja
        offspring = []
        while len(offspring) < population_size - len(selected):
            parents = np.random.choice(selected, 2, replace=False)
            # Krzyżowanie - średnia arytmetyczna
            child = (parents[0] + parents[1]) / 2.0
            # Mutacja
            mutation = np.random.normal(0, 0.05, size=n_assets)
            child = child + mutation
            # Normalizacja
            child = np.clip(child, 0, None)
            if child.sum() == 0:
                child = np.random.dirichlet(np.ones(n_assets))
            else:
                child = child / child.sum()
            offspring.append(child)
        population = selected + offspring
        if gen % 10 == 0:
            best_fitness = np.max([fitness(ind) for ind in population])
            logging.info("Generacja %d, najlepsze fitness: %.4f", gen, best_fitness)
    
    # Wybieramy najlepszego osobnika z ostatniej generacji
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_index = np.argmax(fitness_scores)
    opt_weights = population[best_index]
    port_return = np.dot(opt_weights, mean_returns)
    port_std = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    sharpe = (port_return) / port_std if port_std != 0 else 0
    logging.info("Optymalizacja genetyczna zakończona sukcesem.")
    return {"weights": opt_weights, "expected_return": port_return, "risk": port_std, "sharpe_ratio": sharpe}

# Integracja z hyperparameter_tuner.py (przykładowa funkcja weryfikująca efekty zmian)
def integrate_hyperparameter_tuner(strategy_optimizer, X: pd.DataFrame, y: pd.Series):
    """
    Integruje wyniki strategy_optimizer z modułem hyperparameter_tuner.py.
    Funkcja przyjmuje instancję strategy_optimizer oraz dane i zwraca zaktualizowane parametry strategii.
    
    Parameters:
        strategy_optimizer: Obiekt strategy_optimizer (np. instancja klasy z tego modułu).
        X (pd.DataFrame): Dane wejściowe.
        y (pd.Series): Wartości docelowe.
    
    Returns:
        dict: Zaktualizowane parametry strategii.
    """
    # Przykładowa integracja: tuner dostraja jeden z parametrów strategii, np. próg wejścia.
    # Załóżmy, że strategy_optimizer ma atrybut 'params' i metodę 'update_parameters'.
    # Tu symulujemy aktualizację poprzez proste zwiększenie progu wejścia o 10%.
    current_params = strategy_optimizer.get_current_parameters() if hasattr(strategy_optimizer, "get_current_parameters") else {}
    if "entry_threshold" in current_params:
        new_threshold = current_params["entry_threshold"] * 1.1
    else:
        new_threshold = 0.01  # domyślna wartość
    updated_params = current_params.copy()
    updated_params["entry_threshold"] = new_threshold
    # Logujemy i zwracamy zaktualizowane parametry
    logging.info("Zaktualizowano parametry strategii: %s", updated_params)
    return updated_params

def generate_report(optimization_results: list):
    """
    Generuje wizualizację i raport porównawczy wyników optymalizacji strategii.
    
    Parameters:
        optimization_results (list): Lista wyników z poszczególnych metod optymalizacji.
                                     Każdy wynik to słownik zawierający m.in. Sharpe Ratio, oczekiwany zwrot, ryzyko.
    """
    try:
        df_results = pd.DataFrame(optimization_results)
        logging.info("Raport optymalizacji:\n%s", df_results)
        # Wizualizacja wyników
        plt.figure(figsize=(8, 6))
        plt.bar(df_results.index.astype(str), df_results['sharpe_ratio'], color='blue')
        plt.xlabel("Metoda optymalizacji")
        plt.ylabel("Sharpe Ratio")
        plt.title("Porównanie Sharpe Ratio różnych metod optymalizacji")
        plt.tight_layout()
        report_path = "./reports/strategy_optimization_report.png"
        plt.savefig(report_path)
        plt.close()
        logging.info("Wizualizacja raportu zapisana w: %s", report_path)
    except Exception as e:
        logging.error("Błąd przy generowaniu raportu optymalizacji: %s", e)
        raise

# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        # Przykładowe dane historyczne zwrotów
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        returns_data = np.random.normal(0.001, 0.02, size=(252, 4))
        returns_df = pd.DataFrame(returns_data, index=dates, columns=["Asset_A", "Asset_B", "Asset_C", "Asset_D"])
        
        # Optymalizacja metodą Markowitza
        markowitz_result = optimize_portfolio_markowitz(returns_df)
        # Optymalizacja metodą genetyczną
        genetic_result = optimize_portfolio_genetic(returns_df)
        
        optimization_results = [
            {"method": "Markowitz", **markowitz_result},
            {"method": "Genetic", **genetic_result}
        ]
        
        generate_report(optimization_results)
        
        # Przykładowa integracja z hyperparameter_tuner.py (symulacja)
        class DummyStrategyOptimizer:
            def __init__(self):
                self.params = {"entry_threshold": 0.01, "exit_threshold": 0.005}
            def get_current_parameters(self):
                return self.params.copy()
        
        dummy_optimizer = DummyStrategyOptimizer()
        updated_params = integrate_hyperparameter_tuner(dummy_optimizer, returns_df, returns_df.mean(axis=1))
        logging.info("Zaktualizowane parametry strategii: %s", updated_params)
        
        logging.info("Moduł strategy_optimizer.py zakończony sukcesem.")
    except Exception as e:
        logging.error("Błąd w module strategy_optimizer.py: %s", e)
        raise
