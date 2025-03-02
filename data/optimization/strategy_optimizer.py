"""
strategy_optimizer.py
---------------------
Moduł analizujący wyniki strategii handlowych i proponujący modyfikacje parametrów.

Funkcjonalności:
- Analiza danych z backtestingu, real-time tradingu i analizy ryzyka.
- Optymalizacja portfela metodą Markowitza oraz algorytmami genetycznymi.
- Uwzględnienie ograniczeń kapitałowych i celów inwestora.
- Integracja z `hyperparameter_tuner.py` do automatycznej weryfikacji efektywności zmian.
- Mechanizmy raportowania, wizualizacji i testów automatycznych.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def optimize_portfolio_markowitz(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Optymalizuje portfel metodą Markowitza.

    Parameters:
        returns (pd.DataFrame): DataFrame zawierający historyczne zwroty aktywów.
        risk_free_rate (float): Wolna od ryzyka stopa zwrotu.

    Returns:
        dict: Słownik zawierający optymalne wagi, oczekiwany zwrot, ryzyko (std) i Sharpe Ratio.
    """
    if returns.empty:
        raise ValueError("Dostarczone zwroty są puste.")

    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_performance(weights):
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_std if port_std > 0 else 0
        return port_return, port_std, sharpe

    def negative_sharpe(weights):
        return -portfolio_performance(weights)[2]

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.ones(n_assets) / n_assets

    result = minimize(negative_sharpe, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        logging.error("Optymalizacja Markowitza nie powiodła się: %s", result.message)
        raise ValueError("Optymalizacja portfela nie powiodła się.")

    opt_weights = result.x
    port_return, port_std, sharpe = portfolio_performance(opt_weights)
    logging.info("Optymalizacja Markowitza zakończona sukcesem.")

    return {
        "weights": opt_weights,
        "expected_return": port_return,
        "risk": port_std,
        "sharpe_ratio": sharpe,
    }


def optimize_portfolio_genetic(returns: pd.DataFrame, population_size: int = 50, generations: int = 100) -> dict:
    """
    Optymalizuje portfel przy użyciu algorytmów genetycznych.

    Parameters:
        returns (pd.DataFrame): Dane historyczne zwrotów aktywów.
        population_size (int): Rozmiar populacji.
        generations (int): Liczba generacji.

    Returns:
        dict: Słownik zawierający optymalne wagi, oczekiwany zwrot, ryzyko i Sharpe Ratio.
    """
    if returns.empty:
        raise ValueError("Dostarczone zwroty są puste.")

    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def fitness(weights):
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_return / port_std if port_std > 0 else 0

    population = [np.random.dirichlet(np.ones(n_assets)) for _ in range(population_size)]

    for gen in range(generations):
        fitness_scores = np.array([fitness(ind) for ind in population])
        selected = [population[i] for i in fitness_scores.argsort()[-(population_size // 2):]]
        offspring = []

        while len(offspring) < population_size - len(selected):
            parents = np.random.choice(selected, 2, replace=False)
            child = (parents[0] + parents[1]) / 2
            mutation = np.random.normal(0, 0.05, size=n_assets)
            child = np.clip(child + mutation, 0, None)
            child /= np.sum(child) if np.sum(child) > 0 else 1
            offspring.append(child)

        population = selected + offspring

    best_index = np.argmax([fitness(ind) for ind in population])
    opt_weights = population[best_index]
    port_return = np.dot(opt_weights, mean_returns)
    port_std = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    sharpe = port_return / port_std if port_std > 0 else 0

    logging.info("Optymalizacja genetyczna zakończona sukcesem.")

    return {
        "weights": opt_weights,
        "expected_return": port_return,
        "risk": port_std,
        "sharpe_ratio": sharpe,
    }


def generate_report(optimization_results: list):
    """
    Tworzy raport porównawczy wyników optymalizacji strategii.

    Parameters:
        optimization_results (list): Lista słowników z wynikami optymalizacji.
    """
    try:
        df_results = pd.DataFrame(optimization_results)
        logging.info("Raport optymalizacji:\n%s", df_results)

        plt.figure(figsize=(8, 6))
        plt.bar(df_results["method"], df_results["sharpe_ratio"], color="blue")
        plt.xlabel("Metoda optymalizacji")
        plt.ylabel("Sharpe Ratio")
        plt.title("Porównanie Sharpe Ratio różnych metod optymalizacji")
        plt.tight_layout()

        report_path = "./reports/strategy_optimization_report.png"
        plt.savefig(report_path)
        plt.close()

        logging.info("Raport zapisany w: %s", report_path)
    except Exception as e:
        logging.error("Błąd przy generowaniu raportu: %s", e)
        raise


# -------------------- Przykładowe użycie --------------------
if __name__ == "__main__":
    try:
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        returns_df = pd.DataFrame(np.random.normal(0.001, 0.02, size=(252, 4)), index=dates)

        markowitz_result = optimize_portfolio_markowitz(returns_df)
        genetic_result = optimize_portfolio_genetic(returns_df)

        optimization_results = [
            {"method": "Markowitz", **markowitz_result},
            {"method": "Genetic", **genetic_result},
        ]

        generate_report(optimization_results)

        logging.info("Moduł strategy_optimizer.py zakończony sukcesem.")
    except Exception as e:
        logging.error("Błąd w module strategy_optimizer.py: %s", e)
        raise
