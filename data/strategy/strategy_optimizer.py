"""
strategy_optimizer.py
---------------------
Moduł analizujący wyniki strategii handlowych i proponujący modyfikacje parametrów.
"""

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def optimize_portfolio_markowitz(
    returns: pd.DataFrame, risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Optymalizuje portfel przy użyciu metody Markowitza.

    Parameters:
        returns (pd.DataFrame): DataFrame z historycznymi zwrotami poszczególnych aktywów.
        risk_free_rate (float): Wolna od ryzyka stopa zwrotu.

    Returns:
        dict: Słownik zawierający optymalne wagi, oczekiwany zwrot, ryzyko i Sharpe Ratio.
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

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.full(n_assets, 1.0 / n_assets)

    result = minimize(
        negative_sharpe,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if result.success:
        opt_weights = result.x
        port_return, port_std, sharpe = portfolio_performance(opt_weights)
        logging.info("Optymalizacja Markowitza zakończona sukcesem.")
        return {
            "weights": opt_weights,
            "expected_return": port_return,
            "risk": port_std,
            "sharpe_ratio": sharpe,
        }
    else:
        logging.error("Optymalizacja Markowitza nie powiodła się.")
        raise ValueError("Optymalizacja portfela nie powiodła się.")


def optimize_portfolio_genetic(
    returns: pd.DataFrame, population_size: int = 50, generations: int = 100
) -> Dict[str, Any]:
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
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return port_return / port_std if port_std != 0 else 0

    population = [
        np.random.dirichlet(np.ones(n_assets)) for _ in range(population_size)
    ]

    for gen in range(generations):
        fitness_scores = np.array([fitness(ind) for ind in population])
        selected_indices = fitness_scores.argsort()[-(population_size // 2) :]
        selected = [population[i] for i in selected_indices]

        offspring = []
        while len(offspring) < population_size - len(selected):
            parents = np.random.choice(selected, 2, replace=False)
            child = (parents[0] + parents[1]) / 2.0 + np.random.normal(
                0, 0.05, size=n_assets
            )
            child = np.clip(child, 0, None)
            child /= child.sum() if child.sum() != 0 else 1
            offspring.append(child)

        population = selected + offspring

        if gen % 10 == 0:
            logging.info(
                "Generacja %d, najlepsze fitness: %.4f", gen, np.max(fitness_scores)
            )

    best_index = np.argmax(fitness_scores)
    opt_weights = population[best_index]
    port_return = np.dot(opt_weights, mean_returns)
    port_std = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    sharpe = port_return / port_std if port_std != 0 else 0
    logging.info("Optymalizacja genetyczna zakończona sukcesem.")
    return {
        "weights": opt_weights,
        "expected_return": port_return,
        "risk": port_std,
        "sharpe_ratio": sharpe,
    }


def generate_report(optimization_results: List[Dict[str, Any]]):
    """
    Generuje wizualizację i raport porównawczy wyników optymalizacji strategii.

    Parameters:
        optimization_results (list): Lista wyników optymalizacji z metod Markowitza i genetycznej.
    """
    try:
        df_results = pd.DataFrame(optimization_results)
        logging.info("Raport optymalizacji:\n%s", df_results)

        plt.figure(figsize=(8, 6))
        plt.bar(df_results["method"], df_results["sharpe_ratio"], color="blue")
        plt.xlabel("Metoda optymalizacji")
        plt.ylabel("Sharpe Ratio")
        plt.title("Porównanie metod optymalizacji portfela")
        plt.tight_layout()
        plt.savefig("./reports/strategy_optimization_report.png")
        plt.close()
        logging.info("Raport zapisany w: ./reports/strategy_optimization_report.png")
    except Exception as e:
        logging.error("Błąd przy generowaniu raportu optymalizacji: %s", e)
        raise


if __name__ == "__main__":
    try:
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        returns_data = np.random.normal(0.001, 0.02, size=(252, 4))
        returns_df = pd.DataFrame(
            returns_data,
            index=dates,
            columns=["Asset_A", "Asset_B", "Asset_C", "Asset_D"],
        )

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
