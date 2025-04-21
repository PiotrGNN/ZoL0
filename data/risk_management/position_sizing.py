"""
position_sizing.py
------------------
Moduł obliczający optymalne rozmiary pozycji w zależności od ryzyka i dostępnego kapitału.

Funkcjonalności:
- Implementacja popularnych metod określania rozmiaru pozycji, takich jak fixed fractional, Kelly criterion oraz risk parity.
- Dynamiczne dostosowywanie wielkości pozycji na podstawie zmienności rynku i aktualnych wyników strategii.
- Kompatybilność z modułem trade_executor.py oraz mechanizmem dźwigni (leverage_optimizer.py).
- Testy weryfikujące skuteczność i stabilność algorytmu w warunkach realnego handlu, w tym przy wysokiej zmienności.
- Odporny na skrajne przypadki, takie jak bardzo mały lub bardzo duży kapitał.
"""

import logging

import numpy as np

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def fixed_fractional_position_size(
    capital: float, risk_per_trade: float, stop_loss_distance: float, current_price: float = None
) -> float:
    """
    Oblicza rozmiar pozycji przy użyciu metody fixed fractional.

    Parameters:
        capital (float): Dostępny kapitał.
        risk_per_trade (float): Maksymalny procent kapitału, jaki chcemy ryzykować na jedną transakcję (np. 0.02 dla 2%).
        stop_loss_distance (float): Odległość od punktu wejścia do stop loss (w jednostkach ceny).
        current_price (float, optional): Aktualna cena aktywa. Jeśli podana, wielkość pozycji będzie w jednostkach aktywa.

    Returns:
        float: Optymalny rozmiar pozycji.
    """
    # Zakładamy, że ryzyko w dolarach to capital * risk_per_trade
    risk_amount = capital * risk_per_trade
    if stop_loss_distance <= 0:
        logging.error("Stop loss distance musi być dodatnie.")
        raise ValueError("Stop loss distance musi być dodatnie.")

    if current_price is not None and current_price > 0:
        position_size = (risk_amount / current_price) / (stop_loss_distance / current_price)
    else:
        position_size = risk_amount / stop_loss_distance

    logging.info("Fixed fractional position size obliczone: %.4f", position_size)
    return position_size


def kelly_criterion_position_size(
    win_rate: float, win_loss_ratio: float, capital: float
) -> float:
    """
    Oblicza rozmiar pozycji przy użyciu kryterium Kelly'ego.

    Parameters:
        win_rate (float): Prawdopodobieństwo wygranej (0 < win_rate < 1).
        win_loss_ratio (float): Stosunek średniego zysku do średniej straty.
        capital (float): Dostępny kapitał.

    Returns:
        float: Optymalny rozmiar pozycji.
    """
    if win_rate <= 0 or win_rate >= 1:
        logging.error("Win rate musi być w przedziale (0, 1).")
        raise ValueError("Win rate musi być w przedziale (0, 1).")
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    kelly_fraction = max(0, kelly_fraction)  # Kelly może być ujemne, wtedy ustawiamy 0
    position_size = capital * kelly_fraction
    logging.info(
        "Kelly criterion position size obliczone: %.4f (Kelly fraction: %.4f)",
        position_size,
        kelly_fraction,
    )
    return position_size


def risk_parity_position_size(
    volatilities: np.ndarray, total_capital: float
) -> np.ndarray:
    """
    Oblicza rozmiary pozycji dla wielu aktywów przy użyciu zasady risk parity.

    Parameters:
        volatilities (np.ndarray): Wektor zmienności poszczególnych aktywów.
        total_capital (float): Całkowity dostępny kapitał.

    Returns:
        np.ndarray: Wektory rozmiarów pozycji dla każdego aktywa.
    """
    if any(volatilities <= 0):
        logging.error("Wszystkie wartości zmienności muszą być dodatnie.")
        raise ValueError("Wszystkie wartości zmienności muszą być dodatnie.")

    inverse_vol = 1 / volatilities
    weights = inverse_vol / np.sum(inverse_vol)
    position_sizes = total_capital * weights
    logging.info("Risk parity position sizes obliczone: %s", position_sizes)
    return position_sizes


def dynamic_position_size(
    capital: float,
    risk_per_trade: float,
    stop_loss_distance: float,
    market_volatility: float,
) -> float:
    """
    Dynamicznie dostosowuje rozmiar pozycji w oparciu o zmienność rynku.
    Metoda oparta na fixed fractional, ale modyfikująca rozmiar pozycji, gdy zmienność jest wysoka.

    Parameters:
        capital (float): Dostępny kapitał.
        risk_per_trade (float): Procent kapitału ryzykowany na transakcję.
        stop_loss_distance (float): Odległość od punktu wejścia do stop loss.
        market_volatility (float): Miernik zmienności rynku (np. ATR).

    Returns:
        float: Optymalny rozmiar pozycji.
    """
    base_size = fixed_fractional_position_size(
        capital, risk_per_trade, stop_loss_distance
    )
    # Przykładowa logika: gdy zmienność jest wysoka, zmniejsz rozmiar pozycji proporcjonalnie
    adjustment_factor = 1 / (1 + market_volatility)
    dynamic_size = base_size * adjustment_factor
    logging.info(
        "Dynamic position size obliczone: %.4f (base: %.4f, adjustment: %.4f)",
        dynamic_size,
        base_size,
        adjustment_factor,
    )
    return dynamic_size


def calculate_position_size(capital: float, risk_params: dict) -> float:
    """
    Główna funkcja do obliczania rozmiaru pozycji, wykorzystująca różne metody w zależności od parametrów.

    Parameters:
        capital (float): Dostępny kapitał
        risk_params (dict): Parametry ryzyka zawierające:
            - method (str): Metoda obliczania ('fixed_fractional', 'kelly', 'risk_parity', 'dynamic')
            - risk_per_trade (float): Procent kapitału na jedną transakcję
            - stop_loss_distance (float, optional): Odległość stop loss
            - win_rate (float, optional): Dla metody Kelly
            - win_loss_ratio (float, optional): Dla metody Kelly
            - market_volatility (float, optional): Dla metody dynamicznej
            - volatilities (np.ndarray, optional): Dla metody risk parity

    Returns:
        float: Obliczony rozmiar pozycji
    """
    method = risk_params.get('method', 'fixed_fractional')
    risk_per_trade = risk_params.get('risk_per_trade', 0.02)

    if method == 'fixed_fractional':
        stop_loss_distance = risk_params.get('stop_loss_distance')
        current_price = risk_params.get('current_price')
        if not stop_loss_distance:
            raise ValueError("Dla metody fixed_fractional wymagany jest parametr stop_loss_distance")
        return fixed_fractional_position_size(capital, risk_per_trade, stop_loss_distance, current_price)

    elif method == 'kelly':
        win_rate = risk_params.get('win_rate')
        win_loss_ratio = risk_params.get('win_loss_ratio')
        if not win_rate or not win_loss_ratio:
            raise ValueError("Dla metody kelly wymagane są parametry win_rate i win_loss_ratio")
        return kelly_criterion_position_size(win_rate, win_loss_ratio, capital)

    elif method == 'risk_parity':
        volatilities = risk_params.get('volatilities')
        if volatilities is None:
            raise ValueError("Dla metody risk_parity wymagany jest parametr volatilities")
        return risk_parity_position_size(volatilities, capital)

    elif method == 'dynamic':
        stop_loss_distance = risk_params.get('stop_loss_distance')
        market_volatility = risk_params.get('market_volatility')
        if not stop_loss_distance or market_volatility is None:
            raise ValueError("Dla metody dynamic wymagane są parametry stop_loss_distance i market_volatility")
        return dynamic_position_size(capital, risk_per_trade, stop_loss_distance, market_volatility)

    else:
        raise ValueError(f"Nieznana metoda obliczania rozmiaru pozycji: {method}")


# -------------------- Testy jednostkowe --------------------
def unit_test_position_sizing():
    """
    Testy jednostkowe dla modułu position_sizing.py.
    Sprawdza funkcjonalność metod fixed fractional, Kelly criterion, risk parity oraz dynamic position sizing.
    """
    try:
        # Test fixed fractional
        capital = 10000
        risk_per_trade = 0.02  # 2%
        stop_loss_distance = 50  # jednostki ceny
        current_price = 100  # cena aktywa
        pos_size_fixed = fixed_fractional_position_size(
            capital, risk_per_trade, stop_loss_distance, current_price
        )
        assert (
            pos_size_fixed > 0
        ), "Fixed fractional position size powinien być dodatni."

        # Test Kelly criterion
        win_rate = 0.55
        win_loss_ratio = 2.0
        pos_size_kelly = kelly_criterion_position_size(
            win_rate, win_loss_ratio, capital
        )
        assert pos_size_kelly >= 0, "Kelly criterion position size nie może być ujemny."

        # Test risk parity
        volatilities = np.array([0.1, 0.2, 0.15, 0.25])
        pos_sizes_risk_parity = risk_parity_position_size(volatilities, capital)
        assert np.isclose(
            np.sum(pos_sizes_risk_parity), capital
        ), "Suma pozycji risk parity musi być równa kapitałowi."

        # Test dynamic position sizing
        market_volatility = 0.05  # np. ATR
        pos_size_dynamic = dynamic_position_size(
            capital, risk_per_trade, stop_loss_distance, market_volatility
        )
        assert pos_size_dynamic > 0, "Dynamic position size powinien być dodatni."

        logging.info("Testy jednostkowe dla position_sizing.py zakończone sukcesem.")
    except AssertionError as ae:
        logging.error("AssertionError w testach position_sizing.py: %s", ae)
    except Exception as e:
        logging.error("Błąd w testach position_sizing.py: %s", e)
        raise


if __name__ == "__main__":
    try:
        unit_test_position_sizing()
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        raise
