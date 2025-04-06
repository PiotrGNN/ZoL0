#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do generowania hierarchicznego drzewka katalogów i plików,
z możliwością wykluczenia wybranych folderów (np. venv, __pycache__, .git).

Wywołanie (w katalogu z tym plikiem lub przez podanie pełnej ścieżki):
    python tree_view.py [opcje]

Przykłady:
    python tree_view.py               # drzewko od bieżącego katalogu
    python tree_view.py --path /projekt/ZoL0 --exclude .git venv
    python tree_view.py --path . --output drzewo.txt
    python tree_view.py --help        # wyświetla pomoc
"""

import argparse
import logging
import os
import sys
from typing import List, Set

__author__ = "Your Name"
__version__ = "1.0.0"
__license__ = "MIT"

# Domyślna lista katalogów do wykluczenia
DEFAULT_EXCLUDES = {".git", ".idea", ".vscode", "__pycache__", "venv", ".DS_Store"}


def setup_logging(level: str = "INFO") -> None:
    """
    Konfiguruje podstawowe logowanie na standardowe wyjście.

    :param level: Poziom logowania (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_tree(
    start_path: str,
    excludes: Set[str],
    prefix: str = "",
    is_last: bool = True,
) -> List[str]:
    """
    Rekurencyjnie buduje listę wierszy (linijek) reprezentujących
    strukturę katalogów w formie drzewka.

    :param start_path: Ścieżka do katalogu, od którego zaczynamy generowanie drzewka.
    :param excludes: Zbiór nazw katalogów do wykluczenia (dokładne dopasowanie).
    :param prefix: Prefiks tekstowy używany przy rysowaniu gałązek (rekurencja).
    :param is_last: Informacja, czy dany element jest ostatni na swoim poziomie (rekurencja).
    :return: Lista wierszy (stringów), które można potem wypisać w konsoli lub zapisać do pliku.
    """
    lines: List[str] = []

    base_name = os.path.basename(os.path.normpath(start_path))
    connector = "└──" if is_last else "├──"
    lines.append(f"{prefix}{connector} {base_name}")

    # Nowy prefiks dla poziomów zagnieżdżenia
    new_prefix = prefix + ("    " if is_last else "│   ")

    # Próba odczytu zawartości katalogu
    try:
        entries = os.listdir(start_path)
    except PermissionError:
        logging.warning("Brak uprawnień do odczytu: %s", start_path)
        return lines
    except FileNotFoundError:
        logging.error("Nie znaleziono ścieżki: %s", start_path)
        return lines

    # Sortujemy, żeby wyświetlać w kolejności alfabetycznej
    entries.sort()

    child_items = []
    for entry in entries:
        full_path = os.path.join(start_path, entry)
        if os.path.isdir(full_path):
            child_items.append((entry, full_path, True))
        else:
            child_items.append((entry, full_path, False))

    # Budujemy nową listę, wykluczając katalogi wymienione w excludes
    filtered_child_items = []
    for entry, full_path, is_dir in child_items:
        # Jeśli to folder i nazwa jest w excludes, pomijamy
        if is_dir and entry in excludes:
            logging.debug("Pomijam katalog %r (wykluczony)", entry)
            continue
        filtered_child_items.append((entry, full_path, is_dir))

    for idx, (entry, full_path, is_dir) in enumerate(filtered_child_items):
        child_is_last = idx == len(filtered_child_items) - 1
        if is_dir:
            lines.extend(build_tree(full_path, excludes, new_prefix, child_is_last))
        else:
            file_connector = "└──" if child_is_last else "├──"
            lines.append(f"{new_prefix}{file_connector} {entry}")

    return lines


def generate_tree_view(
    path: str,
    excludes: Set[str],
) -> List[str]:
    """
    Generuje pełną listę linii drzewa dla wskazanej ścieżki (folderu).
    Dodatkowo obsługuje przypadek, gdy ścieżka jest plikiem.

    :param path: Ścieżka startowa.
    :param excludes: Zbiór nazw katalogów do wykluczenia.
    :return: Lista linijek tekstu z drzewkiem.
    """
    if not os.path.exists(path):
        logging.error("Ścieżka %s nie istnieje!", path)
        return []

    if os.path.isfile(path):
        # Jeśli ktoś podał plik zamiast folderu – tylko wyświetlamy nazwę pliku
        return [f"└── {os.path.basename(path)}"]

    # Jeśli to katalog, budujemy rekurencyjnie
    return build_tree(path, excludes, prefix="", is_last=True)


def main() -> None:
    """
    Główna funkcja uruchamiająca cały skrypt:
    - Parsuje argumenty,
    - Konfiguruje logowanie,
    - Generuje drzewko,
    - Wyświetla lub zapisuje wynik.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generuje hierarchiczne drzewko katalogów i plików, "
            "z możliwością wykluczenia niektórych folderów (venv, __pycache__, itp.)."
        ),
        epilog="Autor: Your Name",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Ścieżka początkowa (domyślnie bieżący katalog).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Lista dodatkowych nazw folderów do wykluczenia (poza domyślnymi).",
    )
    parser.add_argument(
        "--no-default-exclude",
        action="store_true",
        help="Jeśli podane, nie używamy wbudowanej listy wykluczeń (.__pycache__, .git, venv itd.).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Ścieżka do pliku wyjściowego. Jeśli nie podana, wyświetla na stdout.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Poziom logowania (domyślnie INFO).",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Zbuduj zestaw wykluczeń
    if args.no_default_exclude:
        excludes = set(args.exclude)
    else:
        excludes = DEFAULT_EXCLUDES.union(args.exclude)

    # Generujemy drzewko
    lines = generate_tree_view(args.path, excludes)

    # Wyświetlamy lub zapisujemy
    if args.output:
        try:
            with open(args.output, mode="w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            logging.info("Zapisano drzewko do pliku: %s", args.output)
        except OSError as e:
            logging.error("Błąd zapisu do pliku %s: %s", args.output, e)
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
