"""
zrobcsv.py
----------
Skrypt do konwersji danych z różnych formatów (JSON, Excel, XML) do formatu CSV.

Funkcjonalności:
- Automatyczne wykrywanie kolumn i mapowanie ich na standardowy schemat:
  timestamp, open, high, low, close, volume.
- Walidacja struktury danych oraz logowanie błędów w razie niezgodności formatu.
- Przetwarzanie dużych plików w trybie streamingowym (np. przy użyciu iteratorów w Pandas).
- Parametry konfiguracyjne (separator, kodowanie znaków) przekazywane przez argumenty wiersza poleceń.
- Testy jednostkowe weryfikujące poprawność konwersji.
"""

import argparse
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET

import pandas as pd

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Standardowy schemat kolumn
STANDARD_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

def detect_file_format(file_path: str) -> str:
    """
    Wykrywa format pliku na podstawie rozszerzenia.

    Parameters:
        file_path (str): Ścieżka do pliku.

    Returns:
        str: Format pliku ('json', 'excel', 'xml', 'csv').
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".json"]:
        return "json"
    elif ext in [".xls", ".xlsx"]:
        return "excel"
    elif ext in [".xml"]:
        return "xml"
    elif ext in [".csv"]:
        return "csv"
    else:
        raise ValueError(f"Nieobsługiwany format pliku: {ext}")

def load_data(file_path: str, file_format: str, chunksize: int = None) -> pd.DataFrame:
    """
    Ładuje dane z pliku do DataFrame.

    Parameters:
        file_path (str): Ścieżka do pliku.
        file_format (str): Format pliku.
        chunksize (int, optional): Rozmiar partii dla trybu streamingowym.

    Returns:
        pd.DataFrame: Załadowane dane.
    """
    try:
        if file_format == "csv":
            if chunksize:
                # Ładuj w trybie streamingowym i połącz wyniki
                chunks = pd.read_csv(file_path, chunksize=chunksize)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
        elif file_format == "json":
            # Zakładamy, że plik jest w formacie JSON lub JSON Lines
            try:
                # Próbujemy załadować jako JSON Lines
                df = pd.read_json(file_path, lines=True)
            except ValueError:
                # W przeciwnym razie zwykły JSON
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = pd.json_normalize(data)
        elif file_format == "excel":
            df = pd.read_excel(file_path)
        elif file_format == "xml":
            # Ładowanie XML i konwersja na listę słowników
            tree = ET.parse(file_path)
            root = tree.getroot()
            data = []
            for elem in root:
                record = {}
                for child in elem:
                    record[child.tag] = child.text
                data.append(record)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Format {file_format} nie jest obsługiwany.")
        logging.info("Dane załadowane z pliku: %s", file_path)
        return df
    except Exception as e:
        logging.error("Błąd przy ładowaniu danych z %s: %s", file_path, e)
        raise

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapuje kolumny wejściowe na standardowy schemat:
    timestamp, open, high, low, close, volume.

    Parameters:
        df (pd.DataFrame): DataFrame z danymi wejściowymi.

    Returns:
        pd.DataFrame: DataFrame z przemapowanymi kolumnami.
    """
    try:
        # Mapowanie kolumn: szukamy kolumn, których nazwy (po lower-case) zawierają słowa kluczowe
        mapping = {}
        for std_col in STANDARD_COLUMNS:
            for col in df.columns:
                if std_col in col.lower():
                    mapping[col] = std_col
                    break
        missing = set(STANDARD_COLUMNS) - set(mapping.values())
        if missing:
            error_msg = f"Brakujące kolumny wymagane do konwersji: {missing}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        # Zmiana nazw kolumn
        df_renamed = df.rename(columns=mapping)
        # Zachowujemy tylko standardowe kolumny
        df_standard = df_renamed[STANDARD_COLUMNS]
        logging.info("Mapowanie kolumn zakończone. Użyte mapowanie: %s", mapping)
        return df_standard
    except Exception as e:
        logging.error("Błąd przy mapowaniu kolumn: %s", e)
        raise

def convert_to_csv(
    input_file: str,
    output_file: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    chunksize: int = None,
):
    """
    Konwertuje dane z pliku wejściowego (JSON, Excel, XML, CSV) do formatu CSV według standardowego schematu.

    Parameters:
        input_file (str): Ścieżka do pliku wejściowego.
        output_file (str): Ścieżka, pod którą zapisany zostanie plik CSV.
        delimiter (str): Separator używany w pliku CSV.
        encoding (str): Kodowanie znaków.
        chunksize (int, optional): Rozmiar partii dla trybu streamingowego.
    """
    try:
        file_format = detect_file_format(input_file)
        df = load_data(input_file, file_format, chunksize=chunksize)
        df_mapped = map_columns(df)
        df_mapped.to_csv(output_file, index=False, sep=delimiter, encoding=encoding)
        logging.info("Konwersja zakończona. Plik CSV zapisany w: %s", output_file)
    except Exception as e:
        logging.error("Błąd podczas konwersji do CSV: %s", e)
        raise

# -------------------- Testy jednostkowe --------------------
def unit_test_conversion():
    """
    Testuje konwersję z różnych formatów do CSV.
    Tworzy przykładowe dane w formacie JSON, Excel i XML,
    konwertuje je do CSV, a następnie sprawdza poprawność przemapowania kolumn.
    """
    import tempfile

    # Przykładowe dane
    sample_data = [
        {
            "timestamp": "2023-01-01 09:30:00",
            "open": 100,
            "high": 105,
            "low": 99,
            "close": 104,
            "volume": 1500,
        },
        {
            "timestamp": "2023-01-02 09:30:00",
            "open": 104,
            "high": 110,
            "low": 102,
            "close": 109,
            "volume": 2000,
        },
    ]

    # Test JSON
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as temp_json:
        json.dump(sample_data, temp_json)
        temp_json_path = temp_json.name

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".csv", delete=False
    ) as temp_csv:
        temp_csv_path = temp_csv.name

    convert_to_csv(temp_json_path, temp_csv_path)
    df_json_csv = pd.read_csv(temp_csv_path)
    assert set(df_json_csv.columns) == set(
        STANDARD_COLUMNS
    ), "JSON konwersja - kolumny nie są zgodne ze standardem."

    # Test Excel
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".xlsx", delete=False
    ) as temp_excel:
        temp_excel_path = temp_excel.name
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_excel(temp_excel_path, index=False)

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".csv", delete=False
    ) as temp_csv2:
        temp_csv_path2 = temp_csv2.name

    convert_to_csv(temp_excel_path, temp_csv_path2)
    df_excel_csv = pd.read_csv(temp_csv_path2)
    assert set(df_excel_csv.columns) == set(
        STANDARD_COLUMNS
    ), "Excel konwersja - kolumny nie są zgodne ze standardem."

    # Test XML
    # Tworzymy prosty XML ze strukturą odpowiadającą sample_data
    xml_content = "<root>"
    for record in sample_data:
        xml_content += "<record>"
        for key, value in record.items():
            xml_content += f"<{key}>{value}</{key}>"
        xml_content += "</record>"
    xml_content += "</root>"
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".xml", delete=False
    ) as temp_xml:
        temp_xml.write(xml_content)
        temp_xml_path = temp_xml.name

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".csv", delete=False
    ) as temp_csv3:
        temp_csv_path3 = temp_csv3.name

    convert_to_csv(temp_xml_path, temp_csv_path3)
    df_xml_csv = pd.read_csv(temp_csv_path3)
    assert set(df_xml_csv.columns) == set(
        STANDARD_COLUMNS
    ), "XML konwersja - kolumny nie są zgodne ze standardem."

    # Sprzątanie tymczasowych plików
    os.remove(temp_json_path)
    os.remove(temp_csv_path)
    os.remove(temp_excel_path)
    os.remove(temp_csv_path2)
    os.remove(temp_xml_path)
    os.remove(temp_csv_path3)
    logging.info("Testy jednostkowe konwersji zakończone sukcesem.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Konwertuje dane z JSON, Excel, XML do formatu CSV."
    )
    parser.add_argument("input_file", type=str, help="Ścieżka do pliku wejściowego.")
    parser.add_argument(
        "output_file", type=str, help="Ścieżka do pliku CSV wyjściowego."
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Separator w pliku CSV (domyślnie ',').",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Kodowanie pliku CSV (domyślnie 'utf-8').",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Rozmiar partii do przetwarzania dużych plików.",
    )

    args = parser.parse_args()

    try:
        convert_to_csv(
            args.input_file,
            args.output_file,
            delimiter=args.delimiter,
            encoding=args.encoding,
            chunksize=args.chunksize,
        )
    except Exception as e:
        logging.error("Konwersja zakończona błędem: %s", e)
        sys.exit(1)

    # Opcjonalnie, uruchom testy jednostkowe
    try:
        unit_test_conversion()
    except Exception as e:
        logging.error("Testy jednostkowe nie powiodły się: %s", e)
        sys.exit(1)
