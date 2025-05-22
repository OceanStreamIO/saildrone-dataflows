import pandas as pd


def load_values_from_xlsx(file_path):
    cal = pd.read_excel(file_path, sheet_name="Sheet1", header=1, usecols="A:D")

    # Correct the spelling and set column names
    cal.columns = ["Variable", "38k short pulse", "38k long pulse", "200k short pulse"]

    return cal