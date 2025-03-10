import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR  # Import directory path from config
from loguru import logger


def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from a CSV file in the 'raw' directory.

    Parameters:
        filename (str): The name of the CSV file to load (must be located in the raw data directory).

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    # Construct the full file path
    file_path = RAW_DATA_DIR / filename

    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File {filename} not found in {RAW_DATA_DIR}")

    logger.info(f"Loading data from {file_path}")

    # Load the data into a pandas DataFrame
    data = pd.read_csv(file_path)

    return data
