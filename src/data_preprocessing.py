import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from src.shared_functions import save_dataframe_to_csv

warnings.filterwarnings("ignore")


class DataPreprocessor:

    def __init__(self, file_path: str, debug: bool = False):
        """
        Initializes the DataPreprocessor with the path to the data file.

        Parameters:
        - file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.data = pd.DataFrame()
        self.debug = debug

    def load_data(
        self, file_path: str, delimiter: str = ",", na_values: list = ["NA", "NULL"]
    ) -> pd.DataFrame:
        """
        Loads the dataset from a specified file path with custom NA values.

        Parameters:
            file_path (str): Path to the data file.
            delimiter (str): Delimiter used in the file. Default is ','.
            header (int): Row number to use as column names. Default is 0.
            na_values (list): List of values to treat as NA/NaN. Default is ['NA', 'NULL'].
            encoding (str): File encoding. Default is 'utf-8'.

        Returns:
            pd.DataFrame: Loaded dataset with '?' values replaced by NaN.
        """
        self.data = pd.read_csv(file_path, delimiter=delimiter, na_values=na_values)
        self.data.replace("?", np.nan, inplace=True)
        if self.debug:
            print("Data loaded successfully with shape:", self.data.shape)
        return self.data
