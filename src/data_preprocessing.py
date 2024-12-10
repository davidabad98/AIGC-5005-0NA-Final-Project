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
    
    def backfill(self, df:pd.DataFrame, columns_backfill: list = ['us_rates_%','CPI','GDP']) -> pd.DataFrame:
        '''
        This function is used to backfill the given columns in order to fill up.
        ''' 
        df[columns_backfill] = df[columns_backfill].fillna(method='bfill')
        return df

    def interpolate_backfill(self, df:pd.DataFrame, columns_to_interploate: list = ['us_rates_%','CPI','GDP','usd_chf','eur_usd']) -> pd.DataFrame:
        '''
        This function will interpolate with respect to time and backfill first rows becase
        first columns do not interpolate as there are not values to start with
        '''
        #Converting date column from object to datetime
        df['date']  = pd.to_datetime(df['date'])
        # We need to set the date column to index for interpolation to work
        df.set_index('date', inplace=True)
        df[columns_to_interploate] = df[columns_to_interploate].interpolate(method='time')
        df.reset_index(inplace=True)
        # First few rows are filled up using backfill
        df[columns_to_interploate] = df[columns_to_interploate].fillna(method='bfill')

        return df
    
    def drop_null(self, df: pd.DataFrame, columns: list = ['date','gold close'] ):
        '''
        Here we will remove the rows where the date and target variable is missing
        '''
        df = df.dropna(subset=columns)
        return df
    
    # This function checks for null values
    def check_null(self,df:pd.DataFrame):
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            print('There are nulls in the dataset')
        else:
            print('No null values found')
            
            
    def date_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        RNNs require numerical or structured inputs. A raw datetime object or string cannot 
        be directly interpreted.
        The date column should be converted into numeric features or representations, 
        such as elapsed time or cyclical features, which RNNs can understand.
        '''
        # This will be one of our columns. we are removing date but we will have number of days elapsed
        # for each entry which will give us a idea of time spent.
        df['days_since_start'] = (df['date']-df['date'].min()).dt.days

        # Drop raw date column
        df = df.drop(columns=['date'])

        return df

    def delete_columns(self, df: pd.DataFrame, delete_columns: list = ['gold open','gold high','gold low']) -> pd.DataFrame:
        df = df.drop(columns=delete_columns)
        return df

    # This function will run all the functions in the class
    def preprocess(self):
        df = self.load_data(self.file_path)
        df = df.sort_values('date')
        print('Data has been loaded')
        df_inter = self.interpolate_backfill(df)
        df_null = self.drop_null(df_inter)
        self.check_null(df_null)
        df_transformation = self.date_transformation(df_null)
        del_df = self.delete_columns(df_transformation)
        return df_transformation
