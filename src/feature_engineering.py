import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.shared_functions import save_dataframe_to_csv


class FeatureEngineering:
    def __init__(self, df: pd.DataFrame, dependent_variable: str):
        self.scaler = StandardScaler()
        self.df = df
        self.dependent_variable = dependent_variable

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

        # We will capture the cyclic nature of months with sin and cosine
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Drop raw date column
        df = df.drop(columns=['date','month'])

        return df


    def standardize_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Standardizes the feature matrix using StandardScaler.

        Parameters:
        X (pd.DataFrame): The feature matrix to be standardized.

        Returns:
        pd.DataFrame: The standardized feature matrix.

        """

        data2 = data.drop(target_column, axis=1)
        X_continuous_scaled = self.scaler.fit_transform(data2)
        X_continuous_scaled = pd.DataFrame(
            X_continuous_scaled, columns=data2.columns, index=data2.index
        )
        X_continuous_scaled[target_column] = data[target_column].values

        return X_continuous_scaled
    
    def delete_columns(self, df: pd.DataFrame, delete_columns: list = ['gold open','gold high','gold low']) -> pd.DataFrame:
        df = df.drop(columns=delete_columns)
        return df
    
    # This is the method where the previous two methods will run, we only need to call this
    def data_transformation(self) -> pd.DataFrame:
        df_date = self.date_transformation(self.df)
        df_del = self.delete_columns(df=df_date)
        df_final = self.standardize_features(df_del, self.dependent_variable)
        
        save_dataframe_to_csv(
            df_final, "./data/clean/transformed.csv"
        )
        print('Data is saved')
        return df_final