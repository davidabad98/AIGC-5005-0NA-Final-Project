import pandas as pd
import yaml
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.train_model import split_train_test


class DataPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def run(self):

        # Creating an instance of the DataPreprocessor class
        dp = DataPreprocessor(file_path=self.file_path)
        # Calling the preprocess method which which will perform all the preprocessing steps
        df_preprocess = dp.preprocess()

        X_train, X_test, y_train, y_test = split_train_test(
            df_preprocess, "gold close", test_size=0.3
        )

        # Creating an instance of the feature engineering class
        fe = FeatureEngineering(X_train, X_test, y_train, y_test)
        fe.scale_data()

        return fe
