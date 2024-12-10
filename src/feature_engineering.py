import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.shared_functions import save_dataframe_to_csv


class FeatureEngineering:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

    def scale_data(self):
        """Scales the input features and target variable."""
        self.X_train_scaled = self.scaler_x.fit_transform(self.X_train.values)
        self.X_test_scaled = self.scaler_x.transform(self.X_test.values)

        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1))
        self.y_test_scaled = self.scaler_y.transform(self.y_test.reshape(-1, 1))
