from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd

class FeatureSelectionWithRFE:
    def __init__(self, X_train, y_train, X_test, n_features_to_select=10):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.n_features_to_select = n_features_to_select

    def perform_feature_selection(self):
        """Performs RFE for feature selection using LightGBM."""
        lgbm_estimator = LGBMRegressor()
        rfe = RFE(estimator=lgbm_estimator, n_features_to_select=self.n_features_to_select)
        
        # Fit RFE
        rfe.fit(self.X_train, self.y_train)
        
        # Get the mask of selected features
        selected_features_mask = rfe.support_
        selected_features_indices = np.where(selected_features_mask)[0]
        
        # Create new datasets with selected features using pandas iloc for proper indexing
        X_train_selected = self.X_train.iloc[:, selected_features_indices]
        X_test_selected = self.X_test.iloc[:, selected_features_indices]
        print('Selected features are index {}'.format(selected_features_indices))

        # Converting to numpy array from pandas dataframe
        X_train_selected = X_train_selected.to_numpy().reshape(X_train_selected.shape[0], X_train_selected.shape[1], 1)
        X_test_selected = X_test_selected.to_numpy().reshape(X_test_selected.shape[0], X_test_selected.shape[1], 1)
        
        return X_train_selected, X_test_selected