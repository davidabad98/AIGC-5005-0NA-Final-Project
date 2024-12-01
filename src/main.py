import os
import sys


# Add the project root to the Python path
project_root = os.path.abspath(
    "..."
)  # Adjust ".." if your notebooks are more deeply nested
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering




# Test the preprocessing
file_path = (
    "./data/raw/financial_regression.csv"  # Adjust the path based on your directory structure
)

# Creating an instance of the DataPreprocessor class
dp = DataPreprocessor(file_path=file_path)

# Calling the preprocess method which which will perform all the preprocessing steps
df_preprocess = dp.preprocess()

# Creating an instance of the feature engineering class
ft = FeatureEngineering(df=df_preprocess, dependent_variable='gold close')

df_transformed = ft.data_transformation()