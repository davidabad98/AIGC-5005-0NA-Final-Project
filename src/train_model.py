import pandas as pd
from sklearn.model_selection import train_test_split
 
# Specify the columns to include
specified_columns = [
    "oil close",
    "nasdaq high",
    "CPI",
    "GDP",
    "silver low",
    "days_since_start",
    "us_rates_%",
    "palladium high",
    "oil high",
    "nasdaq open"
]

def split_train_test(df: pd.DataFrame, target_column: str = 'gold close', test_size: float = 0.3):
    x = df.drop(columns=[target_column])
    y = df[target_column]    
    
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Select the specified columns directly from the datasets
    X_train_selected = X_train[specified_columns]
    X_test_selected = X_test[specified_columns]
    
    y_train_converted = y_train.to_numpy() if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train
    y_test_converted = y_test.to_numpy() if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test
    
    return X_train_selected, X_test_selected, y_train_converted, y_test_converted
