import os
import sys

import yaml
from scripts.data_pipeline import DataPipeline
from src.model import GoldPricePredictionModel

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the raw data file
data_path = os.path.join(BASE_DIR, "data", "raw", "financial_regression.csv")

# Initialize DataPipeline with test data
data_pipeline = DataPipeline(file_path=data_path)

# Read Hyperparams
learning_rate = 0.001
num_epochs = 200

# Preprocess and retrieve dataframe, independent (X) and dependent (y) variables
fe = data_pipeline.run()

# Bringing in the algorithm
# Assuming X_train_selected, X_test_selected, y_train_converted, y_test_converted are available
model = GoldPricePredictionModel(
    fe.X_train_scaled,
    fe.X_test_scaled,
    fe.y_train_scaled,
    fe.y_test_scaled,
    scaler_y=fe.scaler_y,
    learning_rate=learning_rate,
    epochs=num_epochs,
)
model.build_model()  # Building the model
model.train_model()  # Training the model
model.plot_loss()  # Plotting the training loss
model.predict()  # Making predictions
model.plot_actual_vs_predicted()  # Plotting actual vs predicted values
model.evaluate_model()  # Evaluating model performance
