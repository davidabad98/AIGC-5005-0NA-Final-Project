import os
import pickle
import sys

import yaml

from scripts.data_pipeline import DataPipeline
from src.model import GoldPricePredictionModel

# Load configuration
config = None
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: config.yaml file not found.")
except yaml.YAMLError as e:
    print(f"Error reading config.yaml: {e}")

# Check if config was loaded successfully
if config is None:
    print("Failed to load configuration. Please check config.yaml file.")
    exit(1)

# Initialize DataPipeline with test data
data_pipeline = DataPipeline(file_path=config["data"]["training_data_path"])

# Read Hyperparams
learning_rate = config["model"]["learning_rate"]
num_epochs = config["model"]["num_iterations"]

# Preprocess and retrieve feature engineering object
fe = data_pipeline.run()

# Bringing in the algorithm
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


# Save the model
with open(config["data"]["output_dir"], "wb") as f:
    pickle.dump(model, f)
