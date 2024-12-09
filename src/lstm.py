import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


class lstm:
    def __init__(self, X_train, y_train, X_test, y_test, units=64, epochs=200, batch_size=16, patience=50, monitor='val_loss'):
        # Initialize hyperparameters and dataset
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.monitor = monitor
        self.model = None

    def build_model(self):
        """Builds and compiles the LSTM model."""
        self.model = Sequential()
        self.model.add(LSTM(units=self.units, input_shape=(self.X_train.shape[1], 1), return_sequences=True))
        self.model.add(LSTM(units=self.units//2, return_sequences=True))
        self.model.add(LSTM(units=self.units//4))
        self.model.add(Dense(1))
        self.model.compile(loss="mse", optimizer="adam")
        
        print("Model built and compiled.")


    def train_model(self):
        """Trains the model on the training data with early stopping."""
        callback = EarlyStopping(monitor=self.monitor, patience=self.patience, verbose=1, mode="auto")
        
        # Reshape X_train for LSTM input
        X_train_reshaped = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        
        # Train the model
        self.history = self.model.fit(
            X_train_reshaped, self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[callback],
            validation_data=(self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1), self.y_test)
        )
        print("Model training completed.")


    def predict(self):
        """Generates predictions on the test set."""
        # Reshape X_test for LSTM input
        X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        # Predict the gold prices
        self.y_pred = self.model.predict(X_test_reshaped).flatten()
        print("Predictions completed.")