import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


class GoldPricePredictionModel:
    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        units=128,
        dropout_rate=0.2,
        batch_size=32,
        epochs=200,
        learning_rate=0.001,
        scaler_y=MinMaxScaler(feature_range=(0, 1)),
    ):
        # Initializing data and hyperparameters
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.units = units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scaler_y = scaler_y

        self.model = None
        self.history = None

    def build_model(self):
        """Builds the LSTM model."""
        # Input layer
        lstm_input = tf.keras.layers.Input(
            shape=(self.X_train.shape[1], 1), name="lstm_input"
        )

        # First LSTM layer
        x = tf.keras.layers.LSTM(self.units, return_sequences=True, name="first_layer")(
            lstm_input
        )
        x = tf.keras.layers.Dropout(self.dropout_rate)(
            x
        )  # Dropout to reduce overfitting

        # Second LSTM layer (added layer)
        x = tf.keras.layers.LSTM(
            self.units // 2, return_sequences=False, name="second_layer"
        )(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(
            x
        )  # Dropout after the second LSTM layer

        # Dense output layer
        x = tf.keras.layers.Dense(1, activation="linear", name="dense_layer")(x)

        # Create and compile the model
        self.model = tf.keras.models.Model(inputs=lstm_input, outputs=x)
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=adam, loss="mse")
        self.model.summary()

    def train_model(self):
        """Trains the model using early stopping."""
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50
        )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            callbacks=[early_stopping],
        )

    def plot_loss(self):
        """Plots training and validation loss."""
        plt.plot(self.history.history["loss"], label="Training loss")
        plt.plot(self.history.history["val_loss"], label="Validation loss")
        plt.legend()
        plt.title("Loss log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    def predict(self):
        """Makes predictions on the test set."""
        y_pred_scaled = self.model.predict(self.X_test)
        # Inverse transform the scaled predictions
        self.y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        self.y_test_actual = self.scaler_y.inverse_transform(self.y_test)

    def plot_actual_vs_predicted(self):
        """Plots actual vs predicted gold close price."""
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.y_test_actual, label="Actual Gold Close Price", color="blue", alpha=0.6
        )
        plt.plot(
            self.y_pred, label="Predicted Gold Close Price", color="red", alpha=0.6
        )
        plt.xlabel("Time Steps (or Data Points)")
        plt.ylabel("Gold Close Price")
        plt.title("Actual vs Predicted Gold Close Prices")
        plt.legend()
        plt.show()

    def evaluate_model(self):
        """Evaluates the model performance using MSE, MAE, and R-squared."""
        mse = mean_squared_error(self.y_test_actual, self.y_pred)
        mae = mean_absolute_error(self.y_test_actual, self.y_pred)
        r2 = r2_score(self.y_test_actual, self.y_pred)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared: {r2}")
