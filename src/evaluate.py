import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, history):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.history = history

    def plot_error_vs_epoch(self):
        # Plotting loss vs epoch
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title("Error vs Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_actual_vs_predicted(self):
        # Reshape input to fit model
        X_test_reshaped = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)

        # Predict with the model
        y_pred = self.model.predict(X_test_reshaped)

        # Plotting Actual vs Predicted
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title("Actual vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Gold Price")
        plt.legend()
        plt.show()