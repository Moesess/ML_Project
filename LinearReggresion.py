import numpy as np

class LinearRegression():

    def __init__(self, learning_rate=0.001, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        number_of_samples, number_of_features = X.shape
        self.weights = np.zeros(number_of_features)
        self.bias = 0

        for _ in range(self.number_of_iterations):
            model_predictions = np.dot(X, self.weights) + self.bias
            dw = (1/number_of_samples) * np.dot(X.T, (model_predictions - y))
            db = (1/number_of_samples) * np.sum(model_predictions - y)
            
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
