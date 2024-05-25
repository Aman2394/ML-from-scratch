import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    """
    # TODO
    """
    def __init__(self, lr=0.001, num_iter=1000):

        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iter):

            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X, threshold=0.5):

        linear_pred = np.dot(X, self.weights) + self.bias
        preds = sigmoid(linear_pred)

        preds_label = [0 if y<threshold else 1 for y in preds]

        return preds_label




