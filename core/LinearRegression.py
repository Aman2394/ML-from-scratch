import numpy as np

class LinearRegression:
    
    """
    # TODO
    """

    def __init__(self, lr=0.001, num_iter=1000):
        
        self.num_iter = num_iter
        self.lr = lr
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        """
        # TODO
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iter):
            y_pred = np.dot(X,self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    
    def predict(self, X):
        """
        # TODO
        """
        preds = np.dot(X,self.weights) + self.bias
        return preds





