from collections import Counter
import numpy as np


def euclidean_distance(x1, x2):
    """
    #TODO
    """
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    """
    #TODO
    """
    
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        """
        #TODO
        """
        self._X_train = X
        self._y_train = y
    
    def predict(self, X):
        """
        # TODO
        """
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):

        distances = [euclidean_distance(x, x_train) for x_train in self._X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labes = [self._y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labes).most_common()

        return most_common[0][0]