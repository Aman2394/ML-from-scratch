from collections import Counter
import numpy as np
from core.DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=100, min_samples_split=10, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split, 
                                n_features=self.n_features)
            X_sample, y_sample = self._get_bootstrap_samples(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _get_bootstrap_samples(self, X, y):

        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _get_common_label(self, y):
        freq = Counter(y)
        most_common_label = freq.most_common(1)[0][0]
        return most_common_label
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        #[[0,1,2,0,1,0],[0,1,1,1,0,1],....n_trees]
        #swap axes
        tree_preds = np.swapaxes(tree_preds,0 ,1)
        #[[0,1,2,0,1,0,...n_trees],[0,1,1,1,0,1,...n_trees],[0,1,1,1,0,1,...n_trees],[0,1,1,1,0,1,...n_trees],[0,1,1,1,0,1,...n_trees]]
        preds = np.array([self._get_common_label(pred) for pred in tree_preds ])
        return preds
