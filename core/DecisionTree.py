
from collections import Counter
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X ,y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):

        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria 
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # we will stop building the tree
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_indxs = np.random.choice(n_feats, self.n_features, replace=False)
        # find the best split 
        best_feature, best_threshold = self._find_best_split(X, y, feat_indxs)
        # create child nodes 
        l_indxs, r_indxs = self._split(X[:,best_feature],best_threshold)

        # call the function again with new sub trees
        left = self._grow_tree(X[l_indxs,:],y[l_indxs], depth+1)
        right = self._grow_tree(X[r_indxs,:],y[r_indxs], depth+1)

        return Node(best_feature, best_threshold, left, right)
    
    def _find_best_split(self, X, y, feat_indxs):
        
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_ind in feat_indxs:
            X_column = X[:,feat_ind]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                ig = self._get_information_gain(X_column, y, threshold)
                if ig > best_gain:
                    best_gain = ig
                    split_idx = feat_ind
                    split_threshold = threshold
        return split_idx, split_threshold
    
    def _get_information_gain(self, X, y, threshold):

        # IG = exp(parent) - weighted avg exp(childs)
        parent_entropy = self._get_entropy(y)

        # get child indx
        l_indxs, r_indxs = self._split(X,threshold)
        if len(l_indxs) == 0 or len(r_indxs) == 0:
            return 0
        
        # calculate child entropy
        l_len, r_len = len(l_indxs), len(r_indxs)
        l_entropy, r_entropy = self._get_entropy(y[l_indxs]), self._get_entropy(y[r_indxs])
        child_entropy = (1/len(y)) * (l_len * l_entropy + r_len *r_entropy)

        ig = parent_entropy - child_entropy
        return ig

    
    def _split(self, X, split_threshold):
        l_indxs = np.argwhere(X <= split_threshold).flatten()
        r_indxs = np.argwhere(X > split_threshold).flatten()
        return l_indxs, r_indxs
    
    def _get_entropy(self, y):
        hist = np.bincount(y)
        px = hist/ len(y)
        entropy = -np.sum([p *  np.log(p) for p in px if p >0])
        return entropy

    def _most_common_label(self, y):
        
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        
        return value

    def predict(self, X):

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)


