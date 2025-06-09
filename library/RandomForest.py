import sys
import os
import time

import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from ocdt import OCDT
import pandas as pd
from utils import *
from sklearn.metrics import mean_squared_error


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from apps import *

class RandomForestOCDT:
    def __init__(self, n_estimators=10, random_state=42, max_samples=None, max_features='sqrt', **ocdt_params):
        """
        Random Forest using OCDT as base learner.
        
        Parameters:
        - n_estimators: Number of trees in the forest
        - max_features: Number of features to consider for each split ('sqrt', 'log2', or int)
        - ocdt_params: Dictionary of hyperparameters for OCDT
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.max_samples = max_samples
        self.ocdt_params = ocdt_params if ocdt_params else {}
        self.trees = []

    def _get_max_features(self, n_features):
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError("Invalid value for max_features")
    
    def fit(self, X, y):
        """
        Train the random forest using bootstrapped samples.
        
        !! Send the original dataset that is not binary hot encoded !!
        """
        start = time.time()
        n_sam, n_features = X.shape
        m_features = self._get_max_features(n_features)
        max_samples = self.max_samples or n_sam
        random_state = self.random_state
        
        self.trees = []
        np.random.seed(random_state)
        for i in range(self.n_estimators):
            X_sample, y_sample = resample(X, y, replace=False, n_samples=max_samples, random_state = random_state)
            feature_indices = np.random.choice(n_features, m_features, replace=False)
            X_sample = X_sample.reset_index(drop=True)
            y_sample = y_sample.reset_index(drop=True)
            
            X_sample = X_sample.iloc[:,feature_indices]
            categorical_cols = X_sample.select_dtypes(include=['object', 'category']).columns
            X_sample = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=True, dtype=int)

            tree = OCDT(**self.ocdt_params)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))
            print(f'Random Forest estimator {i} is done in {tree.training_duration} seconds!!!')
        end = time.time()
        self.training_duration = end-start
    
    def predict(self, X,numtarget):
        """
        Predict using the average of the OCDT trees.
        """
        predictions = np.zeros((X.shape[0], self.n_estimators,numtarget))
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_sample = X.iloc[:,feature_indices]
            categorical_cols = X_sample.select_dtypes(include=['object', 'category']).columns
            X_sample = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=True, dtype=int)
            predictions[:, i] = tree.predict(X_sample)
        
        return np.mean(predictions, axis=1)
    
    def score(self, X, y):
        """
        Computes accuracy on the test set.
        
        :param X: Test features.
        :param y: True labels.
        :return: Accuracy score.
        """
        _, numtarget = y.shape
        y_pred = self.predict(X,numtarget)
        y_sample = y
        return mean_squared_error(y_sample, y_pred)







