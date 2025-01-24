# Based on the work by baydoganm/mtTrees

import time
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Node:
    """
    Node class for a Decision Tree.

    Attributes:
        right (Node): Right child node.
        left (Node): Left child node.
        column (int): Index of the feature used for splitting.
        column_name (str): Name of the feature.
        threshold (float): Threshold for the feature split.
        id (int): Identifier for the node.
        depth (int): Depth of the node in the tree.
        is_terminal (bool): Indicates if the node is a terminal node.
        prediction (numpy.ndarray): Predicted values for the node.
        count (int): Number of samples in the node.

    Methods:
        No specific methods are defined in this class.

        It will be mainly used in the construction of the tree.
    """
    def __init__(self):
        self.right = None
        self.left = None
        self.column = None
        self.column_name = None
        self.threshold = None
        self.id = None
        self.depth = None
        self.is_terminal = False
        self.prediction = None
        self.count = None

class OCDT:
    """
    Predictive Clustering Tree.

    Args:
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        split_style (str): Splitting style.
        verbose (bool): Whether to print verbose information.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        split_style (str): Splitting style (e.g. 'custom')
        verbose (bool): Whether to print verbose information.
        Tree (Node): Root node of the predictive clustering tree.

    Methods:
        buildDT(features, labels, node):
            Build the predictive clustering tree.

        fit(features, labels):
            Fit the predictive clustering tree to the data.

        nodePredictions(y):
            Calculate predictions for a node.

        applySample(features, depth, node):
            Passes one object through the decision tree and returns the prediction.

        apply(features, depth):
            Returns the node id for each X.

        get_rules(features, depth, node, rules):
            Returns the decision rules for feature selection.

        calcBestSplit(features, labels, current_label):
            Calculates the best split based on features and labels.

        calcBestSplitCustom(features, labels):
            Calculates the best custom split for features and labels.
    """
    def __init__(self, max_depth = 5,
                 min_samples_leaf = 5,
                 min_samples_split = 10,
                 split_criteria = None,
                 leaf_prediction_method = None,
                 nof_infeasibilities_method = None,
                 verbose = False,
                 use_hashmaps = True,
                 use_initial_solution = True):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.split_criteria = split_criteria
        self.nof_infeasibilities_method = nof_infeasibilities_method
        self.leaf_prediction_method = leaf_prediction_method
        self.verbose = verbose
        self.Tree = None
        self.use_hashmaps = use_hashmaps
        self.use_initial_solution = use_initial_solution

    def buildDT(self, features, labels, node):
        """
        Build the predictive clustering tree.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
            node (Node): The current node in the tree being built.
        """
        node.prediction, _ = self.split_criteria(labels.to_numpy(), features.to_numpy(), self.nof_infeasibilities_method, self.best_solution)

        node.count = labels.shape[0]
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return

        if features.shape[0] < self.min_samples_split:
            node.is_terminal = True
            return

        current_label = range(labels.shape[1])
        target = labels

        split_info, split_gain, n_cuts = self.calcBestSplitCustom(features, target)

        if n_cuts == 0:
            node.is_terminal = True
            return

        min_max_scaler = preprocessing.MinMaxScaler()
        split_gain_scaled_total = min_max_scaler.fit_transform(split_gain)[:, 0]
        mean_rank_sort = np.argsort(split_gain_scaled_total)

        splitCol = int(split_info[mean_rank_sort[0], 0])
        thresh = split_info[mean_rank_sort[0], 1]

        node.column = splitCol
        node.column_name = features.columns[splitCol]
        node.threshold = thresh

        labels_left = labels.loc[features.iloc[:,splitCol] <= thresh, :]
        labels_right = labels.loc[features.iloc[:,splitCol] > thresh, :]

        features_left = features.loc[features.iloc[:,splitCol] <= thresh]
        features_right = features.loc[features.iloc[:,splitCol] > thresh]

        # creating left and right child nodes
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.id = 2 * node.id

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.id = 2 * node.id + 1

        # splitting recursively
        self.buildDT(features_left, labels_left, node.left)
        self.buildDT(features_right, labels_right, node.right)


    def fit(self, features, labels):
        """
        Fit the predictive clustering tree to the data.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
        """
        start = time.time()
        self.features = features
        self.labels = labels
        self.preds_dict = {}
        self.Tree = Node()
        self.Tree.depth = 0
        self.Tree.id = 1
        self.best_solution = np.zeros(self.labels.shape[1])
        self.best_solution_perf = float('inf')
        self.buildDT(features, labels, self.Tree)
        leaves = self.apply(features)
        leaf_predictions = {}
        for leaf_id in np.unique(leaves):
            leaf_indices = np.where(leaves == leaf_id)[0]
            leaf_labels = labels.iloc[leaf_indices].to_numpy()
            leaf_features = features.iloc[leaf_indices].to_numpy()
            leaf_predictions[leaf_id], _ = self.leaf_prediction_method(leaf_labels, leaf_features, self.nof_infeasibilities_method, self.best_solution)
        self.leaf_predictions_df = pd.DataFrame(leaf_predictions)
        end = time.time()
        self.training_duration = end-start

    def predict(self, features):
        '''
        Returns the labels for each X
        '''
        leaves = self.apply(features)
        predictions = self.leaf_predictions_df[leaves].T

        return np.asarray(predictions)

    def predictSample(self, features, depth, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.prediction

        # if we have reached the provided depth
        if node.depth == depth:
            return node.prediction

        if features.iloc[node.column] > node.threshold:
            predicted = self.predictSample(features, depth, node.right)
        else:
            predicted = self.predictSample(features, depth, node.left)

        return predicted

    def applySample(self, features, depth, node):
        """
        Passes one object through the predictive clustering tree and returns the leaf ID.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.

        Returns:
            predicted (int): The predicted node ID.
        """

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.id

        # if we have reached the provided depth
        if node.depth == depth:
            return node.id

        if features.iloc[node.column] > node.threshold:
            predicted = self.applySample(features, depth, node.right)
        else:
            predicted = self.applySample(features, depth, node.left)

        return predicted

    def apply(self, features):
        """
        Returns the node ID for each input object.

        Args:
            features (pandas.DataFrame): The input features for multiple objects.

        Returns:
            predicted_ids (numpy.ndarray): The predicted node IDs for each input object.
        """
        predicted_ids = [self.applySample(features.loc[i], self.max_depth, self.Tree) for i in features.index]
        predicted_ids = np.asarray(predicted_ids)
        return predicted_ids

    def get_rules(self, features, depth, node, rules):
        """
        Returns the decision rules for leaf node assignment.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.
            rules (list): A list to store the decision rules.

        Returns:
            rules (list): The updated list of decision rules.
        """
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            msg = f'Ended at terminal node with ID: {node.id}'
            print(msg)
            return rules

        # if we have reached the provided depth
        if node.depth == depth:
            msg = f'Ended at depth' + str(node.depth)
            print(msg)
            return rules

        if features.iloc[:,node.column].values[0] > node.threshold:
            msg = f'Going right: Node ID: {node.id}, Rule: {features.columns[node.column]} > {node.threshold}'
            print(msg)
            rules.append({features.columns[node.column]: {'min': node.threshold}})
            rules = self.get_rules(features, depth, node.right, rules)
        else:
            msg = f'Going left: Node ID: {node.id}, Rule: {features.columns[node.column]} <= {node.threshold}'
            print(msg)
            rules.append({features.columns[node.column]: {'max': node.threshold}})
            rules = self.get_rules(features, depth, node.left, rules)

        return rules

    def calcBestSplitCustom(self, features, labels):
        evaluated_thresholds = {feat: [] for feat in range(features.shape[1])}
        n = features.shape[0]
        cut_id = 0
        n_obj = 1
        split_perf = np.zeros((n * features.shape[1], n_obj))
        split_info = np.zeros((n * features.shape[1], 2))
        for k in range(features.shape[1]):
            if self.verbose:
                print(f'Feature Index: {k}')
            x = features.iloc[:, k].to_numpy()
            y = labels.to_numpy()
            sort_idx = np.argsort(x)
            sort_x = x[sort_idx]
            sort_y = y[sort_idx, :]

            for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                xi = sort_x[i]
                prev_val = sort_x[i - 1]

                left_yi = sort_y[:i, :]
                right_yi = sort_y[i:, :]

                if self.use_hashmaps:
                    left_idx = tuple(sorted(features.iloc[sort_idx[:i]].index))
                    right_idx = tuple(sorted(features.iloc[sort_idx[i:]].index))
                    
                    left_xi = features.to_numpy()[sort_idx[:i]]
                    right_xi = features.to_numpy()[sort_idx[i:]]

                    if left_idx not in self.preds_dict:
                        left_prediction, left_perf = self.split_criteria(left_yi, left_xi, self.nof_infeasibilities_method, self.best_solution)
                        self.preds_dict[left_idx] = {'preds': left_prediction, 'perf': left_perf}
                    else:
                        left_prediction, left_perf = self.preds_dict[left_idx]['preds'], self.preds_dict[left_idx]['perf']

                    if right_idx not in self.preds_dict:
                        right_prediction, right_perf = self.split_criteria(right_yi, right_xi, self.nof_infeasibilities_method, self.best_solution)
                        self.preds_dict[right_idx] = {'preds': right_prediction, 'perf': right_perf}
                    else:
                        right_prediction, right_perf = self.preds_dict[right_idx]['preds'], self.preds_dict[right_idx]['perf']
                else:
                    left_xi = features.to_numpy()[sort_idx][:i]
                    right_xi = features.to_numpy()[sort_idx][i:]

                    left_prediction, left_perf = self.split_criteria(left_yi, left_xi, self.nof_infeasibilities_method, self.best_solution)
                    right_prediction, right_perf = self.split_criteria(right_yi, right_xi, self.nof_infeasibilities_method, self.best_solution)

                if self.use_initial_solution:
                    if left_perf < self.best_solution_perf:
                        self.best_solution_perf = left_perf
                        self.best_solution = left_prediction
                    if right_perf < self.best_solution_perf:
                        self.best_solution_perf = right_perf
                        self.best_solution = right_prediction

                left_instance_count = left_yi.shape[0]
                right_instance_count = right_yi.shape[0]

                curr_score = (left_perf * left_instance_count + right_perf * right_instance_count) / n
                threshold_val = (xi + prev_val) / 2

                if ((xi == sort_x[i - 1]) or (xi in evaluated_thresholds[k])):
                    continue
                else:
                    evaluated_thresholds[k].append(xi)

                split_perf[cut_id, 0] = curr_score
                split_info[cut_id, 0] = k
                split_info[cut_id, 1] = threshold_val

                cut_id += 1

        split_info = split_info[range(cut_id), :]
        split_gain = split_perf[range(cut_id), :]
        n_cuts = cut_id

        split_info = split_info[~np.isnan(split_gain).any(axis=1),:]
        split_gain = split_gain[~np.isnan(split_gain).any(axis=1),:]

        return split_info, split_gain, n_cuts
