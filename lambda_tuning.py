import os
import copy
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from apps import *
from library.utils import *
from library.ocdt import OCDT

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

base_folder = os.getcwd()

if __name__ == '__main__':
    ocdt_min_samples_split = 10
    ocdt_min_samples_leaf = 5
    number_of_folds = 5
    verbose = False
    ocdt_depth_list = [12]
    evaluation_method_list = ['mse']
    prediction_method_leaf_list = ['medoid', 'optimal', 'lagrangian']
    lagrangian_multiplier_list = [100, 1000, 10000]

    perf_df = pd.DataFrame()
    for ocdt_depth in ocdt_depth_list:
        optimization_problem = formulate_and_solve_lp_scores_data
        feature_cols = ['gender', 'race/ethnicity', 'parental level of education',
                        'lunch', 'test preparation course']
        target_cols = ['math score', 'reading score', 'writing score']
        full_df = pd.read_csv(f'{base_folder}/data/constrained_exams.csv')

        features_df = full_df[feature_cols]
        targets_df = full_df[target_cols]
        num_rows = features_df.shape[0]
        num_targets = targets_df.shape[1]

        kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=0)
        for cv_fold, (tr_idx, te_idx) in enumerate(kf.split(features_df)):
            print(f'Fold: {cv_fold}')

            features_df = pd.get_dummies(features_df, columns=features_df.columns, drop_first=True, dtype=int)

            X_train, y_train = features_df.iloc[tr_idx], targets_df.iloc[tr_idx]
            X_test, y_test = features_df.iloc[te_idx], targets_df.iloc[te_idx]
            for lagrangian_multiplier in lagrangian_multiplier_list:
                for prediction_method_leaf in prediction_method_leaf_list:
                    for evaluation_method in evaluation_method_list:
                        print("==============")
                        print(f'Lagrangian Multiplier: {lagrangian_multiplier}')
                        print(f'Leaf Prediction: {prediction_method_leaf}')

                        nof_infeasibilities_method = lambda y, x: calculate_number_of_infeasibilities(y, x, 'scores',
                                                                                'OCDT', ocdt_depth, target_cols, verbose)
                        split_criteria = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                                lagrangian_multiplier, 'optimal', evaluation_method, optimization_problem, verbose)
                        
                        if prediction_method_leaf == 'optimal':
                            lagrangian_multiplier_leaf = 0
                        else:
                            lagrangian_multiplier_leaf = copy.deepcopy(lagrangian_multiplier)
                        
                        leaf_prediction_method = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                                lagrangian_multiplier_leaf, prediction_method_leaf, evaluation_method, optimization_problem, verbose)

                        tree = OCDT(max_depth=ocdt_depth, min_samples_leaf=ocdt_min_samples_leaf, min_samples_split=ocdt_min_samples_split,
                                    split_criteria=split_criteria, leaf_prediction_method=leaf_prediction_method,
                                    nof_infeasibilities_method=nof_infeasibilities_method, verbose=verbose)
                        tree.fit(X_train, y_train)

                        y_pred = tree.predict(X_test)
                        y_pred_df = pd.DataFrame(y_pred)
                        y_pred_df['leaf_id'] = tree.apply(X_test)
                        y_pred_df = y_pred_df.drop_duplicates()
                        ocdt_mse = mean_squared_error(y_test, y_pred)
                        print(f'OCDT MSE: {ocdt_mse}')
                        ocdt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, 'scores', 'OCDT', ocdt_depth, target_cols)

                        perf_df = pd.concat([perf_df, pd.DataFrame({'fold': cv_fold, 'lagrangian_multiplier': lagrangian_multiplier,
                                                                    'depth': ocdt_depth, 'min_samples_leaf': ocdt_min_samples_leaf,
                                                                    'min_samples_split': ocdt_min_samples_split, 
                                                                    'evaluation_method': evaluation_method,
                                                                    'prediction_method_leaf': prediction_method_leaf,
                                                                    'mse': ocdt_mse, 'nof_infeasibilities': ocdt_nof_infeasibilities,
                                                                    'training_duration': tree.training_duration}, index=[0])])
                        perf_df.to_csv('data/results/perf_df_scores_df_lambda_experimentation.csv', index=False)
    