import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from apps import formulate_and_solve_lp_scores_data
from library.utils import *
from library.ocdt import OCDT

SEED = 0
np.random.seed(SEED)

base_folder = os.getcwd()

if __name__ == '__main__':
    ocdt_depth = 15
    ocdt_min_samples_split = 10
    ocdt_min_samples_leaf = 5
    number_of_folds = 5
    verbose = False
    evaluation_method = 'mse' # mse, mad, poisson
    prediction_method_leaf = 'optimal' # medoid, optimal
    prediction_method = 'optimal' # mean, medoid, optimal

    perf_df = pd.DataFrame()
    optimization_problem = formulate_and_solve_lp_scores_data
    feature_cols = ['gender', 'race/ethnicity', 'parental level of education',
                    'lunch', 'test preparation course']
    target_cols = ['math score', 'reading score', 'writing score']
    lagrangian_multiplier = 0
    full_df = pd.read_csv(f'{base_folder}/data/constrained_exams.csv')

    features_df = full_df[feature_cols]
    features_df = pd.get_dummies(features_df, columns=features_df.columns, drop_first=True, dtype=int) # for scores data
    targets_df = full_df[target_cols]

    num_rows = features_df.shape[0]
    num_targets = targets_df.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(features_df, targets_df, test_size = 0.2, random_state=SEED)

    nof_infeasibilities_method = lambda y, x: calculate_number_of_infeasibilities(y, x, 'scores','OCDT', ocdt_depth, target_cols, verbose)
    split_criteria = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
            lagrangian_multiplier, prediction_method, evaluation_method, optimization_problem, verbose)
    leaf_prediction_method = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
            lagrangian_multiplier, prediction_method_leaf, evaluation_method, optimization_problem, verbose)

    tree = OCDT(max_depth=ocdt_depth, min_samples_leaf=ocdt_min_samples_leaf, min_samples_split=ocdt_min_samples_split,
                split_criteria=split_criteria, leaf_prediction_method=leaf_prediction_method,
                nof_infeasibilities_method=nof_infeasibilities_method, verbose=verbose)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    ocdt_mse = mean_squared_error(y_test, y_pred)
    print(f'OCDT MSE: {ocdt_mse}')
    ocdt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, 'scores', 'OCDT', ocdt_depth, target_cols)

    perf_df = pd.concat([perf_df, pd.DataFrame({'depth': ocdt_depth, 'min_samples_leaf': ocdt_min_samples_leaf,
                                                'min_samples_split': ocdt_min_samples_split,
                                                'prediction_method': prediction_method,
                                                'prediction_method_leaf': prediction_method_leaf,
                                                'evaluation_method': evaluation_method,
                                                'mse': ocdt_mse, 'nof_infeasibilities': ocdt_nof_infeasibilities,
                                                'training_duration': tree.training_duration}, index=[0])])
    perf_df.to_csv(f'data/perf_df.csv', index=False)
    y_pred_df.to_csv(f'data/preds.csv', index=False)
