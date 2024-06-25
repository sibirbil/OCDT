import os
import time
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from apps import *
from library.utils import *
from library.ocdt import OCDT

base_folder = os.getcwd()

if __name__ == '__main__':
    ocdt_min_samples_split = 10
    ocdt_min_samples_leaf = 5
    number_of_folds = 5
    verbose = False
    ocdt_depth_list = [15]
    class_target_size = 7
    class_size = 1000
    dataset_list = ['class'] # class, cars, scores
    evaluation_method_list = ['mse'] # mse, mad, poisson
    prediction_method_leaf_list = ['medoid'] # medoid, optimal
    prediction_method_list = ['mean'] # mean, medoid, optimal

    perf_df = pd.DataFrame()
    for dataset in dataset_list:
        for ocdt_depth in ocdt_depth_list:
            if dataset == 'class':
                optimization_problem = formulate_and_solve_lp_class_data
                target_cols = [f'Course{id + 1}' for id in range(class_target_size)]
                lagrangian_multiplier = 1500

                feature_cols = ['EnrolledElectiveBefore', 'GradeAvgPrevElec',
                                'Grade', 'Major', 'Class', 'GradePerm']

                full_df = pd.read_csv(f'{base_folder}/data/class_df_size_{class_size}_targets_{class_target_size}.csv')

                features_df = full_df[feature_cols]
                targets_df = full_df[target_cols]
            elif dataset == 'forecasting':
                optimization_problem = formulate_and_solve_lp_forecasting_data
                full_df = pd.read_csv(f'{base_folder}/data/forecasting.csv')
                feature_cols = [f'Feature {id + 1}' for id in range(30)]
                target_cols = [f'Target {id + 1}' for id in range(6)]
                lagrangian_multiplier = 1000
                features_df = full_df[feature_cols]
                targets_df = full_df[target_cols]
            elif dataset == 'scores':
                optimization_problem = formulate_and_solve_lp_scores_data
                feature_cols = ['gender', 'race/ethnicity', 'parental level of education',
                                'lunch', 'test preparation course']
                target_cols = ['math score', 'reading score', 'writing score']
                lagrangian_multiplier = 500
                full_df = pd.read_csv(f'{base_folder}/data/constrained_exams.csv')

                features_df = full_df[feature_cols]
                targets_df = full_df[target_cols]
            else:
                optimization_problem = formulate_and_solve_lp_cars_data
                full_df = pd.read_csv(f'{base_folder}/data/insurance_evaluation_data.csv').drop(columns=['INDEX']).dropna()[:500]

                target_cols = ['TARGET_FLAG', 'TARGET_AMT']
                targets_df = full_df[target_cols]
                lagrangian_multiplier = 20000000
                feature_cols = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'HOME_VAL', 'TRAVTIME',
                                'BLUEBOOK', 'TIF', 'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE']
                features_df = full_df[feature_cols]
                currency_cols = features_df.select_dtypes('object').columns
                features_df.loc[:, currency_cols] = features_df[currency_cols].replace('[\$,]', '', regex=True).astype(float)

            num_rows = features_df.shape[0]
            num_targets = targets_df.shape[1]

            kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=0)
            for cv_fold, (tr_idx, te_idx) in enumerate(kf.split(features_df)):
                print(f'Fold: {cv_fold}')

                # One-hot encoding for categorical features
                if dataset in ['scores']:
                    features_df = pd.get_dummies(features_df, columns=features_df.columns, drop_first=True, dtype=int)

                X_train, y_train = features_df.iloc[tr_idx], targets_df.iloc[tr_idx]
                X_test, y_test = features_df.iloc[te_idx], targets_df.iloc[te_idx]

                regressor = DecisionTreeRegressor(random_state=20, min_samples_leaf=ocdt_min_samples_leaf,
                                                min_samples_split=ocdt_min_samples_split, max_depth=ocdt_depth)
                start = time.time()
                regressor.fit(X_train, y_train)
                end = time.time()
                y_pred_sklearn = regressor.predict(X_test)
                dt_mse = mean_squared_error(y_test, y_pred_sklearn)
                print(f'DT MSE: {dt_mse}')
                dt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred_sklearn, X_test, dataset, 'DT',
                                                                            regressor.get_depth(), target_cols)

                perf_df = pd.concat([perf_df, pd.DataFrame({'data': dataset, 'fold': cv_fold, 'depth': ocdt_depth,
                                                            'min_samples_leaf': ocdt_min_samples_leaf,
                                                            'min_samples_split': ocdt_min_samples_split,
                                                            'prediction_method': 'sklearn',
                                                            'prediction_method_leaf': 'sklearn',
                                                            'evaluation_method': 'sklearn',
                                                            'mse': dt_mse, 'nof_infeasibilities': dt_nof_infeasibilities,
                                                            'training_duration': end - start}, index=[0])])

                for evaluation_method in evaluation_method_list:
                    for prediction_method in prediction_method_list:
                        for prediction_method_leaf in prediction_method_leaf_list:
                            print("==============")
                            print(f'Evaluation: {evaluation_method}')
                            print(f'Split Prediction: {prediction_method}')
                            print(f'Leaf Prediction: {prediction_method_leaf}')

                            nof_infeasibilities_method = lambda y, x: calculate_number_of_infeasibilities(y, x, dataset,
                                                                                    'OCDT', ocdt_depth, target_cols, verbose)
                            lagrangian_multiplier = 0
                            split_criteria = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                                    lagrangian_multiplier, prediction_method, evaluation_method, optimization_problem, verbose)
                            leaf_prediction_method = lambda y, x, nof_infeasibilities_method: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                                    lagrangian_multiplier, prediction_method_leaf, evaluation_method, optimization_problem, verbose)

                            tree = OCDT(max_depth=ocdt_depth, min_samples_leaf=ocdt_min_samples_leaf, min_samples_split=ocdt_min_samples_split,
                                        split_criteria=split_criteria, leaf_prediction_method=leaf_prediction_method,
                                        nof_infeasibilities_method=nof_infeasibilities_method, verbose=verbose)
                            tree.fit(X_train, y_train)
                            # tree.get_rules(X_train.iloc[1:40], tree.max_depth, tree.Tree, [])
                            y_pred = tree.predict(X_test)
                            y_pred_df = pd.DataFrame(y_pred)
                            y_pred_df['leaf_id'] = tree.apply(X_test)
                            y_pred_df = y_pred_df.drop_duplicates()
                            ocdt_mse = mean_squared_error(y_test, y_pred)
                            print(f'OCDT MSE: {ocdt_mse}')
                            ocdt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, dataset, 'OCDT', ocdt_depth, target_cols)

                            perf_df = pd.concat([perf_df, pd.DataFrame({'data': dataset, 'fold': cv_fold, 'depth': ocdt_depth,
                                                                        'min_samples_leaf': ocdt_min_samples_leaf,
                                                                        'min_samples_split': ocdt_min_samples_split,
                                                                        'prediction_method': prediction_method,
                                                                        'prediction_method_leaf': prediction_method_leaf,
                                                                        'evaluation_method': evaluation_method,
                                                                        'mse': ocdt_mse, 'nof_infeasibilities': ocdt_nof_infeasibilities,
                                                                        'training_duration': tree.training_duration}, index=[0])])
                            perf_df.to_csv(f'data/perf_df_class_df_experimentation.csv', index=False)
