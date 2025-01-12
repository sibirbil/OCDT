import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

def return_medoid(y):
    return y[np.argmin(euclidean_distances(y).mean(axis=1))]

def return_mean(y):
    return np.asarray(y.mean(axis=0))

def calculate_mse(y, predictions):
    return ((y - predictions) ** 2).mean()

def calculate_mad(y, predictions):
    return np.mean(np.abs(y - predictions))

def calculate_poisson_deviance(y, predictions):
    return 2 * np.sum(predictions - y - y * np.log(predictions / y))

def calculate_number_of_infeasibilities(y_pred, X_test, dataset, model, ocdt_depth, target_cols, verbose=True):
    if dataset == 'class':
        cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
        nof_infeasibilities = np.sum(cumsums >= 3)
    elif dataset == 'scores':
        nof_infeasibilities = 0
        for i in range(len(y_pred)):
            if (y_pred[i][1] < 50) and (y_pred[i][2] > 0.0001):
                nof_infeasibilities += 1
            elif (round(y_pred[i][1] + y_pred[i][2], 4) < 110) and (y_pred[i][0] > 0.0001):
                nof_infeasibilities += 1
    elif dataset == 'forecasting':
        nof_infeasibilities = 0
        nof_infeasibilities += y_pred[y_pred > 100].shape[0]
        y_pred_df = pd.DataFrame(y_pred, index=X_test.index)
        y_pred_df = pd.concat([X_test.iloc[:, -3:], y_pred_df], axis=1)
        y_pred_df.columns = range(y_pred_df.shape[1])
        for i in range(1, y_pred_df.shape[1] - 2):
            nof_infeasibilities += y_pred_df[(y_pred_df.loc[:, [i, i+1]].sum(axis=1) > 70) &
                                             (y_pred_df.loc[:, i+2] > 50)].shape[0]
        for i in range(y_pred_df.shape[1] - 3):
            nof_infeasibilities += y_pred_df[(y_pred_df.loc[:, i: i + 2].sum(axis=1) > 120) &
                                             (y_pred_df.loc[:, i + 3] > 10)].shape[0]
    else:
        y_pred_df = pd.DataFrame(y_pred, columns=target_cols)
        nof_infeasibilities = y_pred_df[(y_pred_df['TARGET_FLAG'] == 0) & (y_pred_df['TARGET_AMT'] > 0)].shape[0]

    if verbose:
        print(f'Number of infeasible predictions for {model} (Depth {ocdt_depth}): {nof_infeasibilities}')

    return nof_infeasibilities

def split_criteria_with_methods(y, x, nof_infeasibilities_method, initial_solution, lagrangian_multiplier, prediction_method,
                                evaluation_method, optimization_problem, verbose=False):
    if prediction_method == 'medoid':
        predictions = return_medoid(y)
    elif prediction_method == 'optimal':
        predictions = optimization_problem(y, x, initial_solution, lagrangian_multiplier, verbose)
    else:
        predictions = return_mean(y)

    if evaluation_method == 'mse':
        split_evaluation = calculate_mse(y, predictions)
    elif evaluation_method == 'mad':
        split_evaluation = calculate_mad(y, predictions)
    else:
        split_evaluation = calculate_poisson_deviance(y, predictions)

    return predictions, split_evaluation