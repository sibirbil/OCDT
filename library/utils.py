import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

def return_medoid(y):
    return y[np.argmin(euclidean_distances(y).mean(axis=1))]

def return_mean(y):
    return np.asarray(y.mean(axis=0))

def return_median(y):
    return np.asarray(np.median(y,axis=0))

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
            if (y_pred[i][1] < .50) and (y_pred[i][2] > 0.0001):
                nof_infeasibilities += 1
            elif (round(y_pred[i][1] + y_pred[i][2], 4) < 1.10) and (y_pred[i][0] > 0.0001):
                nof_infeasibilities += 1
    elif dataset == 'synthetic':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i]) - num_targets*0.5) > 1e-4:
                nof_infeasibilities += 1
            elif (sum(y_pred[i][j] for j in range(int(num_targets/2))) - 0.4*int(num_targets/2) > 1e-4): 
                nof_infeasibilities += 1
            elif(sum(y_pred[i][j] for j in range(int(num_targets/2)+1,num_targets)) - 0.6*(int(num_targets/2)+1) > 1e-4):
                nof_infeasibilities += 1
    elif dataset == 'synthetic_manifold':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i][j] for j in range(num_targets//2)) - 1) > 1e-4:
                nof_infeasibilities += 1
            else:
                t = 1
                for k in range(num_targets//2,num_targets,2):
                    if abs(y_pred[i][k-1] - y_pred[i][k] - (t)/10) > 1e-4:
                        nof_infeasibilities += 1
                        break
                    t+=1 
    elif dataset == 'hts':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if (sum(1 for x in y_pred[i] if x != 0) > 13-9): 
                nof_infeasibilities += 1
            elif abs(sum(y_pred[i][j] for j in range(num_targets)) - 15) > 1e-4: 
                nof_infeasibilities += 1
            else:
                for j in range(num_targets):
                    if y_pred[i][j] < 0:
                        nof_infeasibilities += 1
                        break
                    
    elif dataset == 'hts_global':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if (sum(1 for x in y_pred[i][:12] if x != 0) > 12-6): 
                nof_infeasibilities += 1
            else:
                if y_pred[i][12] != y_pred[i][0]+y_pred[i][1]+y_pred[i][2]:
                    nof_infeasibilities += 1
                elif y_pred[i][13] != y_pred[i][3]+y_pred[i][4]+y_pred[i][5]:
                    nof_infeasibilities += 1
                elif y_pred[i][14] != y_pred[i][6]+y_pred[i][7]+y_pred[i][8]:
                    nof_infeasibilities += 1
                elif y_pred[i][15] != y_pred[i][9]+y_pred[i][10]+y_pred[i][11]:
                    nof_infeasibilities += 1
                elif y_pred[i][16] != y_pred[i][12]+y_pred[i][13]+y_pred[i][14]+y_pred[i][15]:
                    nof_infeasibilities += 1
                else:
                    for j in range(12):
                        if y_pred[i][j] > 10:
                            nof_infeasibilities += 1
                            break
                        elif y_pred[i][j] < 0:
                            nof_infeasibilities += 1
                            break
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
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i][j] for j in range(num_targets//2)) - 1) > 1e-4:
                nof_infeasibilities += 1
            else:
                t = 1
                for k in range(num_targets//2,num_targets,2):
                    if abs(y_pred[i][k-1] - y_pred[i][k] - (t)/10) > 1e-4:
                        nof_infeasibilities += 1
                        break
                    t+=1 

    if verbose:
        print(f'Number of infeasible predictions for {model} (Depth {ocdt_depth}): {nof_infeasibilities}')

    return nof_infeasibilities

def split_criteria_with_methods(y, x, nof_infeasibilities_method, initial_solution, lagrangian_multiplier, prediction_method,
                                evaluation_method, optimization_problem, verbose=False, bforce = True):
    if not bforce:
        return [], None
    if prediction_method == 'medoid':
        predictions = return_medoid(y)
    elif prediction_method == 'optimal':
        predictions = optimization_problem(y, x, initial_solution, lagrangian_multiplier, verbose)
    elif prediction_method == 'median':
        predictions = return_median(y)
    else:
        predictions = return_mean(y)

    if evaluation_method == 'mse':
        split_evaluation = calculate_mse(y, predictions)
    elif evaluation_method == 'mad':
        split_evaluation = calculate_mad(y, predictions)
    else:
        split_evaluation = calculate_poisson_deviance(y, predictions)

    return predictions, split_evaluation