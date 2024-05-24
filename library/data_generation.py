import os
import random
import numpy as np
import pandas as pd

np.random.seed(0)
random.seed(0)

base_folder = os.getcwd()

def generate_class_data(target_size = 7, data_size = 2000):
    feature_cols = ['EnrolledElectiveBefore', 'GradeAvgPrevElec',
                                'Grade', 'Major', 'Class', 'GradePerm']
    target_cols = [f'Course{id+1}' for id in range(target_size)]

    feature_0 = np.random.randint(low=2, size=data_size) # EnrolledElectiveBefore

    feature_1 = np.random.normal(loc=60, scale=15, size=data_size) # GradeAvgPrevElec
    feature_1 = np.where(feature_1 > 100, 100, feature_1)
    feature_1 = np.where(feature_1 < 0, 0, feature_1)
    feature_1 = np.where(feature_0 == 0, 0, feature_1)
    feature_1 = np.round(feature_1, 2)

    feature_2 = np.random.normal(loc=70, scale=10, size=data_size) # Grade
    feature_2 = np.where(feature_2 > 100, 100, feature_2)
    feature_2 = np.where(feature_2 < 0, 0, feature_2)
    feature_2 = np.round(feature_2, 2)

    feature_3 = np.random.choice([1, 2, 3], size=data_size) # Major
    feature_4 = np.random.choice([1, 2, 3, 4], size=data_size) # Class
    feature_5 = np.random.permutation(feature_2) # GradePerm

    all_features = np.array([feature_0, feature_1, feature_2, feature_3,
                            feature_4, feature_5]).swapaxes(0, 1)

    features_df = pd.DataFrame(all_features, columns=feature_cols)

    targets_list = np.empty((0, target_size))
    for _, row in features_df.iterrows():
        target_row = np.zeros(target_size)
        if row['GradeAvgPrevElec'] > 0:
            course_grades = (0.5 * row['GradeAvgPrevElec'] + 0.5 * row['Grade'] +
                            np.random.uniform(low=-10, high=10, size=target_size))
        else:
            course_grades = row['Grade'] + np.random.normal(loc=10, scale=2, size=target_size)
        course_grades = np.round(course_grades, 2)
        course_grades = np.where(course_grades < 0, 0, course_grades)
        course_grades = np.where(course_grades > 100, 100, course_grades)
        if row['EnrolledElectiveBefore'] == 0:
            selected_course = random.sample(range(target_size-1, target_size-4, -1), 1)
            # np.random.choice([2, 3, 4])
            target_row[selected_course] = course_grades[selected_course]
            targets_list = np.vstack([targets_list, target_row])
            continue
        if row['Class'] == 1:
            selected_course = random.sample([0, 1], 1)
            target_row[selected_course] = course_grades[selected_course]
            targets_list = np.vstack([targets_list, target_row])
            continue
        if (row['Major'] > 2) & (row['Class'] > 2):
            target_candidates = [0] + [x for x in range(target_size-1, target_size-3, -1)]
            selected_course = random.sample(target_candidates, 2)
            target_row[selected_course] = course_grades[selected_course]
            targets_list = np.vstack([targets_list, target_row])
            continue
        if row['Grade'] < 70:
            selected_course = random.sample(range(target_size-1, target_size-4, -1), 2)
            target_row[selected_course] = course_grades[selected_course]
            targets_list = np.vstack([targets_list, target_row])
            continue
        selected_course = random.sample(range(target_size), 2)
        target_row[selected_course] = course_grades[selected_course]
        targets_list = np.vstack([targets_list, target_row])

    targets_df = pd.DataFrame(targets_list, columns=target_cols)
    full_df = pd.concat([features_df, targets_df], axis=1)

    full_df.to_csv(f'{base_folder}/data/class_df_size_{data_size}_targets_{target_size}.csv', index=False)

def apply_constraints_on_scores_data():
    full_df = pd.read_csv(f'{base_folder}/exams.csv')

    flag = full_df['reading score'] <= 50
    full_df.loc[flag, 'writing score'] = 0

    flag = full_df['reading score'] + full_df['writing score'] <= 110
    full_df.loc[flag, 'math score'] = 0

    full_df.to_csv(f'{base_folder}/constrained_exams.csv', index=False)

def generate_forecasting_data(series_length = 1000):
    dataset = np.zeros(series_length)

    #### Constraints
    # Each value in the series should be less than 100
    dataset[0:2] = np.random.randint(0, 100, size=2)

    # If two consecutive values sum up to 70 or more, the next value should be less than 50
    if (dataset[1] + dataset[0] >= 70):
        dataset[2] = np.random.randint(0, 50)

    # If three consecutive values sum up to 120 or more, the next value should be less than 10
    for i in range(3, series_length):
        dataset[i] = np.random.randint(0, 100)
        if (dataset[i-1] + dataset[i-2] >= 70):
            dataset[i] = np.random.randint(0, 50)
        if (dataset[i-1] + dataset[i-2] + dataset[i-3] >= 120):
            dataset[i] = np.random.randint(0, 10)

    # Function to prepare dataset for multi-target prediction
    def prepare_dataset(data, n_features, n_targets):
        X, y = [], []
        endi = len(data) - (n_features + n_targets) + 1
        for i in range(endi):
            features = data[i:i+n_features]
            targets = data[i+n_features:i+n_features+n_targets]
            X.append(features)
            y.append(targets)

        return np.array(X), np.array(y)

    # Define number of features and number of steps ahead for prediction
    n_features = 30
    n_targets = 6

    # Prepare dataset
    X, y = prepare_dataset(dataset, n_features, n_targets)

    # Reshape X to have n_features per sample
    X = X.reshape((X.shape[0], n_features))

    # Print the input-output pairs
    for i in range(len(X)):
        print(X[i], y[i])

    feature_cols = [f'Feature {id+1}' for id in range(X.shape[1])]
    target_cols = [f'Target {id + 1}' for id in range(y.shape[1])]

    feature_df = pd.DataFrame(X, columns=feature_cols)
    target_df = pd.DataFrame(y, columns=target_cols)

    full_df = pd.concat([feature_df, target_df], axis=1)
    full_df.to_csv(f'{base_folder}/data/forecasting.csv', index=False)
