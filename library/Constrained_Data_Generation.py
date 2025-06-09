# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:02:30 2025

@author: ht
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
from scipy.linalg import null_space
import itertools
import os

base_folder = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generate_constrained_dataset_with_nonlinearity(
    n_samples=500,
    n_features=6,
    n_targets=5,
    noise_level=0.01,
    random_state=None
):
    rng = check_random_state(random_state)

    # Step 1: Generate X in [0, 1]
    num_unique = 10
    values = np.array([i/num_unique for i in range(num_unique+1)]) 
    X = rng.choice(values, size=(n_samples, n_features))

    # Step 2: Define linear constraints A y = b
    def generate_list(n):
        result = []
        for i in range(n_targets//2,n,2):
            row = [0] * n
            row[i-1] = 1
            row[i] = -1
            result.append(row)
        return result
    temp = generate_list(n_targets)
    temp2 = [[1 if i <n_targets//2 else 0 for i in range(n_targets)]]
    
    A = np.array(temp2+temp)
    
    tempb = [1] + [(i+1)/10 for i in range(len(temp))]
    b = np.array(tempb)

    
    def generate_random_feature_coefficients(n_targets, n_features, seed=None):
        """
        For each target, generate random nonlinear transformation types, coefficients, 
        and strength multipliers for each feature.
        """
    
        if seed is not None:
            np.random.seed(seed)
    
        function_types = ["sin", "cos", "exp", "tanh", "linear", "poly"]
    
        feature_coeffs = []
        for _ in range(n_targets):
            target_coeffs = []
            for _ in range(n_features):
                func_type = np.random.choice(function_types)
    
                if func_type in ["sin", "cos"]:
                    coeff = np.random.uniform(0.5, 5.0) * np.pi
                elif func_type == "exp":
                    coeff = np.random.uniform(0.5, 5.0)
                elif func_type == "tanh":
                    coeff = np.random.uniform(1.0, 10.0)
                elif func_type == "linear":
                    coeff = np.random.uniform(0.5, 2.0)
                elif func_type == "poly":
                    coeff = (np.random.choice([2, 3]), np.random.choice([2, 3]))
                else:
                    raise ValueError(f"Unknown function type {func_type}")
    
                # Add a strength multiplier: 0.1 (weak) to 1.5 (strong)
                strength = np.random.uniform(0.1, 1.5)
    
                target_coeffs.append((func_type, coeff, strength))
            feature_coeffs.append(target_coeffs)
    
        return feature_coeffs

    def nonlinear_features(x, feature_coeffs):
        """
        Given input x (size n_features), and feature_coeffs,
        compute nonlinear transformed vector (size n_targets).
        """
    
        n_targets = len(feature_coeffs)
        transformed_targets = []
    
        for target_idx in range(n_targets):
            coeffs_for_target = feature_coeffs[target_idx]
    
            target_value = 0.0
            for feature_idx, (func_type, coeff, strength) in enumerate(coeffs_for_target):
                xi = x[feature_idx]
    
                if func_type == "sin":
                    contrib = np.sin(coeff * xi)
                elif func_type == "cos":
                    contrib = np.cos(coeff * xi)
                elif func_type == "exp":
                    contrib = np.exp(-coeff * xi)
                elif func_type == "tanh":
                    contrib = np.tanh(coeff * (xi - 0.5))
                elif func_type == "linear":
                    contrib = coeff * xi
                elif func_type == "poly":
                    p1, p2 = coeff
                    contrib = xi ** p1 + xi ** p2
                else:
                    raise ValueError(f"Unknown function type {func_type}")
    
                # Apply the strength multiplier
                target_value += strength * contrib
    
            transformed_targets.append(target_value)
    
        return np.array(transformed_targets)


    feature_coeffs = generate_random_feature_coefficients(n_targets,n_features, seed=random_state)
    Y_raw = np.array([nonlinear_features(x, feature_coeffs) for x in X])  # shape: (n_samples, n_targets)

    # Step 4: Project Z onto the affine constraint manifold Ay = b
    def project_to_constraints(z):
        A_pinv = A @ A.T
        lambda_vec = np.linalg.solve(A_pinv, A @ z - b)
        y = z - A.T @ lambda_vec
        return y

    Y_clean = np.array([project_to_constraints(y) for y in Y_raw])
    Y_noisy = Y_clean + rng.normal(0, noise_level, Y_clean.shape)

    return X, Y_clean, Y_noisy, A, b

def create_dataframe(X, Y, prefix='y'):
    feature_cols = [f'X{i+1}' for i in range(X.shape[1])]
    target_cols = [f'{prefix}{i+1}' for i in range(Y.shape[1])]
    return pd.DataFrame(np.hstack([X, Y]), columns=feature_cols + target_cols)

instancesize_list = [500,1000,2000]
targetsize_list = [5,9]
seed_list = [i for i in range(3)]
noisestd = 0.1
params = list(itertools.product(instancesize_list, targetsize_list,seed_list))

for instancesize, targetsize, s in params:
    X, Y_clean, Y_noisy, A, b = generate_constrained_dataset_with_nonlinearity(
        n_samples=instancesize,
        n_features=6,
        n_targets=targetsize,
        noise_level=noisestd,
        random_state=s
    )
        
    df_noisefree = create_dataframe(X, Y_clean, prefix='y')
    df = create_dataframe(X, Y_noisy, prefix='y')
    df.to_csv(f'{parent_dir}/data/synthetic_manifold_df_size_{instancesize}_targets_{targetsize}_seed_{s}.csv', index=False)

