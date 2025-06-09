# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:22:24 2025

@author: ht
"""

import numpy as np
import pandas as pd
import itertools
import os

base_folder = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generate_features(n_samples, n_features, n_unique_values=10, random_state=None):
    rng = np.random.RandomState(random_state)
    values = np.linspace(0, 1, n_unique_values)
    X = rng.choice(values, size=(n_samples, n_features))
    return X

def generate_sparsity_mask(n_samples, n_targets, sparsity_level, random_state=None):
    rng = np.random.RandomState(random_state)
    return rng.binomial(1, 1 - sparsity_level, size=(n_samples, n_targets))

def generate_deterministic_sparsity_mask(n_samples, n_targets, sparsity_level, random_state=None):
    """
    Ensure that each row has the same number of non-zero entries (deterministic sparsity).
    """
    rng = np.random.RandomState(random_state)
    k = int(np.ceil((1 - sparsity_level) * n_targets))  # number of non-zero targets per row
    mask = np.zeros((n_samples, n_targets), dtype=int)

    for i in range(n_samples):
        nonzero_indices = rng.choice(n_targets, size=k, replace=False)
        mask[i, nonzero_indices] = 1

    return mask


def generate_nonlinear_integer_targets(X, n_targets, upper_bound, random_state=None):
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    Y = np.zeros((n_samples, n_targets))

    for j in range(n_targets):
        feature_indices = rng.choice(n_features, size=rng.randint(2, min(4, n_features)+1), replace=False)
        interaction = np.random.uniform(1.0, 10.0) * X[:, feature_indices].prod(axis=1)
        trig = np.sin(np.random.uniform(0.5, 5.0) * np.pi * X[:, feature_indices[0]]) + np.cos(np.random.uniform(0.5, 5.0) * np.pi * X[:, feature_indices[-1]])
        poly = (X[:, feature_indices[0]] ** np.random.uniform(2.0, 5.0)) * (np.random.uniform(-1.0, 1.0) + X[:, feature_indices[1]])

        combined = interaction + trig + poly
        noise = rng.normal(0, 0.2, size=n_samples)
        y_j = combined + noise

        # Scale to [0, upper_bound] and convert to integer
        y_scaled = np.interp(y_j, (y_j.min(), y_j.max()), (0, upper_bound))
        Y[:, j] = np.round(np.clip(y_scaled, 0, upper_bound)).astype(int)

    return Y


def apply_sparsity(Y, mask):
    return Y * mask

def enforce_target_sum_constraint(Y, target_sum=20):
    """
    Scales each row of Y to ensure the sum of nonzero entries equals `target_sum`.
    Keeps zeros as-is and rescales nonzero elements proportionally.
    """
    Y_scaled = Y.copy()
    for i in range(Y.shape[0]):
        row = Y[i]
        nonzero_indices = row > 0
        nonzero_values = row[nonzero_indices]

        if nonzero_values.sum() > 0:
            # Rescale nonzero values proportionally
            scaled_values = nonzero_values * (target_sum / nonzero_values.sum())
            rounded_values = np.floor(scaled_values).astype(int)

            # Adjust one value to correct rounding error
            diff = target_sum - np.sum(rounded_values)
            if diff != 0:
                adjust_idx = np.argmax(rounded_values)  # or random
                rounded_values[adjust_idx] += diff

            Y_scaled[i, nonzero_indices] = np.clip(rounded_values, 0, None)

    return Y_scaled


def introduce_infeasibility(Y, upper_bound, infeasibility_level, random_state=None):
    rng = np.random.RandomState(random_state)
    Y_infeasible = Y.copy()
    n_samples, n_targets = Y.shape
    n_violations = int(infeasibility_level * n_samples)

    indices = rng.choice(n_samples, n_violations, replace=False)
    for i in indices:
        target_idx = rng.choice(n_targets//2)
        Y_infeasible[i, target_idx] = rng.randint(upper_bound + 1, upper_bound * 2)

    return Y_infeasible


def generate_dataset(
    n_samples,
    n_features,
    n_targets,
    upper_bound,
    sparsity_level,
    infeasibility_level,
    target_sum=None,
    n_unique_values=10,
    random_state=None
):
    X = generate_features(n_samples, n_features, n_unique_values, random_state)
    sparsity_mask = generate_deterministic_sparsity_mask(n_samples, n_targets, sparsity_level, random_state)
    Y_nonlinear = generate_nonlinear_integer_targets(X, n_targets, upper_bound, random_state)
    Y_sparse = apply_sparsity(Y_nonlinear, sparsity_mask)
    
    if target_sum is not None:
        Y_sparse = enforce_target_sum_constraint(Y_sparse, target_sum=target_sum)

    Y_final = introduce_infeasibility(Y_sparse, upper_bound, infeasibility_level, random_state)
    return X, Y_final

instancesize_list = [500]
seed_list = [i for i in range(5)]
params = list(itertools.product(instancesize_list,seed_list))
for instancesize, s in params:
    X, Y = generate_dataset(
        n_samples=instancesize,
        n_features=6,
        n_targets=13,
        upper_bound=10,
        sparsity_level=0.7,
        infeasibility_level=0.,
        target_sum=15,
        random_state=s
    )
    
    
    # Convert to DataFrame
    df = pd.DataFrame(np.hstack([X, Y]), columns=[f"X{i+1}" for i in range(X.shape[1])] + [f"y{i+1}" for i in range(Y.shape[1])])
    df.to_csv(f'{parent_dir}/data/hts_df_size_{instancesize}_seed_{s}_v2.csv', index=False)


