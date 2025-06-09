# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:06:20 2025

@author: ht
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})


base_folder = os.getcwd()
df = pd.read_csv(f'{base_folder}/data/results/Synthetic_Results_All.csv')

dfhts = pd.read_csv(f'{base_folder}/data/results/HTS_Results_All.csv')

custom_method_rf_names = {
    'sklearn|no': 'SKLEARN',
    'mean|no': 'EP-OCRT',
    'optimal|no': 'E-OCRT',
    'sing-depthMIP|no': 'M-OCRT',
    'mean|yes': 'EP-RF',
    'optimal|yes': 'E-RF',
    'sing-depthMIP|yes': 'M-RF',
    # Add more combinations if needed
}

def boxplot():
    # Step 1: Extract size, targets, seed from the 'data' column
    df[['size', 'targets', 'seed']] = df['data'].str.extract(
        r'size_(\d+)_targets_(\d+)_seed_(\d+)', expand=True
    ).astype(int)
    
    df['method_RF'] = df['prediction_method'] + " | RF: " + df['RF']
    
    combinations = df[['size', 'targets']].drop_duplicates()
    
    
    for _, row in combinations.iterrows():
        s, t= row['size'], row['targets']
        subset = df[(df['size'] == s) & (df['targets'] == t)]
    
        if subset.empty:
            continue
    
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=subset, x='method_RF', y='mse')
        plt.xticks(rotation=45)
        plt.title(f'MSE by Method + RF (size={s}, targets={t})')
        plt.xlabel('Prediction Method + RF')
        plt.ylabel('MSE')
        plt.tight_layout()
        plt.show()

# boxplot()

def boxplotHTS():
    # Step 1: Extract size, targets, seed from the 'data' column
    dfhts[['size', 'seed']] = dfhts['data'].str.extract(
        r'size_(\d+)_seed_(\d+)_', expand=True
    ).astype(int)
    
    
    # combinations = dfhts[['size']].drop_duplicates()
    combinations = dfhts[['size', 'depth']].drop_duplicates()
    
    
    for _, row in combinations.iterrows():
        # s = row['size']
        # subset = dfhts[(dfhts['size'] == s)]
    
        # if subset.empty:
        #     continue
    
        # plt.figure(figsize=(10, 6))
        # sns.boxplot(data=subset, x='prediction_method', y='mse')
        s, t= row['size'], row['depth']
        subset = dfhts[(dfhts['size'] == s) & (dfhts['depth'] == t)]
    
        if subset.empty:
            continue
    
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=subset, x='prediction_method', y='mse')
        plt.xticks(rotation=45)
        plt.title(f'MSE by Method + RF (size={s}, depth={t})')
        plt.xlabel('Prediction Method')
        plt.ylabel('MSE')
        plt.tight_layout()
        plt.show()

# boxplotHTS()

def boxplot2():
    # Step 1: Extract fields from 'data'
    df[['size', 'targets', 'seed']] = df['data'].str.extract(
        r'size_(\d+)_targets_(\d+)_seed_(\d+)', expand=True
    ).astype(int)
    
    
    grouped = df.groupby(['size', 'targets', 'seed', 'prediction_method', 'RF'], as_index=False)['mse'].mean()
    grouped['method_RF'] = grouped['prediction_method'] + " | RF: " + grouped['RF']
    
    # Step 5: Loop over all (size, targets) pairs and plot boxplots
    unique_pairs = grouped[['size', 'targets']].drop_duplicates()
    
    for _, row in unique_pairs.iterrows():
        s, t = row['size'], row['targets']
        subset = grouped[(grouped['size'] == s) & (grouped['targets'] == t)]
        
        if subset.empty:
            continue
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=subset, x='method_RF', y='mse')
        plt.title(f'Boxplot of Averaged MSEs by Method + RF\n(size={s}, targets={t})')
        plt.xlabel('Prediction Method + RF')
        plt.ylabel('Average MSE (over seeds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# boxplot2()

def boxplot3():
    # Step 1: Extract fields from 'data'
    df[['size', 'targets', 'seed']] = df['data'].str.extract(
        r'size_(\d+)_targets_(\d+)_seed_(\d+)', expand=True
    ).astype(int)
    
    # Step 2: Ensure MSE is numeric
    # df['mse'] = pd.to_numeric(df['mse'], errors='coerce')
    
    # Step 4: Compute average MSE per (size, targets, seed, method, RF)
    grouped = df.groupby(['size', 'targets', 'seed', 'prediction_method', 'RF'], as_index=False)['mse gap'].mean()
    grouped['method_RF'] = grouped['prediction_method'] + " | RF: " + grouped['RF']
    
    unique_sizes = grouped['size'].unique()
    
    for s in unique_sizes:
        subset = grouped[grouped['size'] == s]
        
        if subset.empty:
            continue
    
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=subset, x='method_RF', y='mse gap')
        plt.title(f'Boxplot of Averaged MSEs by Method + RF\n(Fixed size = {s}, averaged over targets + seeds)')
        plt.xlabel('Prediction Method + RF')
        plt.ylabel('Average MSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    unique_targets = grouped['targets'].unique()
    
    for t in unique_targets:
        subset = grouped[grouped['targets'] == t]
        
        if subset.empty:
            continue
    
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=subset, x='method_RF', y='mse gap')
        plt.title(f'Boxplot of Averaged MSEs by Method + RF\n(Fixed targets = {t}, averaged over sizes + seeds)')
        plt.xlabel('Prediction Method + RF')
        plt.ylabel('Average MSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
# boxplot3()  
def plot_avg_training_duration(df,custom_labels):
    # Ensure training_duration is numeric
    df['training_duration'] = pd.to_numeric(df['training_duration'], errors='coerce')
    df = df.dropna(subset=['training_duration'])
    
    # Create raw method_RF key
    df['method_RF_key'] = df['prediction_method'] + '|' + df['RF']

    # Apply custom labels
    df['method_RF'] = df['method_RF_key'].map(custom_labels).fillna(df['method_RF_key'])
    
    df = df[~df['method_RF_key'].isin(['sing-depthMIP|yes'])]

    # Compute average training duration
    avg_duration = (
        df.groupby('method_RF', as_index=False)['training_duration']
        .mean()
        .sort_values('training_duration', ascending=True)
    )

    # Plot
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=avg_duration,
        x='method_RF',
        y='training_duration',
        sort=False,  # Respect current order
        color='blue',
        linewidth=2,
        zorder=1
    )
    sns.scatterplot(
        data=avg_duration,
        x='method_RF',
        y='training_duration',
        s=150,
        color='blue'
    )
    plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Method")
    plt.xlabel("")
    plt.ylabel("Average Training Duration (seconds)")
    plt.legend(title=None)
    # plt.title("Average Training Duration per Method")
    # plt.grid(False, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("avg_train_time.pdf")
    plt.show()
plot_avg_training_duration(df,custom_method_rf_names)

def plot_training_duration_by_size(df, custom_labels=None):
    # Ensure numeric values
    df['training_duration'] = pd.to_numeric(df['training_duration'], errors='coerce')
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df = df.dropna(subset=['training_duration', 'Size'])

    # Create method-RF key
    df['method_RF_key'] = df['prediction_method'] + '|' + df['RF']

    df = df[~df['method_RF_key'].isin(['sing-depthMIP|yes'])]

    # Apply custom labels if provided
    if custom_labels:
        df['method_RF'] = df['method_RF_key'].map(custom_labels).fillna(df['method_RF_key'])
    else:
        df['method_RF'] = df['method_RF_key']

    # Compute average training duration for each method per size
    avg_df = (
        df.groupby(['Size', 'method_RF'], as_index=False)['training_duration']
        .mean()
    )
    
    avg_df['log_training_duration'] = np.log1p(avg_df['training_duration'])

    # Plot
    plt.figure(figsize=(10, 7))
    lineplot  = sns.lineplot(
        data=avg_df,
        x='Size',
        y='log_training_duration',
        hue='method_RF',
        style='method_RF',
        markers=True,
        dashes=False,
        linewidth=1,
        palette='tab10'
    )
    plt.setp(lineplot.lines, markersize=15)
    plt.xticks(sorted(df['Size'].unique()))
    plt.xlabel("n")
    plt.ylabel("Average Training Duration (log scale)")
    plt.legend(title=None,loc='upper left')
    # plt.title("Average Training Duration per Method by Size")
    # plt.grid(False, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig("avg_train_time_size.pdf")

    plt.show()
plot_training_duration_by_size(df,custom_method_rf_names)