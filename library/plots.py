import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

def generate_performance_barplot(perf_df, dataset = 'class', report_metric = 'mse'):
    report_cols = [report_metric, 'training_duration', 'nof_infeasibilities']
    perf_df_dataset = perf_df[perf_df['data'] == dataset]

    perf_df_plot = perf_df_dataset.groupby(['prediction_method', 'prediction_method_leaf'])[report_cols].mean().reset_index()
    perf_df_plot.loc[perf_df_plot['prediction_method'].isin(['mean', 'sklearn']), 'prediction_method'] = 'average'
    perf_df_plot.loc[perf_df_plot['prediction_method'] == 'lagrangian', 'prediction_method'] = 'relaxation'
    perf_df_plot.loc[perf_df_plot['prediction_method_leaf'] == 'lagrangian', 'prediction_method_leaf'] = 'relaxation'
    perf_df_plot.loc[perf_df_plot['prediction_method_leaf'] == 'sklearn', 'prediction_method_leaf'] = 'average'
    perf_df_plot = perf_df_plot.set_index(['prediction_method', 'prediction_method_leaf'])
    # perf_df_plot['run'] = [f'S({x[0][0].upper()}) - L({x[1][0].upper()}) \n 
    #                        ({round(row.nof_infeasibilities, 1)})' for x, row in perf_df_plot.iterrows()]
    perf_df_plot['run'] = [f'S({x[0][0].upper()}):L({x[1][0].upper()})' for x, row in perf_df_plot.iterrows()]

    perf_df_plot['mse'] = perf_df_plot['mse'] / perf_df_plot['mse'].max()
    perf_df_plot['training_duration'] = perf_df_plot['training_duration'] / perf_df_plot['training_duration'].max()

    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1 = perf_df_plot[report_metric].plot.bar()
    
    # ax.set_title(f"Dataset: {dataset.upper()}", weight='bold', fontsize=20) # \n Metric: {report_metric.upper()}")
    ax1.set_ylim((0, perf_df_plot[report_metric].max() * 1.1))
    ax1.set_xlabel("Class Dataset", weight='bold', labelpad=15, fontsize=22)
    ax1.set_ylabel("(Scaled) Mean Squared Error", weight='bold', labelpad=15, fontsize=22)
    ax1.set_xticklabels(perf_df_plot.run, color='black', fontsize=20)
    ax1.set_yticklabels([x/10 for x in range(0, 11, 2)], fontsize=20)
    ax1.tick_params(axis='x', labelrotation=0)

    # Axis formatting.
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color('#DDDDDD')
    ax1.tick_params(bottom=False, left=False, length=0)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='#EEEEEE')
    ax1.xaxis.grid(False)
    ax1.yaxis.label.set_color('#1f77b4')

    rects = ax1.patches
    perf_df_plot_nof_inf = perf_df_dataset.groupby(['prediction_method', 'prediction_method_leaf'])[
        'nof_infeasibilities'].mean()

    labels = [f'{round(perf_df_plot.mse.values[i], 3):.3f}' + '\n' + f'({round(
        perf_df_plot_nof_inf.values[i], 1):.1f})' for i in range(len(perf_df_plot.mse.values))]
    # labels = [f'{round(perf_df_plot.mse.values[i], 3):.3f}' for i in range(len(perf_df_plot.mse.values))]

    bar_color = ax1.patches[0].get_facecolor()
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label, 
                ha="center", va="bottom", color='blue', fontsize=20)
        
    ax2 = ax1.twinx()
    # ax2.plot(perf_df_plot['training_duration'].values, 'r-', linewidth=4)
    sns.scatterplot(x=perf_df_plot['run'].values, y=perf_df_plot['training_duration'].values, 
                    ax=ax1, s=200, markers='+', color='darkred', legend=False)

    # Make the y-axis label and tick labels match the line color.
    ax2.set_ylabel('(Scaled) Training Time', color='black', weight='bold', labelpad=15, fontsize=22)

    # Axis formatting.
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_color('#DDDDDD')
    ax2.tick_params(bottom=False, left=False, length=0)
    ax2.yaxis.set_tick_params(labelsize=20)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, color='#EEEEEE')
    ax2.xaxis.grid(False)
    ax2.yaxis.label.set_color('darkred')
    ax2.set_ylim((0, perf_df_plot['training_duration'].max() * 1.1))

    fig.tight_layout()

    plt.show()

def generate_performance_boxplots(perf_df, report_metric = 'mse', include_texts = False):
    # perf_df_plot = perf_df.groupby(['prediction_method', 'prediction_method_leaf'])[report_metric].mean().reset_index()
    perf_df_plot = perf_df.copy()
    perf_df_plot.loc[perf_df_plot['prediction_method'].isin(['mean', 'sklearn']), 'prediction_method'] = 'average'
    perf_df_plot.loc[perf_df_plot['prediction_method'] == 'lagrangian', 'prediction_method'] = 'relaxation'
    perf_df_plot.loc[perf_df_plot['prediction_method_leaf'] == 'lagrangian', 'prediction_method_leaf'] = 'relaxation'
    perf_df_plot.loc[perf_df_plot['prediction_method_leaf'] == 'sklearn', 'prediction_method_leaf'] = 'average'
    perf_df_plot = perf_df_plot.set_index(['prediction_method', 'prediction_method_leaf'])
    perf_df_plot['run'] = [f'S({x[0][0].upper()}):L({x[1][0].upper()})' for x in perf_df_plot.index]
    perf_df_plot = perf_df_plot[perf_df_plot['run'] != 'S(A):L(A)']

    perf_df_plot[report_metric] = perf_df_plot[report_metric] / perf_df_plot[report_metric].max()
    # perf_df_plot['training_duration'] = perf_df_plot['training_duration'] / perf_df_plot['training_duration'].max()

    sns.set(font_scale=2)  # crazy big
    ax1 = sns.boxplot(data=perf_df_plot, x='run', y=report_metric)

    if include_texts:
        nof_inf_mean = perf_df_plot.groupby(['run'])['nof_infeasibilities'].mean()
        mse_mean = perf_df_plot.groupby(['run'])['mse'].mean()
        vertical_offset = mse_mean * 1.05 # offset from median for display

        for xtick in ax1.get_yticks():
            ax1.text(xtick, mse_mean[xtick] + 0.05, f'N. of. Inf: {nof_inf_mean[xtick]}', 
                color='black', size=10, weight='semibold')
    
    # ax.set_title(f"Dataset: {dataset.upper()}", weight='bold', fontsize=20) # \n Metric: {report_metric.upper()}")
    ax1.set_ylabel("(Scaled) Training Time", weight='bold', labelpad=15, fontsize=22)
    ax1.set_xlabel("", weight='bold', labelpad=15, fontsize=22)
    # ax1.set_xticklabels([0] + [x/10 for x in range(0, 11, 2)], color='black', fontsize=16)
    # ax1.set_yticklabels(perf_df_plot['run'].values, fontsize=16)
    ax1.tick_params(axis='x', labelrotation=0)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color('#DDDDDD')
    ax1.tick_params(bottom=False, left=False, length=0)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='#EEEEEE')
    ax1.xaxis.grid(False)
    ax1.set_facecolor('white')
    
    # ax1.yaxis.tick_right()

    plt.show()

def generate_leaf_predictions_barplot(y_pred_df, leaf_id):
    y_pred_df_plot = y_pred_df[y_pred_df['leaf_id'] == leaf_id].drop(columns=['leaf_id'])
    data = {}
    for id in range(len(y_pred_df_plot.columns)):
        data[f'{id+1}'] = y_pred_df_plot[id].values[0]

    courses = list(data.keys())
    predictions = list(data.values())

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.bar(courses, predictions, color=colors)
    plt.xlabel('Courses', fontsize=18)
    plt.ylabel('Predictions', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def create_depth_vs_infeasibilities_plot(features_df, targets_df, nplots = 25):
    rstate = 23
    x_train, x_test, y_train, y_test = train_test_split(features_df, targets_df, test_size = 0.3, random_state=rstate)

    results = np.zeros((nplots-1, 2))
    for md in range(1, nplots):
        regressor = DecisionTreeRegressor(max_depth=md, min_samples_leaf=1, random_state=rstate)
        regressor.fit(x_train, y_train)
        y_pred = regressor.predict(x_test)
        acc = mean_squared_error(y_test, y_pred)
        results[md-1, 0] = acc
        print('Depth: ', md)
        print('DT Accuracy: ', acc)
        cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
        infeasible = np.sum(cumsums >= 3)
        results[md-1, 1] = infeasible
        print('Number of infeasible predictions for DT: ', infeasible)
        print()

    y0 = results[:, 0] / np.max(results[:, 0])
    y1 = 100*(results[:, 1] / len(y_test))
    fig, (ax2, ax1) = plt.subplots(1 ,2, figsize=(16, 5))
    # fig.suptitle('Mean Squared Error vs. Feasibility for Different Tree Depths')
    x = np.arange(1, nplots)
    y = y0
    ax1.scatter(x, y)
    ax1.set_xlabel('Depth', weight='bold', fontsize=12)
    ax1.set_ylabel('(Scaled) Mean Squared Error', weight='bold', fontsize=12)
    y = y1
    ax2.bar(x, y)
    ax2.set_xlabel('Depth', weight='bold', fontsize=12)
    ax2.set_ylabel('Percentage of Infeasible Predictions \n (over 1200 test samples)', weight='bold', fontsize=12)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    min_mse_pt = np.argmin(results[:, 0])+1
    ax1.scatter(min_mse_pt, np.min(y0), color='darkred')
    ax2.bar(min_mse_pt, y1[min_mse_pt]+1.0, color='darkred')

    # Axis formatting.
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_color('#DDDDDD')
    ax1.tick_params(bottom=False, left=False)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color='#EEEEEE')
    ax1.xaxis.grid(False)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_color('#DDDDDD')
    ax2.tick_params(bottom=False, left=False)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(True, color='#EEEEEE')
    ax2.xaxis.grid(False)