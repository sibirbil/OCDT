import pandas as pd

report_metric = 'mse'
dataset = 'class'

report_cols = [report_metric, 'training_duration']

if dataset == 'class':
    perf_df_dataset = pd.read_csv('data/perf_df_class_df_all_w_lagrangian.csv')
    perf_df_plot = perf_df_dataset.groupby(['prediction_method', 'prediction_method_leaf', 
                                            'num_targets', 'num_rows', 'fold'])[report_cols].mean().reset_index()
else:
    perf_df = pd.read_csv('data/latest_results/perf_df_all.csv')
    perf_df_dataset = perf_df[perf_df['data'] == dataset]
    perf_df_plot = perf_df_dataset.groupby(['prediction_method', 'prediction_method_leaf', 
                                            'fold'])[report_cols].mean().reset_index()

perf_df_plot.loc[perf_df_plot['prediction_method'].isin(['mean', 'sklearn']), 'prediction_method'] = 'average'
perf_df_plot.loc[perf_df_plot['prediction_method'] == 'lagrangian', 'prediction_method'] = 'relaxation'
perf_df_plot.loc[perf_df_plot['prediction_method_leaf'] == 'lagrangian', 'prediction_method_leaf'] = 'relaxation'
perf_df_plot.loc[perf_df_plot['prediction_method_leaf'] == 'sklearn', 'prediction_method_leaf'] = 'average'
perf_df_plot = perf_df_plot.set_index(['prediction_method', 'prediction_method_leaf'])
perf_df_plot['run'] = [f'S({x[0][0].upper()}):L({x[1][0].upper()})' for x in perf_df_plot.index]

if dataset == 'class':
    pd.pivot_table(perf_df_plot, values='mse', columns='run', 
                   index=['fold', 'num_targets', 'num_rows']).to_csv(
                       'data/latest_results/perf_df_class_mean.csv')
else:
    pd.pivot_table(perf_df_plot, values='mse', columns='run', 
                   index='fold').to_csv(f'data/latest_results/perf_df_{dataset}_mean.csv')

