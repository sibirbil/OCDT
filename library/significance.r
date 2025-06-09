require(tsutils)

base_folder <- getwd()
data_folder <- paste0(base_folder, '/data/results/')

nemenyi_input_ocdt <- read.csv(paste0(data_folder, 'perf_df_cars_mean.csv'), check.names=FALSE)
nemenyi_input_ocdt <- read.csv(paste0(data_folder, 'perf_df_scores_mean.csv'), check.names=FALSE)
nemenyi(nemenyi_input_ocdt[,-1],conf.level=0.95,plottype="mcb")

nemenyi_input_ocdt <- read.csv(paste0(data_folder, 'perf_df_class_mean.csv'), check.names=FALSE)
nemenyi(nemenyi_input_ocdt[,-1:-3],conf.level=0.95,plottype="mcb")

