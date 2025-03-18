library(putils)
library(tidyverse)
library(xts)
library(dplyr)
library(R.utils)
library(lightgbm)
library(reshape2)
library(quantmod)
library(zoo)
library(ggplot2)
library(TTR)


source('training_functions.R')
source('feature_engineering_functions.R')
source('model_evaluation_functions.R')

df <- read.csv('df_train.csv')                    # Read in csv
df$date <- as.Date(df$date, format = "%Y-%m-%d")  # Make the date column date instead of char
df <- df %>% arrange(symbol, date)                # Order according to symbol then date like in case study lecture

tickers <- unique(df$symbol)

df_with_features <- add_features(df)
df_with_features <- as.data.frame(df_with_features)

response_vars <- colnames(df_with_features %>% select(contains("fwd")))
covariate_vars <- setdiff(colnames(df_with_features), c(response_vars, 'date', 'symbol'))
categorical_vars <- c()


df_with_features_train <- df_with_features[df_with_features$date < as.Date('2013-01-01'), ]
df_with_features_test <- df_with_features[df_with_features$date >= as.Date('2013-01-01'), ]

# Hyper-parameter combination grid
param_df <- expand.grid(
  train_length=c(252, 252*2),
  valid_length=c(21,63),
  lookahead=c(1,5,21),
  num_leaves=c(5,10,50),
  min_data_in_leaf=c(250,1000),
  learning_rate=c(0.01,0.03,0.1),
  feature_fraction=c(0.3,0.6,0.95),
  bagging_fraction=c(0.3,0.6,0.95),
  num_iterations=c(30,100)
)


training_log <- hyperparameter_grid_training_lgbm(df_with_features_train, param_df, 100, covariate_vars, categorical_vars)

head(sort_data_frame(training_log, 'ic', decreasing=T))


################################################################################
# Evaluation
################################################################################

lgbm_features_effects_plot(df_with_features_train, covariate_vars, training_log[1, ])
lgbm_hyperparameters_marginal_effect_plot(training_log)
dev.off()



all_dates <- sort(unique(df$date))

