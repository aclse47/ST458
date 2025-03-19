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

df_with_features <- add_features(df, dV_kalman = 10, dW_kalman = 0.0001)
df_with_features <- as.data.frame(df_with_features)

response_vars <- colnames(df_with_features %>% select(contains("fwd")))
covariate_vars <- setdiff(colnames(df_with_features), c(response_vars, 'date', 'symbol'))

# Deliberately leaking in data to see how it performs.
# Note: We get 160% rate of return!
# covariate_vars <- c(covariate_vars, 'simple_returns_fwd_day_5')

categorical_vars <- c()


df_with_features_train <- df_with_features[df_with_features$date < as.Date('2013-01-01'), ]
df_with_features_test <- df_with_features[df_with_features$date >= as.Date('2013-01-01'), ]

# Hyper-parameter combination grid
param_df <- expand.grid(
  train_length=c(252, 252*2, 126),
  valid_length=c(21, 63),
  lookahead=c(5,21),
  num_leaves=c(5,10,50),
  min_data_in_leaf=c(250,1000),
  learning_rate=c(0.01,0.03,0.1),
  feature_fraction=c(0.3,0.6,0.95),
  bagging_fraction=c(0.3,0.6,0.95),
  num_iterations=c(30,100)
)


training_log <- hyperparameter_grid_training_lgbm(df_with_features_train, param_df, 100, covariate_vars, categorical_vars)
training_log <- sort_data_frame(training_log, 'ic', decreasing=T)
head(training_log)

################################################################################
# Evaluation of LGBM with some plots
################################################################################

lgbm_features_effects_plot(df_with_features_train, covariate_vars, training_log[1, ])
lgbm_hyperparameters_marginal_effect_plot(training_log)
dev.off()


################################################################################
# Back-testing a trading algorithm on the validation set.
################################################################################

hyperparameters <- training_log[1, ]
y_preds <- lgbm_get_validation_set_predictions(df_with_features, df_with_features_test, covariate_vars, categorical_vars, hyperparameters)

# This implements basic strategy of buy top 5 highest returns and short bottom 5 lowest returns
# combined_position <- lgbm_get_positions_based_on_predictions(df_with_features, df_with_features_test, y_preds, hyperparameters)

# This implements Kelly Criterion
combined_position_kelly <- lgbm_get_positions_based_on_kelly(df_with_features, df_with_features_test, y_preds, hyperparameters)


combined_position_min_var <- lgbm_get_positions_based_on_wmv(df_with_features, df_with_features_test, y_preds, hyperparameters)

wealth_and_pnl <- get_pnl_based_on_position(df_with_features, df_with_features_test, combined_position_min_var)
performance_evaluation_of_wealth(wealth_and_pnl$wealth, wealth_and_pnl$daily_pnl, 0.03)


