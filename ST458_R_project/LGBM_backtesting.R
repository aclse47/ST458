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
library(urca)
library(dplyr)


source("training_functions.R")
source("feature_engineering_functions.R")
source("model_evaluation_functions.R")
source("evaluation_metrics.R")

################################################################################
# Data Preprocessing 
################################################################################

df <- read.csv("df_train.csv") # Read in csv
df$date <- as.Date(df$date, format = "%Y-%m-%d") # Make the date column date instead of char
df <- df %>% arrange(symbol, date) # Order according to symbol then date like in case study lecture

df_with_features <- add_features(df, dV_kalman = 10, dW_kalman = 0.0001)
df_with_features <- as.data.frame(df_with_features)
head(df_with_features)

tickers <- unique(df$symbol)

################################################################################
# LGBM
################################################################################

response_vars <- colnames(df_with_features %>% dplyr::select(matches("fwd")))
covariate_vars <- setdiff(colnames(df_with_features), c(response_vars, "date", "symbol")) # 'residual', 'residual_lag1', 'residual_lag2'
# covariate_vars <- c("open", "high", "low", "close", "volume", "month_of_year", "day_of_week", "simple_returns", "log_returns", "gap", "abs_gap", "VWAP", "dollar_volume", "volume_shock", "range", "kalman_filtered_close_dV_10_dW_1e-04") # THIS LINE IS USED TO TEST WITH NO FEATURES
# Deliberately leaking in data to see how it performs.
# Note: We get 160% rate of return!

categorical_vars <- c('month_of_year', 'day_of_week', 'is_month_start', 'is_month_end')
# categorical_vars <- c()

df_with_features_train <- df_with_features[df_with_features$date < as.Date("2013-01-01"), ]
df_with_features_test <- df_with_features[df_with_features$date >= as.Date("2013-01-01"), ]

# Hyper-parameter combination grid

param_df <- expand.grid(
  train_length = c(252, 252 * 2),
  valid_length = c(21, 63),
  lookahead = c(5),
  num_leaves = c(50, 75, 100),
  min_data_in_leaf = c(250, 1000),
  learning_rate = c(0.1, 0.15, 0.2, 0.5),
  feature_fraction = c(0.6, 0.95, 1.00),
  bagging_fraction = c(0.3, 0.6, 0.95),
  num_iterations = c(200, 250, 300),
  number_stocks_chosen = c(10, 15, 20, 40)
)

training_log <- hyperparameter_grid_training_lgbm(df_with_features_train, param_df, 100, covariate_vars, categorical_vars)
training_log <- sort_data_frame(training_log, "ic", decreasing = T)
head(training_log)

################################################################################
# Evaluation of LGBM with some plots
################################################################################

lgbm_features_effects_plot(df_with_features_train, covariate_vars, categorical_vars, training_log[1, ])
lgbm_hyperparameters_marginal_effect_plot(training_log)
dev.off()

################################################################################
# Back-testing a trading algorithm on the validation set.
################################################################################

hyperparameters <- training_log[1, ]

bottom_liquid_covariates <- unique(get_bottom_n_liquid_assets(df_with_features_train, hyperparameters[10]$number_stocks_chosen)$symbol)

df_with_features_filtered <- get_filtered_given_symbols(df_with_features, bottom_liquid_covariates)
df_with_features_train_filtered <- get_filtered_given_symbols(df_with_features_train, bottom_liquid_covariates)
df_with_features_test_filtered <- get_filtered_given_symbols(df_with_features_test, bottom_liquid_covariates)

y_preds <- lgbm_get_validation_set_predictions(df_with_features_filtered, df_with_features_test_filtered, covariate_vars, categorical_vars, hyperparameters)
# This implements basic strategy of buy top 5 highest returns and short bottom 5 lowest returns
combined_position <- lgbm_get_positions_based_on_predictions(df_with_features_filtered, df_with_features_test_filtered, y_preds, hyperparameters)
# This implements Kelly Criterion
combined_position_kelly <- lgbm_get_positions_based_on_kelly(df_with_features_filtered, df_with_features_test_filtered, y_preds, hyperparameters)
combined_position_min_var <- lgbm_get_positions_based_on_wmv(df_with_features_filtered, df_with_features_test_filtered, y_preds, hyperparameters)
combined_position_mkt <- lgbm_get_positions_based_on_wmkt(df_with_features_filtered, df_with_features_test_filtered, y_preds, hyperparameters)

wealth_and_pnl <- get_pnl_based_on_position(df_with_features_filtered, df_with_features_test_filtered, combined_position)

dev.off()
performance_evaluation_of_wealth(wealth_and_pnl$wealth, wealth_and_pnl$daily_pnl, 0.03)
calculate_metrics(as.numeric(wealth_and_pnl$wealth), index(wealth_and_pnl$wealth), risk_free_rate = 0.03/250)

# Pruning Features:
# checking for highly correlated features (to remove redundant ones)
numeric_covariates <- df_with_features %>%
  dplyr::select(all_of(covariate_vars)) %>%
  dplyr::select(where(is.numeric))
cor_matrix <- cor(numeric_covariates, use = "pairwise.complete.obs")
heatmap(cor_matrix, symm = TRUE)

get_highly_correlated_pairs <- function(cor_matrix, threshold = 0.95) {
  cor_pairs <- which(abs(cor_matrix) > threshold, arr.ind = TRUE)
  cor_pairs <- cor_pairs[cor_pairs[, 1] < cor_pairs[, 2], , drop = FALSE] # remove duplicates
  data.frame(
    feature_1 = rownames(cor_matrix)[cor_pairs[, 1]],
    feature_2 = colnames(cor_matrix)[cor_pairs[, 2]],
    correlation = cor_matrix[cor_pairs]
  )
}

redundant_pairs <- get_highly_correlated_pairs(cor_matrix, threshold = 0.95)
as.data.frame(redundant_pairs)
redundant_pairs <- redundant_pairs %>%
  arrange(desc(correlation))
print(redundant_pairs)


# least important features (function added to training_functions.R)
least_important <- extract_least_important_features(
  df = df_with_features_train,
  training_log = training_log,
  covariate_vars = covariate_vars,
  categorical_vars = categorical_vars,
  response_var = "simple_returns_fwd_day_5"
)
least_important
