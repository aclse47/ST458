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

source('training_functions.R')
source('feature_engineering_functions.R')
source('model_evaluation_functions.R')

################################################################################
# Data Preprocessing & Cointegration 
################################################################################

df <- read.csv('df_train.csv')                    # Read in csv
df$date <- as.Date(df$date, format = "%Y-%m-%d")  # Make the date column date instead of char
df <- df %>% arrange(symbol, date)                # Order according to symbol then date like in case study lecture

df_wide <- df %>%                        # convert to wide so that each column = stock closing price over time
  dplyr::select(date, symbol, close) %>% # close price 
  tidyr::pivot_wider(names_from = symbol, values_from = close)
df_wide <- as.data.frame(df_wide)

rownames(df_wide) <- df_wide$date
df_wide$date <- NULL

df_wide <- df_wide[, sapply(df_wide, is.numeric), drop = FALSE]

df_wide <- df_wide %>% mutate(across(everything(), ~ zoo::na.locf(.x, na.rm = FALSE))) # if price is NA replace w last known price (cannot have NA for Johansen)

# Johansen Test & VECM
num_assets_per_group <- 5 
johansen_results <- list()
vecm_models <- list()
residuals_list <- list()
asset_names <- colnames(df_wide)

# loop thru assets and get a subset 
for (i in seq(1, length(asset_names), by = num_assets_per_group)) {
  asset_subset <- asset_names[i:min(i + num_assets_per_group - 1, length(asset_names))]
  df_subset <- df_wide[, asset_subset, drop = FALSE]
  
  if (any(is.na(df_subset))) next  
  
  # Johansen 
  johansen_test <- ca.jo(df_subset, type = "trace", ecdet = "none", K = 2) # trace stat for cointegration
  trace_stat <- johansen_test@teststat 
  critical_values <- johansen_test@cval
  
  significant_ranks <- which(trace_stat > critical_values[, 2])  # 5% level
  significant_ranks <- ifelse(length(significant_ranks) == 0, NA, length(significant_ranks)) # if no rank, set NA
  
  johansen_results[[paste(asset_subset, collapse = ", ")]] <- list(
    Trace_Stats = trace_stat,
    Significant_Ranks = significant_ranks
  )
  
  # VECM if cointegration is found
  if (!is.na(significant_ranks) && significant_ranks > 0) {
    vecm_model <- cajorls(johansen_test, r = significant_ranks)
    vecm_models[[paste(asset_subset, collapse = ", ")]] <- vecm_model
    
    vecm_residuals <- residuals(vecm_model$rlm)
    residuals_df <- as.data.frame(vecm_residuals)
    
    #  align residuals with dates
    if (nrow(residuals_df) != nrow(df_wide)) {
      residuals_df$date <- tail(df$date, nrow(residuals_df))
    } else {
      residuals_df$date <- df$date
    }
    
    colnames(residuals_df)[1:ncol(vecm_residuals)] <- paste0(asset_subset[1:ncol(vecm_residuals)], "_residual")
    
    residuals_list[[paste(asset_subset, collapse = ", ")]] <- residuals_df
  }
}

if (length(residuals_list) > 0) {
  all_residuals <- Reduce(function(x, y) full_join(x, y, by = "date"), residuals_list)
} else {
  all_residuals <- data.frame(date = df$date)
}

head(all_residuals)

residuals_df <- as.data.frame(all_residuals)
residuals_df$date <- as.Date(residuals_df$date, format = "%Y-%m-%d")

residuals_long <- residuals_df %>%
  pivot_longer(-date, names_to = "symbol", values_to = "residual") %>%
  mutate(symbol = gsub("_residual", "", symbol))

residuals_all <- residuals_long %>% # adding lag 1 and lag 2 residuals
  group_by(symbol) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(
    residual_lag1 = lag(residual, 1),
    residual_lag2 = lag(residual, 2)
  ) %>%
  ungroup()

df_with_residuals <- df %>%
  left_join(residuals_all, by = c("date", "symbol"))                              

head(df_with_residuals[df_with_residuals$symbol == "TPLF", ]) # check to see if works

tickers <- unique(df$symbol)

df_with_features <- add_features(df_with_residuals, dV_kalman = 10, dW_kalman = 0.0001)
df_with_features <- as.data.frame(df_with_features)
head(df_with_features)

################################################################################
# LGBM
################################################################################

response_vars <- colnames(df_with_features %>% dplyr::select(matches("fwd")))
covariate_vars <- setdiff(colnames(df_with_features), c(response_vars, 'date', 'symbol'))
# Deliberately leaking in data to see how it performs.
# Note: We get 160% rate of return!
# covariate_vars <- c(covariate_vars, 'simple_returns_fwd_day_5')

categorical_vars <- c()

df_with_features_train <- df_with_features[df_with_features$date < as.Date('2013-01-01'), ]
df_with_features_test <- df_with_features[df_with_features$date >= as.Date('2013-01-01'), ]

# # Hyper-parameter combination grid
# param_df <- expand.grid(
#   train_length = c(252, 252*2, 126),
#   valid_length = c(21, 63),
#   lookahead = c(5),
#   num_leaves = c(5,10,50),
#   min_data_in_leaf = c(250,1000),
#   learning_rate = c(0.01,0.03,0.1),
#   feature_fraction = c(0.3,0.6,0.95),
#   bagging_fraction = c(0.3,0.6,0.95),
#   num_iterations = c(30,200)
#   
#   # atr_window = c(14, 20, 50),
#   # sma_window = c(10, 20, 50, 200),
#   # ema_window = c(10, 20, 50, 200),
#   # rsi_window = c(7, 14, 21),
#   # macd_fast = c(12, 26),
#   # macd_slow = c(26, 50)
# )
# 
# param_df <- expand.grid(
#   train_length = c(252, 252*2),
#   valid_length = c(21, 63),
#   lookahead = c(5),
#   num_leaves = c(50, 75, 100),
#   min_data_in_leaf = c(250,1000),
#   learning_rate = c(0.1, 0.15, 0.2),
#   feature_fraction = c(0.3,0.6,0.95),
#   bagging_fraction = c(0.3,0.6,0.95),
#   num_iterations = c(200, 250, 300)
#   
#   # atr_window = c(14, 20, 50),
#   # sma_window = c(10, 20, 50, 200),
#   # ema_window = c(10, 20, 50, 200),
#   # rsi_window = c(7, 14, 21),
#   # macd_fast = c(12, 26),
#   # macd_slow = c(26, 50)
# )


param_df <- expand.grid(
  train_length = c(252*2),
  valid_length = c(21, 63),
  lookahead = c(5),
  num_leaves = c(50, 75, 100),
  min_data_in_leaf = c(250,1000),
  learning_rate = c(0.1, 0.15, 0.2, 0.5),
  feature_fraction = c(0.95, 1.00),
  bagging_fraction = c(0.3,0.6,0.95),
  num_iterations = c(200, 250, 300)
)

training_log <- hyperparameter_grid_training_lgbm(df_with_features_train, param_df, 100, covariate_vars, categorical_vars)
training_log <- sort_data_frame(training_log, 'ic', decreasing=T)
head(training_log)
################################################################################
# Evaluation of LGBM with some plots
################################################################################

lgbm_features_effects_plot(df_with_features_train, covariate_vars, training_log[1, ])
lgbm_hyperparameters_marginal_effect_plot(training_log)$lookahead
dev.off()
lgbm_hyperparameters_marginal_effect_plot(lookahead$training_log)

################################################################################
# Back-testing a trading algorithm on the validation set.
################################################################################

hyperparameters <- training_log[1, ]
y_preds <- lgbm_get_validation_set_predictions(df_with_features, df_with_features_test, covariate_vars, categorical_vars, hyperparameters)

# This implements basic strategy of buy top 5 highest returns and short bottom 5 lowest returns
combined_position <- lgbm_get_positions_based_on_predictions(df_with_features, df_with_features_test, y_preds, hyperparameters)

# This implements Kelly Criterion
combined_position_kelly <- lgbm_get_positions_based_on_kelly(df_with_features, df_with_features_test, y_preds, hyperparameters)
combined_position_min_var <- lgbm_get_positions_based_on_wmv(df_with_features, df_with_features_test, y_preds, hyperparameters)
combined_position_mkt <- lgbm_get_positions_based_on_wmkt(df_with_features, df_with_features_test, y_preds, hyperparameters)

wealth_and_pnl <- get_pnl_based_on_position(df_with_features, df_with_features_test, combined_position)

performance_evaluation_of_wealth(wealth_and_pnl$wealth, wealth_and_pnl$daily_pnl, 0.03)
