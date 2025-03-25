# We shouldn't have to send this script to him in the end. These are functions helping us train our models. 
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


# Time-series cross validation
# Follows the below diagram
# |-------------------------------------------------|
# |-----Train-----|L|--Val--|
#         |-----Train-----|L|--Val--|
#                 |-----Train-----|L|--Val--|
#                         |-----Train-----|L|--Val--|
# Notice the non-overlapping validation sets.
time_series_split <- function(dates, n_splits=0, train_length=126, valid_length=21, lookahead=1){
  unique_dates <- sort(unique(dates), decreasing=T)
  
  if (n_splits == 0){
    block_length <- train_length + valid_length + lookahead
    n_splits <- (length(unique_dates) - block_length) %/% valid_length
  }
  
  valid_end_idx <- ((1:n_splits) - 1) * valid_length + 1
  valid_start_idx <- valid_end_idx + valid_length - 1
  train_end_idx <- valid_start_idx + lookahead + 1
  train_start_idx <- train_end_idx + train_length - 1
  
  
  splits <- list()
  for (i in 1:n_splits){
    train = which(dates >= unique_dates[train_start_idx[i]] & dates <= unique_dates[train_end_idx[i]])
    test = which(dates >= unique_dates[valid_start_idx[i]] & dates <= unique_dates[valid_end_idx[i]])
    splits[[i]] <- list(train=train, test = test)
  }
  
  return(splits)
}


# Compute Information Coefficient
# Spearman's correlation coefficient: Used when we want to check monotonic relationship instead of a linear correlation 
# i.e. does one increase as other increases?

# In quantitative trading, Spearman Information Coefficient (Spearman IC) is more commonly used than Pearson Information Coefficient (Pearson IC).
# Why Spearman IC is preferred:
#   Spearman IC measures rank correlation, which captures the order of predictions relative to actual returns, rather than the exact numerical values.
# It is robust to outliers and non-linear relationships, which are common in financial markets.
# Financial returns often exhibit non-normal distributions and heavy tails, making Spearman IC more appropriate.
# But it does ignore magnitude of prediction
# Doesn't assume a linear relationship, could be a quadratic, exponential, etc.
# As long as one increases with the other.


# Date in: 1050 rows
# Date unique: 21
# For each unique day, there are 50 stocks
# y_true, y_pred: 1050
# Within each day of 21 days, there are 50 stocks.
# There are the real returns of 50 stocks and then our predicted returns of the 50 stocks
# We calculate the Spearman's coefficient of these 50 stocks within each day
# It returns 21 Spearman coefficients - one for each day

compute_ic <- function(date, y_true, y_pred){
  sub_dfs <- split(data.frame(y_true=y_true, y_pred=y_pred), date)
  spearman_cor <- rep(0, length(sub_dfs))
  for (i in 1:length(sub_dfs)){
    sub_df <- sub_dfs[[i]]
    spearman_cor[i] <- tryCatch(
      cor(sub_df$y_true, sub_df$y_pred, method='spearman'),
      error = function(e) return(0), # If there's an error/warning, return 0
      warning=function(w) return(0)
    )
  }
  return(spearman_cor)
}







# Conducts training across the splits for a single parameter combination
n_fold_training_lgbm <- function(df_with_features,
                                 splits, 
                                 covariate_var, 
                                 categorical_var, 
                                 response_var,
                                 lgbm_params
                                 ){
  nfolds <- length(splits)
  ic_by_day <- numeric()
  
  for (fold in 1: nfolds){
    bunch(train_idx, valid_idx) %=% splits[[fold]]
    
    dtrain <- lgb.Dataset(data = as.matrix(df_with_features[train_idx, covariate_var]),
                          label = as.numeric(df_with_features[[response_var]][train_idx]),  
                          categorical_feature = as.character(categorical_var))
    
    
    model <- lgb.train(params = lgbm_params, data = dtrain, verbose = -1)
    y_pred <- predict(model, as.matrix(df_with_features[valid_idx, covariate_var]))
    y_pred <- unname(y_pred)
    y_true <- as.numeric(df_with_features[[response_var]][valid_idx])  
    ic_in_fold <- compute_ic(df_with_features[valid_idx,'date'], y_true, y_pred)
    ic_by_day <- c(ic_by_day, ic_in_fold)
    
    cor(y_true, y_pred)
  }
  return(mean(ic_by_day))
}






# Conducts training for all params in a hyper-parameter grid
hyperparameter_grid_training_lgbm <- function(df_with_features, hyper_parameter_grid, num_param_comb, covariate_var, categorical_var){
  
  set.seed(1)
  training_log <- hyper_parameter_grid[sample(nrow(hyper_parameter_grid), num_param_comb), ]
  training_log$ic <- 0
  
  
  for (i in 1:num_param_comb){
    bunch(train_length, valid_length, lookahead, num_leaves,
          min_data_in_leaf, learning_rate, feature_fraction,
          bagging_fraction, num_iterations) %=% training_log[i, 1:9]
    
    lgbm_params = list(
      objective = 'regression',
      num_iterations = num_iterations,
      num_leaves = num_leaves,
      learning_rate = learning_rate,
      feature_fraction = feature_fraction,
      bagging_fraction = bagging_fraction
    )
    
    
    response_var <- sprintf("simple_returns_fwd_day_%01d", training_log$lookahead[i])

    #df_with_features <- df_with_features[!is.na(df_with_features[[response_var]]), ]
    
    splits <- time_series_split(df_with_features$date, train_length = train_length, valid_length = valid_length, lookahead = lookahead)
    
    
    training_log$ic[i] <- mean_ic <- n_fold_training_lgbm(df_with_features, splits, covariate_var, categorical_var, response_var, lgbm_params)
    printPercentage(i, num_param_comb)
  }
  
  return(training_log)
}





