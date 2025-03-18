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



# rank features according to importance in predicting future returns
lgbm_features_effects_plot <- function(df_train, covariate_vars, hyperparameters){
  bunch(train_length, valid_length, lookahead) %=% hyperparameters[1:3]
  response_var <- sprintf("simple_returns_fwd_day_%01d", hyperparameters$lookahead)
  
  params <- list(
    objective ='regression',
    num_iterations = hyperparameters$num_iterations,
    num_leaves = hyperparameters$num_leaves,
    learning_rate = hyperparameters$learning_rate,
    feature_fraction = hyperparameters$feature_fraction,
    bagging_fraction = hyperparameters$bagging_fraction
  )
   
  dtrain <- lgb.Dataset(data = as.matrix(df_train[, covariate_vars]), label = df_train[, response_var])
  model <- lgb.train(params = params, data = dtrain)
  
  
  importance <- lgb.importance(model)
  for (measure in c('Gain','Cover','Frequency')){
    lgb.plot.importance(importance, top_n=10, measure=measure)
  }
}

# Marginal effect of hyper-parameters on model performance.
lgbm_hyperparameters_marginal_effect_plot <- function(training_log){
  par(mfrow=c(3,3), mar=c(3,2.5,1,1), mgp=c(1.5,0.5,0))
  for (j in 1:9){
    plot(as.factor(training_log[[j]]), training_log$ic,
         xlab=colnames(training_log)[j])
  }
}




lgbm_get_validation_set_predictions <- function(df_all, df_test, covariate_vars, categorical_vars, hyperparameters){
  bunch(train_length, valid_length, lookahead) %=% hyperparameters[1:3]
  response_var <- sprintf("simple_returns_fwd_day_%01d", hyperparameters$lookahead)
  lgbm_params <- list(
    objective ='regression',
    num_iterations = hyperparameters$num_iterations,
    num_leaves = hyperparameters$num_leaves,
    learning_rate = hyperparameters$learning_rate,
    feature_fraction = hyperparameters$feature_fraction,
    bagging_fraction = hyperparameters$bagging_fraction
  ) 
  
  
  all_dates <- sort(unique(df_all$date))
  
  r1fwd <-  reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  y_pred <- matrix(NA, nrow(r1fwd), ncol(r1fwd), dimnames= dimnames(r1fwd))
  
  
  for (i in 1: nrow(y_pred)){
    date <- rownames(y_pred)[i]
    if (i %% valid_length == 1){
      date_idx <- match(date, all_dates)
      date_idx_end <- date_idx - lookahead
      date_idx_start <- date_idx_end - train_length + 1
      
      date_range <- all_dates[date_idx_start:date_idx_end]
      train_filter <- df_all$date %in% date_range
      
      
      
      dtrain <- lgb.Dataset(data=as.matrix(df_all[train_filter, covariate_vars]),
                            label=df_all[train_filter, response_var],
                            categorical_feature=categorical_vars)
      model <- lgb.train(params = lgbm_params, data = dtrain, verbose=-1)
    }
    
    
    test_filter <- df_all$date %in% date
    
    preds <- predict(model, as.matrix(df_all[test_filter, covariate_vars]))
    y_pred[date, df_all$symbol[test_filter]] <- preds
    
    printPercentage(date, row.names(y_pred))
  }
  return(y_pred)
}

# Implements a basic strategy of top 5 buy, bottom 5 sell equal amounts
lgbm_get_positions_based_on_predictions <- function(df_all, df_test, y_preds){
  all_dates <- sort(unique(df_all$date))
  
  r1fwd <-  reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  position <- matrix(NA, nrow(r1fwd), ncol(r1fwd), dimnames= dimnames(r1fwd))
  
  for (i in 1: (nrow(y_preds))){
    is_short <- rank(y_preds[i, ]) <= 5
    is_long <- rank(y_preds[i, ]) > 45
    position[i, ] <- 0
    position[i, is_short] <- -1/200
    position[i, is_long] <- 1/200
  }
  
  
  combined_position <- position
  for (i in 1:nrow(position)){
    j = max(1, i-20)
    combined_position[i,] <- colSums(position[j:i, , drop=FALSE])
  }
  
  print(max(rowSums(abs(combined_position))))
  
  return(combined_position)
}



get_pnl_based_on_position <- function(df_all, df_test, combined_position){
  all_dates <- sort(unique(df_all$date))
  
  r1fwd <-  reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  
  daily_pnl <- log(1 + rowSums(r1fwd * combined_position))
  daily_pnl <- xts(daily_pnl, order.by=as.Date(rownames(combined_position)))
  daily_pnl <- lag(daily_pnl)
  daily_pnl[1] <- 0
  
  wealth <- exp(cumsum(daily_pnl)) / 1
  return(list(wealth = wealth, daily_pnl = daily_pnl)) 
}


performance_evaluation_of_wealth <- function(wealth, daily_pnl, risk_free_rate){
  wealth_scaled <- wealth / as.numeric(wealth[1]) * 100
  
  getSymbols('SPY', from='2013-01-02', to='2014-01-01')
  SPY_price <- SPY$SPY.Adjusted
  SPY_scaled <- SPY_price / as.numeric(SPY_price[1]) * 100
  
  plot(cbind(wealth_scaled, SPY_scaled), legend.loc='topleft')
  
  SPY_ret <- diff(log(SPY_price))
  SPY_ret[1] <- 0
  
  sharpe_ratio <- mean(daily_pnl - risk_free_rate / 252) / sd(daily_pnl) * sqrt(252)
  
  
  print((daily_pnl))
  print((SPY_ret))
  
  correlation_of_returns <- cor(daily_pnl, SPY_ret, use = "complete.obs")
  
  cat("Coefficients\n")
  print(coef(lm(daily_pnl ~ SPY_ret)))
  cat("Sharpe Ratio: ", sharpe_ratio, "\n")
  cat("Correlation of Returns: ", correlation_of_returns, "\n")
}


