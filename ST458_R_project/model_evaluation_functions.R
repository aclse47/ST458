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
      
      
      # date_range is in unix epoch
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





# Implements a basic strategy of top 5 buy, bottom 5 sell equal amounts.
# Results in positions across assets along the time frame.
lgbm_get_positions_based_on_predictions <- function(df_all, df_test, y_preds, hyperparameters){
  bunch(train_length, valid_length, lookahead) %=% hyperparameters[1:3]
  
  all_dates <- sort(unique(df_all$date))
  r1fwd <-  reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  
  position <- matrix(NA, nrow(r1fwd), ncol(r1fwd), dimnames= dimnames(r1fwd))
  
  for (i in 1: (nrow(y_preds))){
    is_short <- rank(y_preds[i, ]) <= 5  
    is_long <- rank(y_preds[i, ]) > 95   
    position[i, ] <- 0                   
    position[i, is_short] <- -1/200      
    position[i, is_long] <- 1/200        
    
    }
  
  combined_position <- position
  for (i in 1:nrow(position)){
    j = max(1, i- (lookahead - 1))
    combined_position[i,] <- colSums(position[j:i, , drop=FALSE])
  }
  
  cat("Max absolute exposure for 1 wealth:", max(rowSums(abs(combined_position))) ,"\n")
  
  return(combined_position)
}





# Backward looking portfolio allocation i.e. no point in doing predictions
lgbm_get_positions_based_on_wmv <- function(df_all, df_test, y_preds, hyperparameters, allocation_factor = 0.5){
  bunch(train_length, valid_length, lookahead) %=% hyperparameters[1:3]
  
  all_dates <- sort(unique(df_all$date))
  test_dates <- sort(unique(df_test$date))
  
  r1fwd_test <- reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd_test) <- r1fwd_test$date
  r1fwd_test <- as.matrix(r1fwd_test[, -1])
  position <- matrix(0, nrow(r1fwd_test), ncol(r1fwd_test), dimnames = dimnames(r1fwd_test))
  
  r1fwd <- reshape2::dcast(df_all, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  
  lookback_window <- 252
  n_assets <- ncol(r1fwd)
  
  for (i in 1:nrow(y_preds)){
    current_date <- test_dates[i]
    current_idx <- which(all_dates == current_date)
    hist_dates <- all_dates[max(1, (current_idx-(lookahead)) - lookback_window):(current_idx-(lookahead+1))]
    
    lookback_returns <- r1fwd[row.names(r1fwd) %in% hist_dates, , drop=F]
    cov_matrix <- cov(lookback_returns, use = "pairwise.complete.obs")
    
    min_var_weights <- solve(cov_matrix) %*% rep(1, n_assets)
    min_var_weights <- min_var_weights / sum(min_var_weights)
    min_var_weights <- min_var_weights * allocation_factor
    
    
    position[i, ] <- min_var_weights
  }
  
  combined_position <- position
  for (i in 1:nrow(position)){
    j = max(1, i- (lookahead - 1))
    combined_position[i,] <- colSums(position[j:i, , drop=FALSE])
  }
  
  cat("Max absolute exposure for 1 wealth:", max(rowSums(abs(combined_position))) ,"\n")
  
  return(combined_position)
}






















# Still technically backward looking but replacing expected return of stock with predicted future return.
lgbm_get_positions_based_on_wmkt <- function(df_all, df_test, y_preds, hyperparameters, allocation_factor = 1, risk_free_rate = 0.03){
  bunch(train_length, valid_length, lookahead) %=% hyperparameters[1:3]
  
  all_dates <- sort(unique(df_all$date))
  test_dates <- sort(unique(df_test$date))
  
  r1fwd_test <- reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd_test) <- r1fwd_test$date
  r1fwd_test <- as.matrix(r1fwd_test[, -1])
  position <- matrix(0, nrow(r1fwd_test), ncol(r1fwd_test), dimnames = dimnames(r1fwd_test))
  
  r1fwd <- reshape2::dcast(df_all, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  
  lookback_window <- 252
  risk_free_rate <- risk_free_rate / 252
  n_assets <- ncol(r1fwd)
  
  for (i in 1:nrow(y_preds)){
    current_date <- test_dates[i]
    daily_preds <- y_preds[i, ]
    excess_returns <- daily_preds - risk_free_rate
    current_idx <- which(all_dates == current_date)
    hist_dates <- all_dates[max(1, (current_idx-(lookahead)) - lookback_window):(current_idx-(lookahead+1))]
    
    lookback_returns <- r1fwd[row.names(r1fwd) %in% hist_dates, , drop=F]
    
    mu <- colMeans(lookback_returns, na.rm = T)
    cov_matrix <- cov(lookback_returns, use = "pairwise.complete.obs")
    # Replace excess returns with mu to get based on expectation of past returns
    # Use excess_returns to use predictions
    tangency_weights <- solve(cov_matrix) %*% excess_returns 
    tangency_weights <- tangency_weights / sum(abs(tangency_weights))
    tangency_weights <- tangency_weights * allocation_factor
    position[i, ] <- tangency_weights
  }
  
  combined_position <- position
  for (i in 1:nrow(position)){
    j = max(1, i- (lookahead - 1))
    combined_position[i,] <- colSums(position[j:i, , drop=FALSE])
  }
  
  cat("Max absolute exposure for 1 wealth:", max(rowSums(abs(combined_position))) ,"\n")
  
  return(combined_position)
}






# Kelly Criterion
# Still technically backward looking but replacing expected return of stock with predicted future return.
lgbm_get_positions_based_on_kelly <- function(df_all, df_test, y_preds, hyperparameters, risk_free_rate = 0.03){
  bunch(train_length, valid_length, lookahead) %=% hyperparameters[1:3]
  
  all_dates <- sort(unique(df_all$date))
  test_dates <- sort(unique(df_test$date))
  
  r1fwd_test <- reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd_test) <- r1fwd_test$date
  r1fwd_test <- as.matrix(r1fwd_test[, -1])
  position <- matrix(0, nrow(r1fwd_test), ncol(r1fwd_test), dimnames = dimnames(r1fwd_test))
  
  r1fwd <- reshape2::dcast(df_all, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  
  lookback_window <- 252
  risk_free_rate <- risk_free_rate / 252
  
  for (i in 1:nrow(y_preds)){
    current_date <- test_dates[i]
    daily_preds <- y_preds[i, ]
    current_idx <- which(all_dates == current_date)
    
    hist_dates <- all_dates[max(1, (current_idx-(lookahead)) - lookback_window):(current_idx-(lookahead+1))]
    lookback_returns <- r1fwd[row.names(r1fwd) %in% hist_dates, , drop=F]
    mu <- colMeans(lookback_returns, na.rm = T)
    cov_matrix <- cov(lookback_returns, use = "pairwise.complete.obs")
    excess_returns <- daily_preds - risk_free_rate
    
    # Replace excess returns with mu to get based on expectation of past returns
    # Use excess_returns to use predictions
    market_weights <- solve(cov_matrix) %*% excess_returns
    market_weights <- market_weights / sum(abs(market_weights))
    
    
    portfolio_return <- sum(market_weights * excess_returns)
    portfolio_variance <- t(market_weights) %*% cov_matrix %*% market_weights
    
    kelly_leverage <- as.numeric(portfolio_return / portfolio_variance)
    
    max_leverage <- 1 
    applied_leverage <- min(kelly_leverage, max_leverage)
    
    position[i, ] <- market_weights * applied_leverage
    # position[i, ] <- market_weights * kelly_leverage
  }
  
  
  combined_position <- position
  for (i in 1:nrow(position)){
    j = max(1, i- (lookahead - 1))
    combined_position[i,] <- colSums(position[j:i, , drop=FALSE])
  }
  
  cat("Max absolute exposure for 1 wealth:", max(rowSums(abs(combined_position))) ,"\n")
  
  return(combined_position)
}





# We input the positions taken on each day
# We output the PnL of the strategy and the wealth if we had invested $1 the beginning.
get_pnl_based_on_position <- function(df_all, df_test, combined_position){
  all_dates <- sort(unique(df_all$date))
  
  r1fwd <-  reshape2::dcast(df_test, date ~ symbol, value.var = 'simple_returns_fwd_day_1')
  row.names(r1fwd) <- r1fwd$date
  r1fwd <- as.matrix(r1fwd[, -1])
  
  # daily PnL in log returns - allows us to cumulative sum to get returns over periods of time.
  daily_pnl <- log(1 + rowSums(r1fwd * combined_position))
  daily_pnl <- xts(daily_pnl, order.by=as.Date(rownames(combined_position)))
  # We lag because the PnL is realised at the end of the next day
  # We assume trades are executed right before close of each day
  #   This implies at the end of first day, we should have PnL of 0
  daily_pnl <- lag(daily_pnl)
  daily_pnl[1] <- 0
  
  # Cumulative summing to get returns relative to first day on each day
  wealth <- exp(cumsum(daily_pnl)) / 1
  return(list(wealth = wealth, daily_pnl = daily_pnl))
}






performance_evaluation_of_wealth <- function(wealth, daily_pnl, risk_free_rate){
  wealth_scaled <- wealth / as.numeric(wealth[1]) * 100
  
  getSymbols('SPY', from='2013-01-02', to='2014-01-01')
  SPY_price <- SPY$SPY.Adjusted
  SPY_scaled <- SPY_price / as.numeric(SPY_price[1]) * 100
  
  p <- plot(cbind(wealth_scaled, SPY_scaled), legend.loc='topleft')
  print(p)
  SPY_ret <- diff(log(SPY_price))
  SPY_ret[1] <- 0
  
  sharpe_ratio <- mean(daily_pnl - risk_free_rate / 252) / sd(daily_pnl) * sqrt(252)
  
  cat("Sharpe Ratio: ", sharpe_ratio, "\n")
}


