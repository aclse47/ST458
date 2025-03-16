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



##########################################################################################
# FUNCTIONS TO ADD FEATURES
##########################################################################################

#-------------------------------------
# Returns related features
#-------------------------------------

# Adding simple returns
add_simple_returns_col <- function(df){
  df_with_simple_returns <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(simple_returns = (close / lag(close)) - 1) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_simple_returns)  
}

# Adding log returns
add_log_returns_col <- function(df){
  df_with_log_returns <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(log_returns = log(close / lag(close))) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_log_returns)
}


#-------------------------------------
# Volatility related features
#-------------------------------------

# Adding Rolling Standard Deviation of log returns
  # Parameter of window size
add_rolling_std_log_returns <- function(df_with_log_returns, window_size){
  df_with_rolling_sd_log_returns <- df_with_log_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("rolling_sd_log_returns_window_size_", window_size) := rollapply(log_returns, width = window_size, FUN = sd, fill = NA, align = "right")) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_rolling_sd_log_returns)
}

# Adding exponential weighted moving average volatility (EWMAV)
add_exp_weighted_moving_avg_vol <- function(df_with_log_returns, window_size){
   df_with_exp_weighted_moving_avg_vol <- df_with_log_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("exp_weighted_rolling_sd_log_returns_window_size_", window_size) := sqrt(TTR::EMA(log_returns^2, n=window_size, wilder=F))) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_exp_weighted_moving_avg_vol)
}

# Average True Range (ATR)
add_avg_true_range_vol<- function(df, window_size){
  df_with_avg_true_range <- df %>%
   group_by(symbol) %>%
   arrange(date) %>%
   mutate(!!paste0("avg_true_range_window_size_", window_size) := TTR::ATR(cbind(high, low, close), n = window_size)[, 'atr']) %>%
   ungroup() %>%
   arrange(symbol, date)
  return(df_with_avg_true_range)
}


#-------------------------------------
# Momentum related features
#-------------------------------------

# TODO: Relative Strength Index





##########################################################################################
# FUNCTIONS TO ADD TARGETS
##########################################################################################
#-------------------------------------
# Adding future returns
#-------------------------------------

add_future_simple_return <- function(df_with_simple_returns, periods_ahead){
  df_with_future_simple_returns <- df_with_simple_returns %>%
   group_by(symbol) %>%
   arrange(date) %>%
   mutate(!!paste0("simple_returns_fwd_day_", periods_ahead) := lead(simple_returns, n=periods_ahead)) %>%
   ungroup() %>%
   arrange(symbol, date)
  return(df_with_future_simple_returns)
}

add_future_log_return <- function(df_with_log_returns, periods_ahead){
  df_with_future_log_returns <- df_with_log_returns %>%
   group_by(symbol) %>%
   arrange(date) %>%
   mutate(!!paste0("log_returns_fwd_day_", periods_ahead) := lead(log_returns, n=periods_ahead)) %>%
   ungroup() %>%
   arrange(symbol, date)
  return(df_with_future_log_returns)
}




##########################################################################################
# FUNCTION FOR FINAL AGGREGAGATION, DOING ALL THE PREPROCESSING STEPS
##########################################################################################

add_features <- function(df, 
                         rolling_std_log_returns_window_size=20, 
                         exp_weighted_moving_avg_vol_window_size=20, 
                         average_true_range_window_size=14,
                         prediction_period_1=1,
                         prediction_period_2=5,
                         prediction_period_3=21){
  df_with_features <- df %>% 
    # Add simple returns
    add_simple_returns_col() %>% 
    add_log_returns_col() %>% 
    # Add volatility measures
    add_rolling_std_log_returns(rolling_std_log_returns_window_size) %>% 
    add_exp_weighted_moving_avg_vol(exp_weighted_moving_avg_vol_window_size) %>%
    add_avg_true_range_vol(average_true_range_window_size) %>%
    # TODO: Add momentum measures
    
    # TODO: Add trend-based measures
    
    # Add targets 
    add_future_simple_return(prediction_period_1) %>%
    add_future_log_return(prediction_period_1) %>%
    add_future_simple_return(prediction_period_2) %>%
    add_future_log_return(prediction_period_2) %>%
    add_future_simple_return(prediction_period_3) %>%
    add_future_log_return(prediction_period_3) %>%
  return(df_with_features)    
}