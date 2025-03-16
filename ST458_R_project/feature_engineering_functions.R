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

# Relative Strength Index (RSI)
add_relative_strength_index <- function(df, window_size){
  df_with_rsi <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("relative_strength_index_window_size_", window_size) := TTR::RSI(close, n=window_size)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_rsi)
}

# Moving Average Convergence Divergence (MACD)
add_moving_average_convergence_divergence <- function(df, window_size_fast, window_size_slow, window_size_signal){
  df_with_macd <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("macd_macd_line_fast_slow_signal_", window_size_fast, "_", window_size_slow, "_", window_size_signal) := TTR::MACD(close, nFast = window_size_fast, nSlow = window_size_slow, nSig = window_size_signal, maType = EMA)[, 'macd']) %>%
    mutate(!!paste0("macd_signal_line_fast_slow_signal_", window_size_fast, "_", window_size_slow, "_", window_size_signal) := TTR::MACD(close, nFast = window_size_fast, nSlow = window_size_slow, nSig = window_size_signal, maType = EMA)[, 'signal']) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_macd)
}


# Rate of Change (ROC)
add_rate_of_change <- function(df, window_size){
   df_with_roc <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("rate_of_change_window_size_", window_size) := TTR::ROC(close, n=window_size, type = 'discrete')) %>%
    ungroup() %>%
    arrange(symbol, date)
}


#-------------------------------------
# Trend related features
#-------------------------------------

# Bollinger Bands
add_bollinger_bands <- function(df, window_size, std){
  df_with_bb <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("bollinger_bands_mavg_window_size_std_", window_size, "_", std) := TTR::BBands(close, n = window_size, sd = std, maType = SMA)[, "mavg"]) %>%
    mutate(!!paste0("bollinger_bands_low_window_size_std_", window_size, "_", std) := TTR::BBands(close, n = window_size, sd = std, maType = SMA)[, "dn"]) %>%
    mutate(!!paste0("bollinger_bands_high_window_size_std_", window_size, "_", std) := TTR::BBands(close, n = window_size, sd = std, maType = SMA)[, "up"]) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_bb)
}

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
                         relative_strength_index_window_size=14,
                         moving_average_convergence_divergence_window_size_fast=12,
                         moving_average_convergence_divergence_window_size_slow=26,
                         moving_average_convergence_divergence_window_size_signal=9,
                         rate_of_change_window_size=14,
                         bb_window_size=20,
                         bb_std=2,
                         prediction_period_1=1,
                         prediction_period_2=5,
                         prediction_period_3=21,
                         prediction_period_4=63,
                         prediction_period_5=126){
  df_with_features <- df %>% 
    # Add simple returns
    add_simple_returns_col() %>% 
    add_log_returns_col() %>% 
    # Add volatility measures
    add_rolling_std_log_returns(rolling_std_log_returns_window_size) %>% 
    add_exp_weighted_moving_avg_vol(exp_weighted_moving_avg_vol_window_size) %>%
    add_avg_true_range_vol(average_true_range_window_size) %>%
    # Add momentum measures
    add_relative_strength_index(relative_strength_index_window_size) %>%
    add_moving_average_convergence_divergence(moving_average_convergence_divergence_window_size_fast,
                                              moving_average_convergence_divergence_window_size_slow,
                                              moving_average_convergence_divergence_window_size_signal) %>%
    add_rate_of_change(rate_of_change_window_size) %>%
    # TODO: Add trend-based measures
    add_bollinger_bands(bb_window_size, bb_std) %>%
    # Add targets 
    add_future_simple_return(prediction_period_1) %>%
    add_future_log_return(prediction_period_1) %>%
    add_future_simple_return(prediction_period_2) %>%
    add_future_log_return(prediction_period_2) %>%
    add_future_simple_return(prediction_period_3) %>%
    add_future_log_return(prediction_period_3) %>%
    add_future_simple_return(prediction_period_4) %>%
    add_future_log_return(prediction_period_4) %>%
    add_future_simple_return(prediction_period_5) %>%
    add_future_log_return(prediction_period_5)
  return(df_with_features)    
}