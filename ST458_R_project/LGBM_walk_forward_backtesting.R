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
library(dlm)


#####################################################################################################
#####################################################################################################
#####################################################################################################
# FEATURE ENGINEERING FUNCTIONS
#####################################################################################################
#####################################################################################################
#####################################################################################################




##########################################################################################
# FUNCTIONS TO ADD FEATURES
##########################################################################################

#-------------------------------------
# Returns related features
#-------------------------------------

# Adding simple returns
add_simple_returns_col <- function(df) {
  df_with_simple_returns <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(simple_returns = (close / lag(close)) - 1) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_simple_returns)
}

# Adding log returns
add_log_returns_col <- function(df) {
  df_with_log_returns <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(log_returns = log(close / lag(close))) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_log_returns)
}

add_past_simple_return <- function(df_with_simple_returns, periods_behind) {
  df_with_past_simple_returns <- df_with_simple_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("simple_returns_bwd_day_", periods_behind) := ((close / lag(close, n = periods_behind)) - 1)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_past_simple_returns)
}

add_past_log_return <- function(df_with_log_returns, periods_behind) {
  df_with_past_log_returns <- df_with_log_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("log_returns_bwd_day_", periods_behind) := log(close / lag(close, n = periods_behind))) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_past_log_returns)
}


#-------------------------------------
# Price related features
#-------------------------------------


add_price_anomalies_features <- function(df) {
  df_with_price_anomaly_features <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(
      close_to_open = (close - open) / open,
      close_to_high = (close - high) / high,
      close_to_low = (close - low) / low,
      high_low_range = (high - low) / low,
      body_size = abs(close - open),
      upper_wick = (high - pmax(open, close)) / body_size,
      lower_wick = (pmin(open, close) - low) / body_size
    ) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_price_anomaly_features)
}

add_gap_features <- function(df) {
  df_with_gaps <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(
      gap = (open - lag(close)) / lag(close), # Gap size
      # gap_up = ifelse(gap > 0, gap, 0),                      # Gap up
      # gap_down = ifelse(gap < 0, gap, 0),                    # Gap down
      abs_gap = abs(gap) # Absolute gap
    ) %>%
    ungroup() %>%
    arrange(symbol, date)

  return(df_with_gaps)
}

#-------------------------------------
# Volume related features
#-------------------------------------

add_vwap <- function(df, window_size) {
  df_with_vwap <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(VWAP = TTR::VWAP(price = close, volume = volume, n = window_size)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_vwap)
}


add_dollar_volume <- function(df) {
  df_with_dollar_vol <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(dollar_volume = close * volume) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_dollar_vol)
}

add_avg_dollar_volume <- function(df, window_size) {
  df_with_dollar_vol <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(avg_dollar_volume = zoo::rollapply(close * volume, window_size, mean, fill = NA, align = "right")) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_dollar_vol)
}

add_relative_volume <- function(df, window_size) {
  df_with_rvol <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(
      avg_vol = zoo::rollmean(volume, window_size, fill = NA, align = "right"),
      RVOL = volume / avg_vol
    ) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_rvol)
}


add_volume_shock <- function(df) {
  df_with_shock <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(volume_shock = (volume - lag(volume)) / lag(volume)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_shock)
}

#-------------------------------------
# Volatility related features
#-------------------------------------

# Adding Rolling Standard Deviation of log returns
# Parameter of window size
add_rolling_std_log_returns <- function(df_with_log_returns, window_size) {
  df_with_rolling_sd_log_returns <- df_with_log_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("rolling_sd_log_returns_window_size_", window_size) := rollapply(log_returns, width = window_size, FUN = sd, fill = NA, align = "right")) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_rolling_sd_log_returns)
}

# Adding exponential weighted moving average volatility (EWMAV)
add_exp_weighted_moving_avg_vol <- function(df_with_log_returns, window_size) {
  df_with_exp_weighted_moving_avg_vol <- df_with_log_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("exp_weighted_rolling_sd_log_returns_window_size_", window_size) := sqrt(TTR::EMA(log_returns^2, n = window_size, wilder = F))) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_exp_weighted_moving_avg_vol)
}

# Average True Range (ATR)
add_avg_true_range_vol <- function(df, window_size) {
  df_with_avg_true_range <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("avg_true_range_window_size_", window_size) := TTR::ATR(cbind(high, low, close), n = window_size)[, "atr"]) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_avg_true_range)
}

# Add Range (High-Low)
add_range_vol <- function(df) {
  df_with_range <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(range = high - low) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_range)
}

#-------------------------------------
# Momentum related features
#-------------------------------------

# Relative Strength Index (RSI)
add_relative_strength_index <- function(df, window_size) {
  df_with_rsi <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("relative_strength_index_window_size_", window_size) := TTR::RSI(close, n = window_size)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_rsi)
}

# Moving Average Convergence Divergence (MACD)
add_moving_average_convergence_divergence <- function(df, window_size_fast, window_size_slow, window_size_signal) {
  df_with_macd <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("macd_macd_line_fast_slow_signal_", window_size_fast, "_", window_size_slow, "_", window_size_signal) := TTR::MACD(close, nFast = window_size_fast, nSlow = window_size_slow, nSig = window_size_signal, maType = EMA)[, "macd"]) %>%
    mutate(!!paste0("macd_signal_line_fast_slow_signal_", window_size_fast, "_", window_size_slow, "_", window_size_signal) := TTR::MACD(close, nFast = window_size_fast, nSlow = window_size_slow, nSig = window_size_signal, maType = EMA)[, "signal"]) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_macd)
}


# Rate of Change (ROC) and other momentum related features
add_momentum_ROC_log_acceleration <- function(df, window_size) {
  df_with_momentum_features <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("rate_of_change_window_size_", window_size) := TTR::ROC(close, n = window_size, type = "discrete")) %>%
    mutate(!!paste0("log_return_momentum_window_size", window_size) := log(close / lag(close, window_size))) %>%
    mutate(!!paste0("price_acceleration_window_size", window_size) := close - 2 * lag(close, window_size) + lag(close, 2 * window_size)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_momentum_features)
}


#-------------------------------------
# Trend related features
#-------------------------------------

# Bollinger Bands
add_bollinger_bands <- function(df, window_size, std) {
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

# Moving Averages - SMA and EMA
add_moving_averages <- function(df, window_size_normal, window_size_exponential) {
  df_with_sma_ema <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("moving_average_window_size_", window_size_normal) := SMA(close, n = window_size_normal)) %>%
    mutate(!!paste0("exponential_moving_average_window_size_", window_size_exponential) := EMA(close, n = window_size_exponential)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_sma_ema)
}


add_kalman_filtered_data <- function(df, dV_kalman, dW_kalman) {
  kalman_model <- dlmModPoly(order = 1, dV = dV_kalman, dW = dW_kalman)
  df_with_kf_close <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("kalman_filtered_close_dV_", dV_kalman, "_dW_", dW_kalman) := dlmFilter(close, kalman_model)$m[-1]) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_kf_close)
}

#-------------------------------------
# Support/Resistance related features
#-------------------------------------

# support and resistance levels (rolling local min/max)
add_support_resistance <- function(df, window = 10) {
  df_with_levels <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(
      support = zoo::rollapply(low, width = window, FUN = min, fill = NA, align = "right"),
      resistance = zoo::rollapply(high, width = window, FUN = max, fill = NA, align = "right"),
      dist_to_support = (close - support) / close,
      dist_to_resistance = (resistance - close) / close,
      close_above_support = as.integer(close > support),
      close_below_resistance = as.integer(close < resistance)
    ) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_levels)
}

#-------------------------------------
# Seasonality related features
#-------------------------------------

# day of the week
add_day_of_week <- function(df) {
  df_with_dow <- df %>%
    mutate(day_of_week = lubridate::wday(date, label = FALSE) - 1) # Monday = 0
  return(df_with_dow)
}

# month of the year
add_month_of_year <- function(df) {
  df_with_month <- df %>%
    mutate(month_of_year = lubridate::month(date))
  return(df_with_month)
}

# quarter
add_quarter <- function(df) {
  df_with_quarter <- df %>%
    mutate(quarter = lubridate::quarter(date))
  return(df_with_quarter)
}

# week of year
add_week_of_year <- function(df) {
  df %>% mutate(week_of_year = lubridate::isoweek(date))
}

# month end
add_month_boundaries <- function(df) {
  df %>%
    mutate(
      is_month_start = as.integer(lubridate::day(date) <= 2),
      is_month_end = as.integer(lubridate::day(date) >= lubridate::days_in_month(date) - 1)
    )
}

# days until month end
add_days_until_month_end <- function(df) {
  df %>%
    mutate(days_until_month_end = lubridate::days_in_month(date) - lubridate::day(date))
}

#-------------------------------------
# Interaction related features
#-------------------------------------

add_volatility_x_momentum <- function(df, momentum_window_size = 21, vol_window_size = 21) {
  df_with_vol_x_momentum <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(vol_x_mom = zoo::rollapply(log(close / lag(close)), vol_window_size, sd, fill = NA, align = "right") * TTR::ROC(close, n = momentum_window_size)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_vol_x_momentum)
}


add_volatility_x_rsi <- function(df, vol_window_size = 21, rsi_period = 14) {
  df_with_vol_x_rsi <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(vol_x_rsi = zoo::rollapply(log(close / lag(close)), vol_window_size, sd, fill = NA, align = "right") * TTR::RSI(close, n = rsi_period)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_vol_x_rsi)
}


add_return_prev_vol <- function(df, vol_window = 30) {
  df_with_features <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(return_x_prev_vol = log(close / lag(close)) * lag(zoo::rollapply(log(close / lag(close)), vol_window, sd, fill = NA, align = "right"))) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_features)
}

add_range_rsi <- function(df, rsi_period = 14) {
  df_with_features <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(range_x_rsi = (high - low) * TTR::RSI(close, n = rsi_period)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_features)
}

add_macd_rsi <- function(df, rsi_period = 14) {
  df_with_features <- df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(macd_x_rsi = TTR::MACD(close)[, "macd"] * TTR::RSI(close, n = rsi_period)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_features)
}

##########################################################################################
# FUNCTIONS TO ADD TARGETS
##########################################################################################
#-------------------------------------
# Adding future returns
#-------------------------------------

add_future_simple_return <- function(df_with_simple_returns, periods_ahead) {
  df_with_future_simple_returns <- df_with_simple_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("simple_returns_fwd_day_", periods_ahead) := ((lead(close, n = periods_ahead) / close) - 1)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_future_simple_returns)
}

add_future_log_return <- function(df_with_log_returns, periods_ahead) {
  df_with_future_log_returns <- df_with_log_returns %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("log_returns_fwd_day_", periods_ahead) := log(lead(close, n = periods_ahead) / close)) %>%
    ungroup() %>%
    arrange(symbol, date)
  return(df_with_future_log_returns)
}


#####################################################################################################
#####################################################################################################
#####################################################################################################
# TRADING ALGO SECTION
#####################################################################################################
#####################################################################################################
#####################################################################################################



##########################################################################################
# FUNCTION FOR FINAL AGGREGAGATION, DOING ALL THE PREPROCESSING STEPS
##########################################################################################

add_features <- function(df,
                         vwap_window_size=20,
                         avg_dollar_volume_window_size = 20,
                         relative_volume_window_size = 20,
                         rolling_std_log_returns_window_size = 20,
                         exp_weighted_moving_avg_vol_window_size=20,
                         average_true_range_window_size = 14,
                         relative_strength_index_window_size = 14,
                         rate_of_change_window_size = 14,
                         sma_window_size = 20,
                         ema_window_size = 20,
                         dV_kalman = 7,
                         dW_kalman = 0.01,
                         prediction_period_1 = 1,
                         prediction_period_2 = 5,
                         prediction_period_3 = 21,
                         prediction_period_4 = 63,
                         prediction_period_5 = 126) {
  df_with_features <- df %>%
    # Add returns
    add_simple_returns_col() %>%
    add_log_returns_col() %>%
    add_past_simple_return(prediction_period_2) %>%
    add_past_log_return(prediction_period_2) %>%
    add_past_simple_return(prediction_period_3) %>%
    add_past_simple_return(prediction_period_4) %>%
    # Add price related features
    add_gap_features() %>%
    # Add volume related features
    add_vwap(vwap_window_size) %>%
    add_dollar_volume() %>%
    add_avg_dollar_volume(avg_dollar_volume_window_size) %>%
    add_relative_volume(relative_volume_window_size) %>%
    add_volume_shock() %>%
    # Add volatility measures
    add_rolling_std_log_returns(rolling_std_log_returns_window_size) %>%
    add_exp_weighted_moving_avg_vol(exp_weighted_moving_avg_vol_window_size) %>%
    add_avg_true_range_vol(average_true_range_window_size) %>%
    add_range_vol() %>%
    # Add momentum measures
    add_relative_strength_index(relative_strength_index_window_size) %>%
    # Add trend-based measures
    add_kalman_filtered_data(dV_kalman, dW_kalman) %>%
    # Add Seasonality-based features
    add_day_of_week() %>%
    add_month_of_year() %>%
    add_month_boundaries() %>%
    add_days_until_month_end() %>%
    # Add interaction-based features
    add_volatility_x_momentum() %>%
    add_volatility_x_rsi() %>%
    add_return_prev_vol() %>%
    add_range_rsi() %>%
    add_macd_rsi() %>%
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

###################################################################################################################################
# HYPERPARAMETERS/GLOBAL VARS
###################################################################################################################################

# column variable names from df we're going to use to train - NEED TO MANUALLY TYPE OUT
response_var <- "simple_returns_fwd_day_5"

covariate_vars <- c(
  "open", "close", "low", "high", "volume", "simple_returns", 
  "log_returns", "simple_returns_bwd_day_5", "log_returns_bwd_day_5", 
  "simple_returns_bwd_day_21", "simple_returns_bwd_day_63", "gap", 
  "abs_gap", "VWAP", "dollar_volume", "avg_dollar_volume", "avg_vol", 
  "RVOL", "volume_shock", "rolling_sd_log_returns_window_size_20", 
  "exp_weighted_rolling_sd_log_returns_window_size_20", 
  "avg_true_range_window_size_14", "range", "relative_strength_index_window_size_14", 
  "kalman_filtered_close_dV_10_dW_1e-04", "day_of_week", "month_of_year", 
  "is_month_start", "is_month_end", "days_until_month_end", 
  "vol_x_mom", "vol_x_rsi", "return_x_prev_vol", 
  "range_x_rsi", "macd_x_rsi"
)
categorical_vars <- c('month_of_year', 'day_of_week')
bottom_liquid_covariates <- c("ACTS", "BCDM", "CLYQ", "HJC", "MWQN", "NMQ", "PNJC", "WSM", "XBRQ", "YVNL")
named_vector_all_tickers <- c(
  "ACTS" = 0, "AMWD" = 0, "ARV" = 0, "BBY" = 0, "BCDM" = 0,
  "BZK" = 0, "BZQM" = 0, "CDM" = 0, "CDRX" = 0, "CLYQ" = 0,
  "DLT" = 0, "DLYK" = 0, "DMTK" = 0, "DRX" = 0, "DYT" = 0,
  "EKXB" = 0, "EZX" = 0, "FBR" = 0, "FJX" = 0, "GCD" = 0,
  "GTFX" = 0, "GTPX" = 0, "GXR" = 0, "GZLT" = 0, "HCLF" = 0,
  "HJC" = 0, "HLC" = 0, "HMY" = 0, "HNTV" = 0, "HNVR" = 0,
  "HVS" = 0, "HXNJ" = 0, "HYWQ" = 0, "ISNR" = 0, "JBDX" = 0,
  "JFK" = 0, "JQKC" = 0, "JQR" = 0, "JZKP" = 0, "KJD" = 0,
  "KJYC" = 0, "KRV" = 0, "KXRV" = 0, "KZMB" = 0, "LKC" = 0,
  "LWF" = 0, "LZK" = 0, "LZT" = 0, "MBW" = 0, "MNG" = 0,
  "MQV" = 0, "MWQN" = 0, "MZNP" = 0, "NMQ" = 0, "NQY" = 0,
  "NVDT" = 0, "NVP" = 0, "OWLR" = 0, "PJYC" = 0, "PNJC" = 0,
  "PNY" = 0, "PQRS" = 0, "PWL" = 0, "QRM" = 0, "QVG" = 0,
  "QVNR" = 0, "RFXP" = 0, "RMG" = 0, "RWXT" = 0, "RYTX" = 0,
  "RYX" = 0, "SJF" = 0, "SWRY" = 0, "SXC" = 0, "SXJD" = 0,
  "SYDR" = 0, "TKL" = 0, "TLP" = 0, "TLWM" = 0, "TLXN" = 0,
  "TPLF" = 0, "TQY" = 0, "VBN" = 0, "VKNT" = 0, "VXT" = 0,
  "VZXB" = 0, "WMGX" = 0, "WQX" = 0, "WSM" = 0, "WVJC" = 0,
  "WYLR" = 0, "WZB" = 0, "XBRQ" = 0, "XFG" = 0, "XPT" = 0,
  "XTG" = 0, "YPN" = 0, "YRD" = 0, "YVNL" = 0, "ZQN" = 0
)

# # Time interval related hyperparams
train_length <- 252
valid_length <- 63
lookahead <- 5

# # LGBM related hyperparams
num_leaves <- 100
min_data_in_leaf <- 1000
learning_rate <- 0.2
feature_fraction <- 0.95
bagging_fraction <- 0.3
num_iterations <- 300

best_lgbm_params <- list(
  objective = 'regression',
  num_iterations = num_iterations,
  num_leaves = num_leaves,
  learning_rate = learning_rate,
  feature_fraction = feature_fraction,
  bagging_fraction = bagging_fraction
)

#####################################################################################################
# HELPER FUNCTIONS FOR WALK FORWARD BACKTESTING
#####################################################################################################
create_list_df_format <- function(data_preprocessed, response_var){
  df_list <- split(data_preprocessed, data_preprocessed$date)
  df_list <- lapply(df_list, function(sub_df){
    tmp <- matrix(NA, length(unique(data_preprocessed$symbol)), length(colnames(data_preprocessed)) - 2)
    rownames(tmp) <- sub_df$symbol
    colnames(tmp) <- colnames(sub_df)[-c(1, 2)]
    tmp[sub_df$symbol, ] <- as.matrix(sub_df[, -c(1, 2)])
    tmp[, response_var] <- NA
    return(tmp)
  })
  
  for (i in 1:train_length){
    df_list[[i]][, response_var] <- df_list[[i+5]][, 'simple_returns_bwd_day_5']
  }
  return(df_list)
}

update_positions_and_get_trades <- function(positions, lookahead, preds){
  position_to_close <- positions[lookahead, ]
  positions[2:lookahead, ] <- positions[1:(lookahead-1), ]
  is_short <- rank(preds) <= 5
  is_long <- rank(preds) > length(colnames(positions)) - 5
  positions[1, ] <- 0
  positions[1, is_short] <- -1/200
  positions[1, is_long] <- 1/200
  trades <- positions[1, ] - position_to_close
  return(list(trades = trades, positions = positions))
}

get_filtered_given_symbols <- function(df_with_features, symbols){
  df_with_features_filtered <- df_with_features %>% filter(symbol %in% symbols)
  return(df_with_features_filtered)
}

#####################################################################################################
# IMPORTANT: FUNCTIONS TO SUBMIT
#####################################################################################################

# Input has to be all data in the training set.
# External variables - 
  # train_length, lookahead, bottom_liquid_covariates
  # These are obtained from HYPERPARAMETERS sections above.
initialise_state <- function(data){
  
  unique_dates <- sort(as.Date(unique(data$date)))
  dates_recent <- tail(unique_dates, train_length + lookahead)
  
  data <- get_filtered_given_symbols(data, bottom_liquid_covariates)

  #Get recent most recent length(lookahead+train_length) dates 
  data <- data[data$date %in% dates_recent, ]
  
  data_preprocessed <- as.data.frame(add_features(data, dV_kalman = 10, dW_kalman = 0.0001))
  # Pre-processing df to get it into list format.
  df_list <- create_list_df_format(data_preprocessed, 'simple_returns_fwd_day_5')
   
  # Creating positions matrix.
  positions <- matrix(0, lookahead, length(unique(data_preprocessed$symbol)))
  colnames(positions) <- rownames(df_list[[1]])
  
  
  
  state <- list(day_idx=0, 
                positions=positions, 
                df_recent=df_list, 
                model=NULL,
                data = data[data$date %in% tail(unique_dates, 35), ])
  
  return(state)
}


# Input has to be the most recent 252 observations, and outputs latest 252 observations
# External variables - 
  # train_length, lookahead, covariate_vars, categorical_vars, best_lgbm_params, response_var, bottom_liquid_covariates, named_vector_all_tickers
  # These are obtained from HYPERPARAMETERS sections above.
trading_algorithm <- function(new_data, state){
  bunch(day_idx, positions, df_recent, model, data) %=% state
  new_data <- get_filtered_given_symbols(new_data, bottom_liquid_covariates)
  new_unique_date <- sort(as.Date(unique(new_data$date)))
  # Add last value to df
  data_with_new_data <- rbind(data, new_data) %>% arrange(symbol, date)
  
  #Add features to this in a rolling fashion
  data_with_new_data_with_features <- as.data.frame(add_features(data_with_new_data, dV_kalman = 10, dW_kalman = 0.0001))
  
  # Get the most recent data with features.
  new_data_with_features <- data_with_new_data_with_features[data_with_new_data_with_features$date %in% new_unique_date, ]
  day_idx = day_idx + 1
  
  
  df_recent[[1]] <- NULL
  #Convert new data to a matrix and append it to df_recent
  tmp <- matrix(NA, length(unique(new_data_with_features$symbol)), length(colnames(new_data_with_features)) - 2)
  rownames(tmp) <- rownames(tail(df_recent, 1)[[1]])
  colnames(tmp) <- colnames(tail(df_recent, 1)[[1]])
  tmp[new_data_with_features$symbol, ] <- as.matrix(new_data_with_features[, -c(1,2)])
  new_date <- as.character(new_data_with_features$date[1])
  df_recent[[new_date]] <- tmp
  # Update the future returns of the past variable
  df_recent[[train_length]][, 'simple_returns_fwd_day_5'] <- df_recent[[new_date]][, 'simple_returns_bwd_day_5']
  
  # Retrain model if valid_length time has passed
  if (day_idx %% valid_length == 1){
    train_mx <- do.call(rbind, df_recent[1:train_length])
    dtrain <- lgb.Dataset(data=train_mx[, covariate_vars], 
                          label=train_mx[, 'simple_returns_fwd_day_5'],
                          categorical_feature=categorical_vars)
    model <- lgb.train(params = best_lgbm_params, data = dtrain, verbose=-1)
  }
  
  # Use model to predict
  preds <- predict(model, df_recent[[new_date]][, covariate_vars])
  bunch(trades, positions) %=% update_positions_and_get_trades(positions, lookahead, preds)
  unique_recent_dates <- sort(as.Date(unique(data_with_new_data$date)))
  new_state <- list(day_idx=day_idx, 
                    positions=positions,
                    df_recent=df_recent, 
                    model=model, 
                    # Get only the most recent data from df i.e. minus first value
                    data=data_with_new_data[data_with_new_data$date %in% tail(unique_recent_dates, 252), ])
  
  #Expand trades (Needed for code to work if predicting < 100 assets)
  trades_all_tickers <- named_vector_all_tickers
  trades_all_tickers[names(trades)] <- trades
  
  return(list(trades=trades_all_tickers, new_state=new_state))
}
