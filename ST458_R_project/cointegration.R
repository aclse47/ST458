## Cointegration 
library(putils)
library(tseries)
library(urca)
library(tidyverse)
library(vars)
library(dplyr)
library(tidyr)
library(TTR)
library(lightgbm)
library(zoo)

source("training_functions.R")
source("feature_engineering.R")

################################################################################
# EG ADF Test 
################################################################################
df <- read.csv('df_train.csv')
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df <- df %>% arrange(symbol, date)
add_future_simple_return <- function(df, periods_ahead = 1) {
  df %>%
    group_by(symbol) %>%
    arrange(date) %>%
    mutate(!!paste0("simple_returns_fwd_day_", periods_ahead) := (lead(close, n = periods_ahead) / close) - 1) %>%
    ungroup()
}
df <- add_future_simple_return(df, periods_ahead = 1)

train_idx <- df$date < '2013-01-01'
test_idx <- df$date >= '2013-01-01'

df_train <- df[train_idx, ]
df_test <- df[test_idx, ]

tickers <- unique(df$symbol)

df_wide_train <- df_train %>%
  dplyr::select(date, symbol, close) %>%
  tidyr::pivot_wider(names_from = symbol, values_from = close)

df_wide_test <- df_test %>%
  dplyr::select(date, symbol, close) %>%
  tidyr::pivot_wider(names_from = symbol, values_from = close)

rownames(df_wide_train) <- df_wide_train$date
df_wide_train$date <- NULL

rownames(df_wide_test) <- df_wide_test$date
df_wide_test$date <- NULL

df_wide_train <- df_wide_train %>% mutate(across(everything(), ~ zoo::na.locf(.x, na.rm = FALSE)))
df_wide_test <- df_wide_test %>% mutate(across(everything(), ~ zoo::na.locf(.x, na.rm = FALSE)))

asset_names <- colnames(df_wide_train)  

# EG ADF Test
cointegration_results <- list()

for (i in 1:(length(asset_names) - 1)) {  
  for (j in (i + 1):length(asset_names)) {
    
    asset1 <- df_wide_train[[asset_names[i]]]
    asset2 <- df_wide_train[[asset_names[j]]]
    
    if (all(is.na(asset1)) || all(is.na(asset2))) next  
    
    ols_model <- lm(asset1 ~ asset2)
    residuals <- residuals(ols_model)
    adf_test <- adf.test(residuals)
    
    cointegration_results[[paste(asset_names[i], asset_names[j], sep = " - ")]] <- adf_test$p.value
  }
}

cointegration_df <- data.frame(
  Pair = names(cointegration_results),
  P_Value = unlist(cointegration_results)
)

cointegrated_pairs <- cointegration_df %>% filter(P_Value < 0.05)

# Johansen test
m <- length(tickers)
coint_pval <- matrix(1, m, m)  
corr <- matrix(0, m, m)  

t <- nrow(df_wide_train)  

for (i in 1:(m-1)) {
  for (j in (i+1):m) {
    
    X <- df_wide_train[(t-251):t, c(tickers[i], tickers[j])]
    
    if (all(is.na(X[,1])) || all(is.na(X[,2]))) next
    
    corr[i, j] <- corr[j, i] <- cor(X[,1], X[,2], use = "complete.obs")
    
    johansen_test <- ca.jo(X, type = "trace", ecdet = "none", K = 2)
    coint_pval[i, j] <- coint_pval[j, i] <- johansen_test@teststat[1]
  }
}

# best cointegrated pair
valid_indices <- which(corr < 0.9, arr.ind = TRUE)
valid_values <- coint_pval[valid_indices]
min_index <- which.min(valid_values)
optimal_idx <- valid_indices[min_index, ]
idx1 <- optimal_idx[1]
idx2 <- optimal_idx[2]

best_pair <- c(tickers[idx1], tickers[idx2]) 
print(best_pair)

x_train <- as.numeric(df_wide_train[(t-251):t, best_pair[1]][[1]])
y_train <- as.numeric(df_wide_train[(t-251):t, best_pair[2]][[1]])

x_train <- zoo::na.locf(x_train, na.rm = FALSE)
y_train <- zoo::na.locf(y_train, na.rm = FALSE)

# OLS regression
ols <- lm(y_train ~ x_train)
coefs <- coef(ols)  
res <- residuals(ols)  

plot(res, type = "l")  

# trading on test data
future_dates <- 1:min(63, nrow(df_wide_test))  

x_test <- as.numeric(df_wide_test[[best_pair[1]]][future_dates])
y_test <- as.numeric(df_wide_test[[best_pair[2]]][future_dates])

x_test <- zoo::na.locf(x_test, na.rm = FALSE)
y_test <- zoo::na.locf(y_test, na.rm = FALSE)

new_res <- as.numeric(y_test - coefs[2] * x_test - coefs[1])
new_res[is.na(new_res)] <- 0 

sigma <- as.numeric(sd(res, na.rm = TRUE))

# trading parameters
position <- "flat"  
beta <- coefs[2]  
trades <- data.frame(day = integer(), share1 = numeric(), share2 = numeric(), price1 = numeric(), price2 = numeric())

for (i in 1:length(future_dates)) {
  
  prices <- as.numeric(df_wide_test[future_dates[i], best_pair])
  prices[is.na(prices)] <- mean(prices, na.rm = TRUE)  
  
  if (!is.na(new_res[i])) {
    
    if (position == "flat" && new_res[i] > sigma) {
      shares <- c(beta, -1)  
      shares <- shares / sum(abs(shares * prices))  
      trades <- rbind(trades, data.frame(day = i, share1 = shares[1], share2 = shares[2], price1 = prices[1], price2 = prices[2]))
      position <- "short"
      
    } else if (position == "flat" && new_res[i] < -sigma) {
      shares <- c(-beta, 1)  
      shares <- shares / sum(abs(shares * prices))  
      trades <- rbind(trades, data.frame(day = i, share1 = shares[1], share2 = shares[2], price1 = prices[1], price2 = prices[2]))
      position <- "long"
      
    } else if ((position == "long" && new_res[i] > 0) || 
               (position == "short" && new_res[i] < 0) || i == length(future_dates)) {
      
      if (nrow(trades) > 0) {
        shares <- -c(trades$share1[nrow(trades)], trades$share2[nrow(trades)])  
        trades <- rbind(trades, data.frame(day = i, share1 = shares[1], share2 = shares[2], price1 = prices[1], price2 = prices[2]))
      }
      position <- "flat"
    }
  }
}

colnames(trades) <- c("day", "share1", "share2", "price1", "price2")
trades # only one trade; crosses threshold only once 

dates <- sort(unique(df_test$date))
tickers <- sort(unique(df_test$symbol))

combined_position <- matrix(0, nrow = length(dates), ncol = length(tickers))
rownames(combined_position) <- as.character(dates)
colnames(combined_position) <- tickers

print(best_pair)
combined_position["2013-01-03", "GCD"] <- 0.5 # Long 
combined_position["2013-01-03", "AMWD"] <- -0.5 # Short

result <- get_pnl_based_on_position(df, df_test, combined_position)
wealth <- result$wealth
daily_pnl <- result$daily_pnl
performance_evaluation_of_wealth(wealth, daily_pnl, risk_free_rate = 0.03) # negative SR - only one trade; doesnt make sense to persue further 

################################################################################
# Johansen & Residuals 
################################################################################
df <- read.csv("df_train.csv") # Read in csv
df$date <- as.Date(df$date, format = "%Y-%m-%d") # Make the date column date instead of char
df <- df %>% arrange(symbol, date) # Order according to symbol then date like in case study lecture

df_wide <- df %>% # convert to wide so that each column = stock closing price over time
  dplyr::select(date, symbol, close) %>% # close price
  tidyr::pivot_wider(names_from = symbol, values_from = close)
df_wide <- as.data.frame(df_wide)

rownames(df_wide) <- df_wide$date
df_wide$date <- NULL

df_wide <- df_wide[, sapply(df_wide, is.numeric), drop = FALSE]

df_wide <- df_wide %>% mutate(across(everything(), ~ zoo::na.locf(.x, na.rm = FALSE))) # if price is NA replace w last known price (cannot have NA for Johansen)

# Johansen Test & VECM
num_assets_per_group <- 5
window_size <- 252 # rolling window of one year so that we only use past data
step_size <- 5 # update weekly
max_lag <- 2 # number of residual lags to compute

df_wide <- df %>%
  dplyr::select(date, symbol, close) %>%
  pivot_wider(names_from = symbol, values_from = close) %>%
  arrange(date)

df_wide <- df_wide %>%
  mutate(across(-date, ~ zoo::na.locf(.x, na.rm = FALSE))) %>%
  column_to_rownames("date")

all_lagged_residuals <- list()
all_dates <- as.Date(rownames(df_wide))
asset_names <- colnames(df_wide)

# loop through asset groups
for (g in seq(1, length(asset_names), by = num_assets_per_group)) {
  asset_subset <- asset_names[g:min(g + num_assets_per_group - 1, length(asset_names))]
  
  for (i in seq(window_size + max_lag, length(all_dates), by = step_size)) {
    date_range <- all_dates[(i - window_size - max_lag + 1):i]
    current_date <- all_dates[i]
    
    df_subset <- df_wide[as.character(date_range), asset_subset, drop = FALSE]
    if (anyNA(df_subset)) next
    
    # Johansen Test
    johansen_test <- tryCatch(
      ca.jo(df_subset, type = "trace", ecdet = "none", K = 2),
      error = function(e) NULL
    )
    if (is.null(johansen_test)) next
    
    trace_stat <- johansen_test@teststat
    crit_vals <- johansen_test@cval
    significant_ranks <- which(trace_stat > crit_vals[, 2]) # 5% level
    r <- ifelse(length(significant_ranks) == 0, 0, length(significant_ranks))
    
    # skip if no cointegration
    if (r == 0) next
    
    vecm_model <- cajorls(johansen_test, r = r)
    resids <- as.data.frame(residuals(vecm_model$rlm))
    
    # only take residuals for the final row (current prediction day)
    current_resid <- tail(resids, 1)
    current_resid$date <- current_date
    colnames(current_resid)[1:ncol(resids)] <- paste0(asset_subset[1:ncol(resids)], "_residual")
    
    all_lagged_residuals[[length(all_lagged_residuals) + 1]] <- current_resid
  }
}
head(all_lagged_residuals)

# merge residuals and reshape
residuals_df <- bind_rows(all_lagged_residuals)

residuals_long <- residuals_df %>%
  pivot_longer(-date, names_to = "symbol", values_to = "residual") %>%
  mutate(symbol = gsub("_residual", "", symbol))

residuals_long_clean <- residuals_long %>%
  filter(!is.na(residual)) %>%
  distinct(date, symbol, .keep_all = TRUE)

residuals_lagged <- residuals_long_clean %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(
    residual_lag1 = lag(residual, 1),
  ) %>%
  ungroup() %>%
  dplyr::select(date, symbol, residual_lag1)

residuals_lagged <- residuals_lagged %>% # standardizing residuals
  group_by(symbol) %>%
  mutate(
    residual_lag1_z = as.numeric(scale(residual_lag1))
  ) %>%
  ungroup()

df_with_residuals <- df %>%
  left_join(residuals_lagged %>% dplyr::select(date, symbol, residual_lag1_z), by = c("date", "symbol"))

df_with_residuals[df_with_residuals$symbol == "ACTS" & df_with_residuals$date == as.Date("2011-01-11"), ] # check to see if works

df_with_features <- add_features(df, dV_kalman = 10, dW_kalman = 0.0001)
df_with_features <- as.data.frame(df_with_features)
# df_with_features <- get_bottom_n_liquid_assets(df_with_features, 20)
head(df_with_features)

# used standardized lag 1 residuals in LGBM - lowered IC than without residuals (also experimented with lag 1, lag 2, standardized lag 2, and no lag)
