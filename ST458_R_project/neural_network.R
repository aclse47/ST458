## Cointegration with NN

library(putils)
library(tseries)
library(urca)
library(tidyverse)
library(vars)
library(dplyr)
library(tidyr)
library(tensorflow)
library(reticulate)
library(keras3) 
library(quantmod)
library(xts)
library(ggplot2)

use_python("/opt/anaconda3/envs/r-tensorflow/bin/python", required = TRUE)
tensorflow::tf_config()

source("training_functions.R")
source("feature_engineering.R")
source("model_evaluation_functions.R")


df <- read.csv("df_train.csv")
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df <- df %>% arrange(symbol, date)

df_wide <- df %>%
  dplyr::select(date, symbol, close) %>%
  tidyr::pivot_wider(names_from = symbol, values_from = close)

rownames(df_wide) <- df_wide$date
df_wide$date <- NULL

df_wide <- df_wide[, sapply(df_wide, is.numeric), drop = FALSE]

## Johansen Test & VECM
num_assets_per_group <- 10
johansen_results <- list()
vecm_models <- list()
residuals_list <- list()
asset_names <- colnames(df_wide)

for (i in seq(1, length(asset_names), by = num_assets_per_group)) {
  asset_subset <- asset_names[i:min(i + num_assets_per_group - 1, length(asset_names))]
  df_subset <- df_wide[, asset_subset, drop = FALSE]
  
  johansen_test <- ca.jo(df_subset, type = "trace", ecdet = "none", K = 2)
  trace_stat <- johansen_test@teststat
  critical_values <- johansen_test@cval
  significant_ranks <- which(trace_stat > critical_values[, 2]) # 5% level
  
  significant_ranks <- ifelse(length(significant_ranks) == 0, NA, length(significant_ranks))
  
  johansen_results[[paste(asset_subset, collapse = ", ")]] <- list(
    Trace_Stats = trace_stat,
    Significant_Ranks = significant_ranks
  )
  
  # fit VECM if cointegration is found
  if (!is.na(significant_ranks) && significant_ranks > 0) {
    vecm_model <- cajorls(johansen_test, r = significant_ranks)
    vecm_models[[paste(asset_subset, collapse = ", ")]] <- vecm_model
    
    vecm_residuals <- residuals(vecm_model$rlm)
    residuals_df <- as.data.frame(vecm_residuals)
    residuals_df$date <- tail(df$date, nrow(residuals_df))
    colnames(residuals_df)[1:ncol(vecm_residuals)] <- paste0(asset_subset, "_residual")
    
    residuals_list[[paste(asset_subset, collapse = ", ")]] <- residuals_df
  }
}

all_residuals <- Reduce(function(x, y) full_join(x, y, by = "date"), residuals_list)
df_with_features <- left_join(df, all_residuals, by = "date")

df_with_features <- add_features(df_with_features)
df_with_features <- as.data.frame(df_with_features)
df_with_features <- df_with_features %>%
  mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) # replacing NAs with column mean

train_length <- 126
valid_length <- 21

unique_dates <- sort(unique(df$date))
train_idx <- which(df$date >= unique_dates[1] & df$date <= unique_dates[train_length])
valid_idx <- which(df$date >= unique_dates[train_length + 1] & df$date <= unique_dates[train_length + valid_length])

response_vars <- df_with_features %>% dplyr::select(contains("fwd")) %>% colnames()
response_var <- response_vars[1]  # first forward return 

covariate_vars <- setdiff(colnames(df_with_features), c(response_vars, 'date', 'symbol'))

# scaling
X_train <- scale(as.matrix(df_with_features[train_idx, covariate_vars, drop = FALSE]))
X_valid <- scale(as.matrix(df_with_features[valid_idx, covariate_vars, drop = FALSE]), 
                 center = attr(X_train, "scaled:center"), 
                 scale = attr(X_train, "scaled:scale"))

y_train <- df_with_features[train_idx, response_var]
y_valid <- df_with_features[valid_idx, response_var]

first <- layer_dense(units = 128, activation = 'relu', input_shape = c(ncol(X_train)))
second <- layer_dropout(rate = 0.3)
third <- layer_dense(units = 64, activation = 'relu', kernel_regularizer = regularizer_l2(0.01))
fourth <- layer_dropout(rate = 0.3) 
fifth <-   layer_dense(units = 32, activation = 'relu') 
output_layer <- layer_dense(units = 1)

model <- keras_model_sequential()

model %>% 
  first %>% 
  second %>% 
  third %>% 
  fourth %>% 
  fifth %>% 
  output_layer

model$compile(optimizer = optimizer_adam(learning_rate = 0.0005), loss = 'mean_squared_error')

early_stop <- callback_early_stopping(
  monitor = "val_loss", 
  patience = 10, 
  restore_best_weights = TRUE  
)

fit(model, X_train, y_train, epochs=27, batch_size=32, validation_data = list(X_valid, y_valid),  callbacks = list(early_stop) )

y_pred <- predict(model, X_valid)
ic_in_fold <- cor(y_pred, y_valid, method = "spearman")  
print(ic_in_fold)

pnl_data <- get_pnl_based_on_position(df_with_features, df_with_features$date >= '2013-01-01', combined_position)

wealth <- pnl_data$wealth
daily_pnl <- pnl_data$daily_pnl

risk_free_rate <- 0.02
performance_evaluation_of_wealth(wealth, daily_pnl, risk_free_rate)

performance_evaluation_of_wealth <- function(wealth, daily_pnl, risk_free_rate)