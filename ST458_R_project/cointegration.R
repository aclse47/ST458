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

source("training_functions.R")
source("feature_engineering.R")

# Load Data
df <- read.csv("df_train.csv")
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df <- df %>% arrange(symbol, date)

df_wide <- df %>%
  dplyr::select(date, symbol, close) %>%
  tidyr::pivot_wider(names_from = symbol, values_from = close)
df_wide <- as.data.frame(df_wide)

rownames(df_wide) <- df_wide$date
df_wide$date <- NULL

df_wide <- df_wide[, sapply(df_wide, is.numeric), drop = FALSE]

df_wide <- df_wide %>% mutate(across(everything(), ~ zoo::na.locf(.x, na.rm = FALSE)))

## Johansen Test & VECM
num_assets_per_group <- 10
johansen_results <- list()
vecm_models <- list()
residuals_list <- list()
asset_names <- colnames(df_wide)

for (i in seq(1, length(asset_names), by = num_assets_per_group)) {
  asset_subset <- asset_names[i:min(i + num_assets_per_group - 1, length(asset_names))]
  df_subset <- df_wide[, asset_subset, drop = FALSE]
  
  if (any(is.na(df_subset))) next  
  
  # Johansen test
  johansen_test <- ca.jo(df_subset, type = "trace", ecdet = "none", K = 2)
  trace_stat <- johansen_test@teststat
  critical_values <- johansen_test@cval
  
  print(paste("Group:", paste(asset_subset, collapse = ", ")))
  print("Trace Statistics:")
  print(trace_stat)
  print("Critical Values (5% level):")
  print(critical_values[, 2])
  
  significant_ranks <- which(trace_stat > critical_values[, 2])  # 5% level
  significant_ranks <- ifelse(length(significant_ranks) == 0, NA, length(significant_ranks))
  
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
    
    #  date column matches residuals length
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

write.csv(all_residuals, "residuals.csv", row.names = FALSE)

df_with_features <- left_join(df, all_residuals, by = "date")

