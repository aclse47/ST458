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

# use close price for cointegration 
# convert to wide so that each column = stock closing price over time
df_wide <- df %>%
  dplyr::select(date, symbol, close) %>%
  tidyr::pivot_wider(names_from = symbol, values_from = close)
df_wide <- as.data.frame(df_wide)

rownames(df_wide) <- df_wide$date
df_wide$date <- NULL

df_wide <- df_wide[, sapply(df_wide, is.numeric), drop = FALSE]

df_wide <- df_wide %>% mutate(across(everything(), ~ zoo::na.locf(.x, na.rm = FALSE))) # if price is NA replace w last known price (cannot have NA for johansen)

## Johansen Test & VECM
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

residuals_long <- residuals_df %>%
  pivot_longer(-date, names_to = "symbol", values_to = "residual") %>%
  mutate(symbol = gsub("_residual", "", symbol))

################################################################################
# Residual plots
################################################################################
ggplot(residuals_long, aes(x = date, y = residual)) +
  geom_line() +
  facet_wrap(~ symbol, scales = "free_y") +
  labs(title = "Residuals by Asset", x = "Date", y = "Residual") +
  theme_minimal()

# oscilate around zero - VECM is capturing the relationships 

# EKXB_residual corresponds to EKXB, EZX_residual corresponds to EZX, etc

# residuals represent the part of the assets returns that are not explained by the cointegration relationship 

# VECM fits a model that assumes we want equilibrum, the residuals shows actual value vs exptected long run equilibrium 

# could possibly contain predictions for features in lgbm - if asset goes back to equilibrium 

colnames(all_residuals)

# significant at lag 1 
acf(residuals_df$EKXB_residual)
acf(residuals_df$EZX_residual)
acf(residuals_df$FBR_residual)
acf(residuals_df$FJX_residual)
acf(residuals_df$GCD_residual)
acf(residuals_df$HVS_residual)
acf(residuals_df$HXNJ_residual)
acf(residuals_df$HYWQ_residual)
acf(residuals_df$ISNR_residual)
acf(residuals_df$JBDX_residual)
acf(residuals_df$SYDR_residual)
acf(residuals_df$TKL_residual)
acf(residuals_df$TLP_residual)
acf(residuals_df$TLWM_residual)
acf(residuals_df$TLXN_residual)
acf(residuals_df$TPLF_residual)
acf(residuals_df$TQY_residual)
acf(residuals_df$VBN_residual)
acf(residuals_df$VKNT_residual)
acf(residuals_df$VXT_residual)

# adding to df: 

residuals_df$date <- as.Date(residuals_df$date, format = "%Y-%m-%d")

residuals_all <- residuals_long %>%
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
head(df_with_residuals)
