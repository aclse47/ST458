## Cointegration 

library(putils)
library(tseries)
library(urca)
library(tidyverse)
library(vars)

df <- read.csv('df_train.csv')
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df <- df %>% arrange(symbol, date)
df_with_features <- add_features(df)

tickers <- unique(df$symbol)

# pivot data to wide format
df_wide <- df %>%
  select(date, symbol, close) %>%
  spread(key = symbol, value = close)

# set date as row names 
rownames(df_wide) <- df_wide$Date
df_wide$Date <- NULL

# EG ADF Test 
# loops through all possible pairs and runs an OLS regression, tests the stationarity of residuals using ADF test, then filters based on p-value 
# only detects pairwise relationships, not as robust as johansens (see below)
cointegration_results <- list()
asset_names <- colnames(df_wide)

for (i in 1:(length(asset_names)-1)) { # -1 to avoid duplicate comparisons
  for (j in (i+1):length(asset_names)) {
    
    asset1 <- df_wide[[i]]
    asset2 <- df_wide[[j]]
    
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

# filter pairs that are cointegrated (p-value < 0.05)
cointegrated_pairs <- cointegration_df %>% filter(P_Value < 0.05)
strongly_cointegrated_pairs <- cointegration_df %>% filter(P_Value < 0.01)

print(cointegrated_pairs)
head(cointegration_df)


# Johansen Test
# tried to create a function to test all combinations of groups of 10 assets but it was way too much for R 
# applies the johansen test to rolling groups of 10 assets at a time. this at least test all the assets. 
# since johansens detects cointegration we can fit a VECM from the test results:
df_wide <- df_wide[, sapply(df_wide, is.numeric), drop = FALSE]

num_assets_per_group <- 10  

johansen_results <- list()
asset_names <- colnames(df_wide)
vecm_models <- list()  

for (i in seq(1, length(asset_names), by = num_assets_per_group)) {
  asset_subset <- asset_names[i:min(i + num_assets_per_group - 1, length(asset_names))]
  df_subset <- df_wide[, asset_subset, drop = FALSE]
  
  johansen_test <- ca.jo(df_subset, type = "trace", ecdet = "none", K = 2)
  
  trace_stat <- johansen_test@teststat # trace stat tells us how strongly cointegrated the assets are 
  critical_values <- johansen_test@cval
  
  significant_ranks <- which(trace_stat > critical_values[, 2])  # 5% significance level
  
  # store numerica value even if empty 
  if (length(significant_ranks) == 0) {
    significant_ranks <- NA  
  } else {
    significant_ranks <- length(significant_ranks)  
  }
  
  johansen_results[[paste(asset_subset, collapse = ", ")]] <- list(
    Trace_Stats = trace_stat,
    Significant_Ranks = significant_ranks
  )
  
  # fit VECM
  if (!is.na(significant_ranks) && significant_ranks > 0) {
    vecm_model <- cajorls(johansen_test, r = significant_ranks)
    vecm_models[[paste(asset_subset, collapse = ", ")]] <- vecm_model
  }
}


johansen_df <- data.frame(
  Group = names(johansen_results),
  Trace_Stats = sapply(johansen_results, function(x) paste(round(x$Trace_Stats, 2), collapse = ", ")),
  Significant_Ranks = sapply(johansen_results, function(x) x$Significant_Ranks)
)

# convert to numeric (NA to 0)
johansen_df$Significant_Ranks <- as.numeric(johansen_df$Significant_Ranks)
johansen_df$Significant_Ranks[is.na(johansen_df$Significant_Ranks)] <- 0

strongly_cointegrated_groups <- johansen_df %>%
  arrange(desc(Significant_Ranks)) %>%
  filter(Significant_Ranks > 0)  # ensure we only get valid results

print(strongly_cointegrated_groups) 
# indicates that the asset group (RYX, SJF, SWRY, SXC, SXJD, SYDR, TKL, TLP, TLWM, TLXN) is cointegrated with 2 significant cointegration ranks - likely moving together in the long run 
# since there are 2 significant cointegration ranks that suggests there are two independent long term relationships between these 10 assets

# VECM Models for the Most Cointegrated Groups:
lapply(vecm_models, summary) 
# beta = 20 (2 groups of 10 assets) - represents the long term equilibrium relationship
# rlm = estimated regressions for each variable (short term dynamics)

cointegration_vectors <- vecm_model$beta
print(cointegration_vectors)

# we could then use the cointegration vectors to compute the spread. then when the spread deviated form zero the assets have diverged from equilibrim, meaning a trading opportunity: 
# first cointegration vector
cointegration_vector <- cointegration_vectors[, 1]

spread <- as.matrix(df_subset) %*% cointegration_vector
spread <- as.vector(spread)

spread_mean <- mean(spread, na.rm = TRUE)
spread_sd <- sd(spread, na.rm = TRUE)
z_score <- (spread - spread_mean) / spread_sd

# trading signals
signals <- ifelse(z_score > 2, -1, ifelse(z_score < -2, 1, 0)) # buy if z-score < -2, sell if z-score > 2
