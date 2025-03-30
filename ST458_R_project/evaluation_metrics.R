# evaluation metrics 

library(PerformanceAnalytics)
library(quantmod)
library(tidyverse)
library(zoo)

# using 252 as trading days 
# might need to adjust for what we choose rf to be
# make sure trading algo stores daily returns based on changes in wealth, then used these to daily returns as input for calculate_metrics(): 


calculate_metrics <- function(wealth, dates, risk_free_rate_annual = 0.03) { # rf rate set as daily 
  returns <- c(0, diff(wealth) / head(wealth, -1)) # daily returns 
  returns_xts <- xts(returns, order.by = as.Date(dates)) 
  annual_return <- mean(returns, na.rm = TRUE) * 252
  rf_daily <- risk_free_rate_annual / 252
  
  sharpe <- SharpeRatio(returns_xts, Rf = rf_daily, FUN = "StdDev", annualize = TRUE)
  sortino <- SortinoRatio(returns_xts, MAR = rf_daily, annualize = TRUE)
  max_drawdown <- maxDrawdown(returns_xts)
  expected_shortfall <- ES(returns_xts, p = 0.95, method = "historical")
  
  daily_excess_returns <- returns - rf_daily
  annual_excess_return <- mean(daily_excess_returns, na.rm = TRUE) * 252
  annual_excess_sd <- sd(daily_excess_returns, na.rm = TRUE) * sqrt(252)
  rf_adjusted_sharpe <- annual_excess_return / annual_excess_sd
  
  calmar <- ifelse(is.na(max_drawdown), NA, annual_return / abs(max_drawdown))
  omega <- as.numeric(Omega(returns_xts, threshold = rf_daily))
  ulcer <- sqrt(mean((cummax(returns) - returns)^2, na.rm = TRUE))
  
  metrics_df <- data.frame(
    Sharpe_Ratio = as.numeric(sharpe),
    RF_Adjusted_Sharpe = rf_adjusted_sharpe,
    Sortino_Ratio = as.numeric(sortino),
    Maximum_Drawdown = max_drawdown,
    Expected_Shortfall = expected_shortfall,
    Calmar_Ratio = calmar,
    Omega_Ratio = omega,
    Ulcer_Index = ulcer
  )
  
  rownames(metrics_df) <- NULL
  return(metrics_df)
}

# 1. Sharpe Ratio  
# Measures return per unit of risk (volatility).  
# Higher values indicate better risk-adjusted performance.  
# Use Case: Compare different investments with the same level of risk.  

# 2. Risk-Free Adjusted Sharpe Ratio  
# Similar to Sharpe but adjusted for the actual risk-free rate  
# Helps compare performance in different interest rate environments.  

# 3. Sortino Ratio  
# Similar to Sharpe, but only considers downside risk (bad volatility).  
# Ignores "good" volatility (price jumps up).  
# Higher values indicate better performance per unit of downside risk.  
# Ideal for investors who care more about avoiding losses than big gains.  

# 4. Maximum Drawdown  
# The biggest loss from peak to trough in portfolio value.  
# If Max Drawdown = -30%, the worst drop from a peak was 30%.  
# Shows how much pain an investor would have suffered in a worst-case scenario.  

# 5. Expected Shortfall (Conditional VaR, CVaR)  
# Measures average loss during the worst 5% of cases.  
# More conservative than Max Drawdown or Value at Risk (VaR).  
# Lower Expected Shortfall indicates less catastrophic risk.  
# Used by risk managers to see how bad things can get in extreme market conditions.  

# 6. Calmar Ratio  
# Measures return versus drawdown risk (instead of volatility like Sharpe).  
# Calmar Ratio = Annualized Return / Maximum Drawdown.  
# Helps assess risk-adjusted performance, especially for funds with high drawdowns.  

# 7. Omega Ratio  
# Compares probability-weighted gains vs. losses beyond a threshold return.  
# More flexible than Sharpe because it considers the entire distribution of returns.  
# Useful for non-normal return distributions and alternative investments.  

# 8. Ulcer Index  
# Measures downside risk by looking at deep and prolonged drawdowns.  
# Lower values indicate smoother performance with fewer sharp declines.  
# Helps assess portfolios with prolonged losing periods.  

