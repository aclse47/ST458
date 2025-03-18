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
library(tidyr)
source('feature_engineering_functions.R')
df <- read.csv('df_train.csv')
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df <- df %>% arrange(symbol, date)
df_with_features <- add_features(df)

tickers <- unique(df$symbol)



##########################################################################################
# DATA EXPLORATION - VISUAL
##########################################################################################

# Plotting the entire dataset
ggplot(df, aes(x = date, y = close, color = symbol, group = symbol)) +
  geom_line() +
  ggtitle("Close Prices for All Tickers") +
  theme_minimal()



# Plotting batch_size number of time series at a time.
# Change batch size depending on amount of time series you want to see.
# Plots time series in order of tickers object.
batch_size <- 5
for (t in seq(1, length(tickers), by = batch_size)){
  tickers_subset <- tickers[t: min(t+batch_size-1, length(tickers))]
  filtered_df <- df %>% filter(symbol %in% tickers_subset)
  
  p <- ggplot(filtered_df, aes(x = date, y = close, color=symbol, group=symbol)) +
    geom_line()+
    ggtitle(paste("Close Price for some tickers")) +
    theme_minimal()
  print(p)
}


# Plotting a single price ticker and all of it's features in a single plot.
plot_single_with_features <- function(df_with_features, symbol_){
  df_filtered <- df_with_features %>%
    filter(symbol == symbol_) %>%
    # select(date, close, where(~!grepl("fwd", colnames(df_with_features))))
    select(date, close, !contains("fwd"))
  print(nrow(df_with_features))
  print(nrow(df_filtered))
  
  # As an example, choosing to plot Bollinger bands with XBRQ Symbol
  df_long <- df_filtered %>%
   select(c(date, close, bollinger_bands_mavg_window_size_std_20_2, bollinger_bands_low_window_size_std_20_2, bollinger_bands_high_window_size_std_20_2)) %>%
   pivot_longer(cols = -date, names_to = "feature", values_to = "value")
  
  
  p <- ggplot(df_long, aes(x = date, y = value, color = feature)) +
    geom_line()+
    theme_minimal()
  print(p)
  
  return(df_long)
}


df_long <- plot_single_with_features(df_with_features, "WSM")





