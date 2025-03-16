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

df <- read.csv('df_train.csv')
df$date <- as.Date(df$date, format = "%d/%m/%Y")
df <- df %>% arrange(symbol, date)

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








