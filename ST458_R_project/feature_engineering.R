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

source('feature_engineering_functions.R')



df <- read.csv('df_train.csv')                    # Read in csv
df$date <- as.Date(df$date, format = "%Y-%m-%d")  # Make the date column date instead of char
df <- df %>% arrange(symbol, date)                # Order according to symbol then date like in case study lecture

tickers <- unique(df$symbol)

df_with_features <- add_features(df)
head(df_with_features)
