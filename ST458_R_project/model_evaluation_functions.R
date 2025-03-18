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



# rank features according to importance in predicting future returns
lgbm_features_effects_plot <- function(df_train, covariate_vars, hyperparameters){
  print(hyperparameters)
  bunch(train_length, valid_length, lookahead) %=% hyperparameters[1:3]
  response_var <- sprintf("simple_returns_fwd_day_%01d", hyperparameters$lookahead)
  
  params <- list(
    objective ='regression',
    num_iterations = hyperparameters$num_iterations,
    num_leaves = hyperparameters$num_leaves,
    learning_rate = hyperparameters$learning_rate,
    feature_fraction = hyperparameters$feature_fraction,
    bagging_fraction = hyperparameters$bagging_fraction
  )
   
  dtrain <- lgb.Dataset(data = as.matrix(df_train[, covariate_vars]), label = df_train[, response_var])
  model <- lgb.train(params = params, data = dtrain)
  
  
  importance <- lgb.importance(model)
  for (measure in c('Gain','Cover','Frequency')){
    lgb.plot.importance(importance, top_n=10, measure=measure)
  }
}

# Marginal effect of hyper-parameters on model performance.
lgbm_hyperparameters_marginal_effect_plot <- function(training_log){
  par(mfrow=c(3,3), mar=c(3,2.5,1,1), mgp=c(1.5,0.5,0))
  for (j in 1:9){
    plot(as.factor(training_log[[j]]), training_log$ic,
         xlab=colnames(training_log)[j])
  }
}
