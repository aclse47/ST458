# using NN to predict stock returns 

library(putils)
library(keras3)
library(tensorflow)
library(reticulate)
library(dlm)
library(xts)


source("training_functions.R")
source("feature_engineering.R")
source("model_evaluation_functions.R")

use_python("/opt/anaconda3/envs/r-tensorflow/bin/python", required = TRUE)

df <- read.csv("df_train.csv")
df <- add_features(df_with_features)
df <- as.data.frame(df_with_features)
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df[is.na(df)] <- 0
range(df$date) 
df <- df %>% arrange(symbol, date)

train_idx <- df$date < '2013-01-01'
test_idx <- df$date >= '2013-01-01'
response_vars <- c('simple_returns_fwd_day_1', 'simple_returns_fwd_day_5', 'simple_returns_fwd_day_21')
covariates <- setdiff(names(df), c(response_vars, 'date', 'symbol', 'year'))
response_var <- "simple_returns_fwd_day_5"

x_train <- as.matrix(df[train_idx, covariates])
y_train <- df[train_idx, response_var]
x_test <- as.matrix(df[test_idx, covariates])
y_test <- as.matrix(df[test_idx, response_var])

x_train <- scale(x_train)
scale_mean <- attr(x_train, 'scaled:center')
scale_sd <- attr(x_train, 'scaled:scale')
x_test <- scale(x_test, scale_mean, scale_sd)

num_param_comb <- 20

param_df <- expand.grid(
  train_length=c(252, 252*4),
  valid_length=c(21,63),
  lookahead=c(1,5,21),
  learning_rate=c(0.01,0.03,0.1), 
  num_layers=c(2,4,6),
  hidden_width=c(10,20,40),
  activation=c('relu', 'sigmoid'),
  dropout_rate=runif(num_param_comb, 0.2, 0.5),
  epochs=c(5),
  batch_size=c(32)
)

set.seed(47)
training_log <- param_df[sample(nrow(param_df), num_param_comb), ]

nn_train <- function(x_train, y_train, params, verbose=1){
  bunch(lr, num_layers, width, activation, dropout_rate, epochs, batch_size) %=% params

  first_hidden <- layer_dense(units = 10, activation = activation,
                              input_shape = ncol(x_train))
  rest_hidden <- replicate(num_layers - 1,
                           layer_dense(units = width, activation = activation))
  output_layer <- layer_dense(units = 1)
  all_layers <- c(first_hidden, rest_hidden, output_layer)
 
  model <- keras_model_sequential()
  
  model <- model %>% 
    layer_dense(units = width, activation = activation, input_shape = ncol(x_train))  # First hidden layer
  
  for (i in 1:num_layers) {
    model <- model %>% 
      layer_dense(units = width, activation = activation) %>%
      layer_dropout(rate = dropout_rate)
  }
  
  model <- model %>% layer_dense(units = 1, activation = 'linear') 
  
  model$compile(optimizer_adam(lr), loss = 'mean_squared_error')
  
  early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 3)
  
  model %>% fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = verbose, validation_data = list(x_valid, y_valid),  # Validation data
                callbacks = list(early_stop))
  
  return(model)
}

training_log$activation <- as.character(training_log$activation)


for (i in 1:num_param_comb){
  print(paste("Running iteration", i, "of", num_param_comb))
  bunch(train_length, valid_length, lookahead) %=% training_log[i, 1:3]
  splits <- time_series_split(df$date, n_splits=10, 
                              train_length, valid_length, lookahead)
  nfolds <- length(splits)
  ic_by_day <- numeric()
  
  for (fold in 1:nfolds){

    bunch(train_idx, valid_idx) %=% splits[[fold]]
    response_var <- sprintf("simple_returns_fwd_day_%d", training_log$lookahead[i])
    
    x_train <- as.matrix(df[train_idx, covariates])
    y_train <- df[train_idx, response_var]
    x_valid <- as.matrix(df[valid_idx, covariates])
    y_valid <- df[valid_idx, response_var]
    
    x_train <- scale(x_train)
    scale_mean <- attr(x_train, 'scaled:center')
    scale_sd <- attr(x_train, 'scaled:scale')
    x_valid <- scale(x_valid, scale_mean, scale_sd)
    
    params <- training_log[i, 4:9]
    
    model <- nn_train(x_train, y_train, params, verbose=0)
    preds <- predict(model, x_valid, verbose=0)
    
    ic_fold <- compute_ic(df[valid_idx, 'date'], y_valid, preds)
    ic_by_day <- c(ic_by_day, ic_fold)
  }
  training_log$ic[i] <- mean(ic_by_day)
  print(paste("Completed iteration", i, "of", num_param_comb))
  printPercentage(i, num_param_comb)
}

i_opt <- which.max(training_log$ic) 
print(training_log$ic)
print(training_log[i_opt, 4:9])

bunch(train_length, valid_length, lookahead) %=% training_log[i_opt, 1:3]
params <- training_log[i_opt, 4:9]
response_var <- sprintf("simple_returns_fwd_day_%d", lookahead)

all_dates <- sort(unique(df$date))
r1fwd <- reshape2::dcast(df, date ~ symbol, value.var='simple_returns_fwd_day_1')
r1fwd <- r1fwd[r1fwd$date >= as.Date('2013-01-01'), ]
row.names(r1fwd) <- as.character(r1fwd$date)
r1fwd <- as.matrix(r1fwd[, -1])
y_pred <- position <- matrix(NA, nrow(r1fwd), ncol(r1fwd), 
                             dimnames=dimnames(r1fwd))

for (i in 1:nrow(y_pred)) {
  date <- rownames(y_pred)[i]
  if (i %% valid_length == 1) { 
    date_idx <- match(date, all_dates)  
    date_idx_end <- date_idx - lookahead  
    date_idx_start <- date_idx_end - train_length + 1  
    date_range <- all_dates[date_idx_start:date_idx_end]
    
    train_filter <- df$date %in% date_range
    
    x_train <- as.matrix(df[train_filter, covariates])
    y_train <- df[train_filter, response_var]
    
    lr <- 0.01
    num_layers <- 2
    hidden_width <- 10
    activation <- 'relu'
    dropout_rate <- 0.3795131
    epochs <- 5
    batch_size <- 32
    
    model <- nn_train(x_train, y_train, 
                      list(lr = lr, num_layers = num_layers, width = width, 
                           activation = activation, dropout_rate = dropout_rate, 
                           epochs = epochs, batch_size = batch_size), verbose = 0)
  }
  
  test_filter <- df$date %in% date
  x_test <- as.matrix(df[test_filter, covariates])
  x_test <- scale(x_test, scale_mean, scale_sd) 
  preds <- predict(model, x_test, verbose=0)
  
  y_pred[date, df$symbol[test_filter]] <- preds
  printPercentage(date, row.names(y_pred))
}

for (i in 1:(nrow(y_pred))){
  rk <- rank(y_pred[i, ], ties.method='random') 
  is_short <- rk <= 5 # short the top 5 (lowest predicted returns)
  is_long <- rk > 45 # long the bottom 5 (highest predicted returns)
  
  position[i, ] <- 0
  position[i, is_short] <- -1/200
  position[i, is_long] <- 1/200
}

combined_position <- position
for (i in 1:nrow(position)){
  j <- max(1, i-20)
  combined_position[i,] <- colSums(position[j:i, , drop=FALSE])
}

max(rowSums(abs(combined_position)))

daily_pnl <- log(1 + rowSums(r1fwd * combined_position))
daily_pnl <- xts(daily_pnl, order.by=as.Date(rownames(y_pred)))
wealth <- exp(cumsum(daily_pnl)) 
plot(wealth)

performance_evaluation_of_wealth(wealth, daily_pnl, 0.03)

# SR = 0.448 




