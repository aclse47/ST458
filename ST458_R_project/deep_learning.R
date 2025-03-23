# using NN to predict stock returns 

library(putils)
library(keras3)
library(tensorflow)
library(reticulate)
library(dlm)
library(xts)

source("feature_engineering.R")
source("model_evaluation_functions.R")

use_python("/opt/anaconda3/envs/r-tensorflow/bin/python", required = TRUE)

df <- read.csv("df_train.csv")
df <- add_features(df)  
df <- as.data.frame(df)
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df[is.na(df)] <- 0
df <- df %>% arrange(symbol, date)

response_var <- "simple_returns_fwd_day_1" # prediction target

response_vars <- grep("returns_fwd_day_", names(df), value = TRUE)
covariates <- setdiff(names(df), c(response_vars, "date", "symbol", "year"))

train_idx <- df$date < '2013-01-01'
test_idx <- df$date >= '2013-01-01'

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

# function to train for the parameters we want 
nn_train <- function(x_train, y_train, x_valid, y_valid, params, verbose = 1) {
  bunch(lr, num_layers, width, activation, dropout_rate, epochs, batch_size) %=% params
  
  x_train[!is.finite(x_train)] <- 0
  y_train[!is.finite(y_train)] <- 0
  x_valid[!is.finite(x_valid)] <- 0
  y_valid[!is.finite(y_valid)] <- 0
  
  # array format for keras3
  x_train <- array(as.numeric(x_train), dim = dim(x_train))
  x_valid <- array(as.numeric(x_valid), dim = dim(x_valid))
  y_train <- as.numeric(y_train)
  y_valid <- as.numeric(y_valid)
  
  model <- keras_model_sequential() %>%
    layer_dense(units = width, activation = activation, input_shape = ncol(x_train))
  
  for (i in 1:(num_layers - 1)) {
    model <- model %>%
      layer_dense(units = width, activation = activation) %>%
      layer_dropout(rate = dropout_rate)
  }
  
  model <- model %>% layer_dense(units = 1, activation = 'linear')
  
  model$compile(
    optimizer = optimizer_adam(learning_rate = lr, clipnorm = 1.0),
    loss = 'mean_squared_error'
  )
  
  early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 3, restore_best_weights = TRUE)
  
  fit(model,
    x_train, y_train,
    validation_data = list(x_valid, y_valid),
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(early_stop),
    verbose = verbose
  )
  
  return(model)
}

training_log$activation <- as.character(training_log$activation)

# finding the best param combo:  
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
    
    params <- training_log[i, c("learning_rate", "num_layers", "hidden_width", 
                                "activation", "dropout_rate", "epochs", "batch_size")]
    
    model <- nn_train(x_train, y_train, x_valid, y_valid, params, verbose = 0)
    preds <- predict(model, x_valid, verbose=0)
    
    ic_fold <- compute_ic(df[valid_idx, 'date'], y_valid, preds)
    ic_by_day <- c(ic_by_day, ic_fold)
  }
  training_log$ic[i] <- mean(ic_by_day)
  print(paste("Completed iteration", i, "of", num_param_comb))
  printPercentage(i, num_param_comb)
}

training_log <- sort_data_frame(training_log, 'ic', decreasing=T)
print(training_log)
i_opt <- which.max(training_log$ic) 
print(training_log$ic) 
# now we have these as the best param combos:
#      train_length valid_length lookahead learning_rate num_layers hidden_width activation dropout_rate epochs batch_size            ic
#        252           63        21          0.01          2           10       relu    0.3795131      5         32  0.9550920616
#         252           21        21          0.03          2           10    sigmoid    0.2722721      5         32  0.8891083394
# will train models with those params definined

lookahead <- 21
train_length <- 252
valid_length <- 63
response_var <- sprintf("simple_returns_fwd_day_%d", lookahead)

all_dates <- sort(unique(df$date))
test_dates <- all_dates[all_dates >= as.Date("2013-01-01")]
symbols <- unique(df$symbol)

# matrix output
y_pred <- matrix(NA, nrow = length(test_dates), ncol = length(symbols),
                 dimnames = list(as.character(test_dates), symbols))

model <- NULL
model_trained <- FALSE
scale_mean <- scale_sd <- NULL

for (i in seq_along(test_dates)) {
  date <- test_dates[i]
  
  # retrain model every 'valid_length' days
  if ((i - 1) %% valid_length == 0) {
    date_idx <- match(date, all_dates)
    date_idx_end <- date_idx - lookahead
    date_idx_start <- date_idx_end - train_length + 1
    
    if (date_idx_start < 1 || date_idx_end < 1) next  # not enough history
    
    train_dates <- all_dates[date_idx_start:date_idx_end]
    train_filter <- df$date %in% train_dates
    
    x_train_raw <- as.matrix(df[train_filter, covariates])
    y_train_raw <- df[train_filter, response_var]
    
    # scale data
    x_train <- scale(x_train_raw)
    scale_mean <- attr(x_train, 'scaled:center')
    scale_sd <- attr(x_train, 'scaled:scale')
    
    # train/val split
    n <- nrow(x_train)
    split_point <- floor(0.8 * n)
    x_train_split <- x_train[1:split_point, ]
    y_train_split <- y_train_raw[1:split_point]
    x_valid_split <- x_train[(split_point + 1):n, ]
    y_valid_split <- y_train_raw[(split_point + 1):n]
    
    # train
    model <- nn_train(
      x_train_split, y_train_split,
      x_valid_split, y_valid_split,
      params = list(
        lr = 0.01,
        num_layers = 2,
        width = 10,
        activation = 'relu',
        dropout_rate = 0.3795131,
        epochs = 5,
        batch_size = 32
      ),
      verbose = 0
    )
    
    model_trained <- TRUE
  }
  
  # predict for current day
  if (model_trained) {
    test_filter <- df$date == date
    if (sum(test_filter) == 0) next
    
    x_test <- as.matrix(df[test_filter, covariates])
    x_test <- scale(x_test, scale_mean, scale_sd)
    
    preds <- as.vector(predict(model, x_test, verbose = 0))
    syms <- df$symbol[test_filter]
    
    y_pred[as.character(date), syms] <- preds
  }
  
  printPercentage(date, test_dates)
}
# took 5 mins to run

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

scaling_factor <- pmax(1, rowSums(abs(combined_position)))
valid_rows <- rowSums(!is.na(combined_position)) > 0
combined_position <- combined_position[valid_rows, ]
r_fwd <- r_fwd[valid_rows, ]

daily_pnl <- rowSums(r_fwd * combined_position)
log_wealth <- cumsum(log(1 + daily_pnl))
wealth <- exp(log_wealth)

performance_evaluation_of_wealth(wealth, daily_pnl, 0.03)
actual_returns <- r_fwd  # # SR = 5.355554
ic_test <- cor(as.vector(y_pred), as.vector(actual_returns), method = "spearman", use = "complete.obs")
ic_test 
# 0.08203199

# stops early because strategy stops making predictions since it only makes predictions on days that the model is trained 


# Second Best Param combo:  
#      train_length valid_length lookahead learning_rate num_layers hidden_width activation dropout_rate epochs batch_size            ic
#         252           21        21          0.03          2           10    sigmoid    0.2722721      5         32  0.8891083394


