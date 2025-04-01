library(putils)
library(keras3)
library(tensorflow)
library(reticulate)
library(dlm)
library(xts)

source("feature_engineering.R")
source("model_evaluation_functions.R")

use_python("/opt/anaconda3/envs/r-tensorflow/bin/python", required = TRUE)

################################################################################
# using NN to predict stock returns 
################################################################################
# param grid search 

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
    
    params <- training_log[i, c("learning_rate", "num_layers", "hidden_width", "activation", "dropout_rate", "epochs", "batch_size")]
    
    model <- nn_train(x_train, y_train, x_valid, y_valid, params, verbose = 0)
    preds <- predict(model, x_valid, verbose=0)
    
    ic_fold <- compute_ic(df[valid_idx, 'date'], y_valid, preds)
    ic_by_day <- c(ic_by_day, ic_fold)
  }
  training_log$ic[i] <- mean(ic_by_day)
  print(paste("Completed iteration", i, "of", num_param_comb))
  printPercentage(i, num_param_comb)
} # takes ~1 hr to run

training_log <- sort_data_frame(training_log, 'ic', decreasing=T)
print(training_log) 
i_opt <- which.max(training_log$ic) 
print(training_log$ic) 

# 0.043636878 - lower than LGBM

################################################################################
# Fixed architecture of 5 dense layers
################################################################################
# first experimentation (static train/test split, fixed architecture, fixed params)

df <- read.csv("df_train.csv")
df$date <- as.Date(df$date, format = "%Y-%m-%d")
df <- df %>% arrange(symbol, date)

df_with_features <- add_features(df)
df_with_features <- as.data.frame(df_with_features)
df_with_features <- df_with_features %>%
  mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) # replacing NAs with column mean

train_length <- 126
valid_length <- 21

unique_dates <- sort(unique(df$date))
train_idx <- which(df$date >= unique_dates[1] & df$date <= unique_dates[train_length])
valid_idx <- which(df$date >= unique_dates[train_length + 1] & df$date <= unique_dates[train_length + valid_length])

response_vars <- df_with_features %>% dplyr::select(contains("fwd")) %>% colnames()
response_var <- response_vars[1]  # first forward return 

covariate_vars <- setdiff(colnames(df_with_features), c(response_vars, 'date', 'symbol'))

# scaling
X_train <- scale(as.matrix(df_with_features[train_idx, covariate_vars, drop = FALSE]))
X_valid <- scale(as.matrix(df_with_features[valid_idx, covariate_vars, drop = FALSE]), 
                 center = attr(X_train, "scaled:center"), 
                 scale = attr(X_train, "scaled:scale"))

y_train <- df_with_features[train_idx, response_var]
y_valid <- df_with_features[valid_idx, response_var]

first <- layer_dense(units = 128, activation = 'relu', input_shape = c(ncol(X_train)))
second <- layer_dropout(rate = 0.3)
third <- layer_dense(units = 64, activation = 'relu', kernel_regularizer = regularizer_l2(0.01))
fourth <- layer_dropout(rate = 0.3) 
fifth <-   layer_dense(units = 32, activation = 'relu') 
output_layer <- layer_dense(units = 1)

model <- keras_model_sequential()

model %>% 
  first %>% 
  second %>% 
  third %>% 
  fourth %>% 
  fifth %>% 
  output_layer

model$compile(optimizer = optimizer_adam(learning_rate = 0.0005), loss = 'mean_squared_error')

early_stop <- callback_early_stopping(
  monitor = "val_loss", 
  patience = 10, 
  restore_best_weights = TRUE  
)

fit(model, X_train, y_train, epochs=27, batch_size=32, validation_data = list(X_valid, y_valid),  callbacks = list(early_stop) )

y_pred <- predict(model, X_valid)
ic_in_fold <- cor(y_pred, y_valid, method = "spearman")  
print(ic_in_fold) # 0.0020 - lower than LGBM 
