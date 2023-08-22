library(MASS)
library(h2o)
library(caret)

# Initialize H2O instance
h2o.init()

# Convert the dataset to an H2O frame
Boston_h2o <- as.h2o(Boston)

# Set the value of k (number of folds)
k <- 5

# Set a seed for reproducibility
set.seed(123)

# Create k-fold cross-validation indices for the dataset
folds <- createFolds(Boston$crim, k = k, returnTrain = TRUE)

# Initialize metric variables
rmse_mlp_l1 <- rmse_mlp_l2 <- rmse_mlp_elastic <- rep(0, k)

# Perform k-fold cross-validation
for (i in 1:k) {
  # Get the training and validation sets
  train_data <- Boston_h2o[folds[[i]], ]
  test_data <- Boston_h2o[-folds[[i]], ]
  
  # MLP with L1 regularization
  model_mlp_l1 <- h2o.deeplearning(
    x = 2:ncol(train_data),
    y = 1,
    training_frame = train_data,
    l1 = 0.01,
    hidden = c(10,10),
    epochs = 50,
    seed = 123  
  )
  
  # MLP with L2 regularization,𝛂 = 0.01
  model_mlp_l2 <- h2o.deeplearning(
    x = 2:ncol(train_data),
    y = 1,
    training_frame = train_data,
    l2 = 0.01,  
    hidden = c(10,10),
    epochs = 50,
    seed = 123
  )
  
  # MLP with Elastic Net regularization,𝛂 = 0.01
  model_mlp_elastic <- h2o.deeplearning(
    x = 2:ncol(train_data),
    y = 1,
    training_frame = train_data,
    l1 = 0.01,  
    l2 = 0.01, 
    hidden = c(10,10),
    epochs = 50,
    seed = 123
  )
  
  # Make predictions on the test set
  mlp_l1_predictions <- h2o.predict(model_mlp_l1, test_data)
  mlp_l2_predictions <- h2o.predict(model_mlp_l2, test_data)
  mlp_elastic_predictions <- h2o.predict(model_mlp_elastic, test_data)
  
  # Compute RMSE for each model
  rmse_mlp_l1[i] <- sqrt(mean((as.vector(test_data$crim) - as.vector(mlp_l1_predictions$predict))^2))
  rmse_mlp_l2[i] <- sqrt(mean((as.vector(test_data$crim) - as.vector(mlp_l2_predictions$predict))^2))
  rmse_mlp_elastic[i] <- sqrt(mean((as.vector(test_data$crim) - as.vector(mlp_elastic_predictions$predict))^2))
  
}

# Compute average metrics
mean_rmse_mlp_l1 <- mean(rmse_mlp_l1)
mean_rmse_mlp_l2 <- mean(rmse_mlp_l2)
mean_rmse_mlp_elastic <- mean(rmse_mlp_elastic)

# Print the results
cat("Mean RMSE - MLP L1:", mean_rmse_mlp_l1, "\n")
cat("Mean RMSE - MLP L2:", mean_rmse_mlp_l2, "\n")
cat("Mean RMSE - MLP Elastic Net:", mean_rmse_mlp_elastic, "\n")

