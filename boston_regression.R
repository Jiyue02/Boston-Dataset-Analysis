library(MASS)
library(glmnet)
library(caret)

lm.fit1 = lm(crim ~ ., data = Boston)
summary(lm.fit1)

# Set the value of k (number of folds)
k <- 5
# Create k-fold cross-validation indices for the dataset
set.seed(123)
folds <- createFolds(Boston$crim, k = k, returnTrain = TRUE)

# Initialize metric variables
rmse_ridge <- rmse_elastic <- rmse_lasso <- rep(0, k)
interpretability_ridge <- interpretability_elastic <- interpretability_lasso <- rep(0, k)
efficiency_ridge <- efficiency_elastic <- efficiency_lasso <- rep(0, k)

# Perform k-fold cross-validation
for (i in 1:k) {
  # Get the training and validation sets
  train_data <- Boston[folds[[i]], ]
  test_data <- Boston[-folds[[i]], ]
  
  # Extract features and labels
  train_x <- as.matrix(train_data[, -1])
  train_y <- train_data$crim
  test_x <- as.matrix(test_data[, -1])
  test_y <- test_data$crim
  
  # Ridge regression (L2)
  cv_model_ridge <- cv.glmnet(train_x, train_y, alpha = 0)
  best_lambda_ridge <- cv_model_ridge$lambda.min
  ridge.mod <- glmnet(train_x, train_y, alpha = 0, lambda = best_lambda_ridge)
  
  # Elastic Net regression (L1 and L2)
  alpha_elastic <- 0.5
  cv_model_elastic <- cv.glmnet(train_x, train_y, alpha = alpha_elastic)
  best_lambda_elastic <- cv_model_elastic$lambda.min
  elastic.mod <- glmnet(train_x, train_y, alpha = alpha_elastic, lambda = best_lambda_elastic)
  
  # Lasso regression (L1)
  cv_model_lasso <- cv.glmnet(train_x, train_y, alpha = 1)
  best_lambda_lasso <- cv_model_lasso$lambda.min
  lasso.mod <- glmnet(train_x, train_y, alpha = 1, lambda = best_lambda_lasso)
  
  # Make predictions on the test set
  ridge.predictions <- predict(ridge.mod, newx = test_x)
  elastic.predictions <- predict(elastic.mod, newx = test_x)
  lasso.predictions <- predict(lasso.mod, newx = test_x)
  
  # Compute performance metric for each model
  rmse_ridge[i] <- sqrt(mean((test_y - ridge.predictions)^2))
  rmse_elastic[i] <- sqrt(mean((test_y - elastic.predictions)^2))
  rmse_lasso[i] <- sqrt(mean((test_y - lasso.predictions)^2))
  
  # Compute interpretability metric for each model
  interpretability_ridge[i] <- sum(coef(ridge.mod, s = best_lambda_ridge) != 0)
  interpretability_elastic[i] <- sum(coef(elastic.mod, s = best_lambda_elastic) != 0)
  interpretability_lasso[i] <- sum(coef(lasso.mod, s = best_lambda_lasso) != 0)
  
  # Compute efficiency metric for each model
  efficiency_ridge[i] <- sum(coef(ridge.mod, s = best_lambda_ridge) != 0) / length(coef(ridge.mod, s = best_lambda_ridge))
  efficiency_elastic[i] <- sum(coef(elastic.mod, s = best_lambda_elastic) != 0) / length(coef(elastic.mod, s = best_lambda_elastic))
  efficiency_lasso[i] <- sum(coef(lasso.mod, s = best_lambda_lasso) != 0) / length(coef(lasso.mod, s = best_lambda_lasso))
}

# Compute average metrics
mean_rmse_ridge <- mean(rmse_ridge)
mean_rmse_elastic <- mean(rmse_elastic)
mean_rmse_lasso <- mean(rmse_lasso)

mean_interpretability_ridge <- mean(interpretability_ridge)
mean_interpretability_elastic <- mean(interpretability_elastic)
mean_interpretability_lasso <- mean(interpretability_lasso)

mean_efficiency_ridge <- mean(efficiency_ridge)
mean_efficiency_elastic <- mean(efficiency_elastic)
mean_efficiency_lasso <- mean(efficiency_lasso)

# Print the results
cat("Mean RMSE - Ridge:", mean_rmse_ridge, "\n")
cat("Mean RMSE - Elastic Net:", mean_rmse_elastic, "\n")
cat("Mean RMSE - Lasso:", mean_rmse_lasso, "\n")


cat("Mean Interpretability - Ridge:", mean_interpretability_ridge, "\n")
cat("Mean Interpretability - Elastic Net:", mean_interpretability_elastic, "\n")
cat("Mean Interpretability - Lasso:", mean_interpretability_lasso, "\n")

cat("Mean Efficiency - Ridge:", mean_efficiency_ridge, "\n")
cat("Mean Efficiency - Elastic Net:", mean_efficiency_elastic, "\n")
cat("Mean Efficiency - Lasso:", mean_efficiency_lasso, "\n")


