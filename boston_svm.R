library(e1071)
library(caret)
library(sparseSVM)
library(MASS)

# Create binary class label
threshold <- median(Boston$crim)
Boston$class_label <- ifelse(Boston$crim > threshold, 1, 0)

# Set the value of k (number of folds)
k <- 5

# Create k-fold cross-validation indices for the dataset
set.seed(123)
folds <- createFolds(Boston$class_label, k = k, returnTrain = TRUE)

# Initialize metric variables
accuracy_svm_l2 <- accuracy_svm_l1 <- accuracy_svm_elastic <- precision_svm_l2 <- precision_svm_l1 <- precision_svm_elastic <- recall_svm_l2 <- recall_svm_l1 <- recall_svm_elastic <- f1_svm_l2 <- f1_svm_l1 <- f1_svm_elastic <- rep(0, k)

# Perform k-fold cross-validation
for (i in 1:k) {
  # Get the training and validation sets
  train_data <- Boston[folds[[i]], ]
  test_data <- Boston[-unlist(folds[i]), ]
  
  # Extract features and labels
  train_x <- train_data[, -which(names(train_data) %in% c("crim", "class_label"))]
  train_y <- train_data$class_label
  test_x <- test_data[, -which(names(test_data) %in% c("crim", "class_label"))]
  test_y <- test_data$class_label
  
  # Train the SVM model with L2 regularization
  svm_model_l2 <- svm(train_x, train_y, type = "C-classification", kernel = "linear", cost = 10)
  svm_pred_l2 <- predict(svm_model_l2, newdata = test_x)
  
  
  # Train the SVM model with L1 regularization, lambda = 0.01
  svm_model_l1 <- sparseSVM(as.matrix(train_x), train_y, alpha = 1, lambda = 0.01)
  svm_pred_l1 <- predict(svm_model_l1, X = as.matrix(test_x))
  
  # Train the SVM model with Elastic Net regularization, lambda = 0.01
  svm_model_elastic <- sparseSVM(as.matrix(train_x), train_y, alpha = 0,lambda = 0.01)
  svm_pred_elastic <- predict(svm_model_elastic, X = as.matrix(test_x))
  
  # Compute confusion matrix and related statistics for L2 model
  cm_l2 <- confusionMatrix(factor(svm_pred_l2), factor(test_y))
  
  # Compute confusion matrix and related statistics for L1 model
  cm_l1 <- confusionMatrix(factor(svm_pred_l1), factor(test_y))
  
  # Compute confusion matrix and related statistics for Elastic Net model
  cm_elastic <- confusionMatrix(factor(svm_pred_elastic), factor(test_y))
  
  # Update metric variables for L2 model
  accuracy_svm_l2[i] <- cm_l2$overall["Accuracy"]
  precision_svm_l2[i] <- cm_l2$byClass["Pos Pred Value"]
  recall_svm_l2[i] <- cm_l2$byClass["Sensitivity"]
  f1_svm_l2[i] <- cm_l2$byClass["F1"]
  
  # Update metric variables for L1 model
  accuracy_svm_l1[i] <- cm_l1$overall["Accuracy"]
  precision_svm_l1[i] <- cm_l1$byClass["Pos Pred Value"]
  recall_svm_l1[i] <- cm_l1$byClass["Sensitivity"]
  f1_svm_l1[i] <- cm_l1$byClass["F1"]
  
  # Update metric variables for Elastic Net model
  accuracy_svm_elastic[i] <- cm_elastic$overall["Accuracy"]
  precision_svm_elastic[i] <- cm_elastic$byClass["Pos Pred Value"]
  recall_svm_elastic[i] <- cm_elastic$byClass["Sensitivity"]
  f1_svm_elastic[i] <- cm_elastic$byClass["F1"]
}

# Calculate average metrics for L2 model
avg_accuracy_svm_l2 <- mean(accuracy_svm_l2)
avg_precision_svm_l2 <- mean(precision_svm_l2)
avg_recall_svm_l2 <- mean(recall_svm_l2)
avg_f1_svm_l2 <- mean(f1_svm_l2)

# Calculate average metrics for L1 model
avg_accuracy_svm_l1 <- mean(accuracy_svm_l1)
avg_precision_svm_l1 <- mean(precision_svm_l1)
avg_recall_svm_l1 <- mean(recall_svm_l1)
avg_f1_svm_l1 <- mean(f1_svm_l1)

# Calculate average metrics for Elastic Net model
avg_accuracy_svm_elastic <- mean(accuracy_svm_elastic)
avg_precision_svm_elastic <- mean(precision_svm_elastic)
avg_recall_svm_elastic <- mean(recall_svm_elastic)
avg_f1_svm_elastic <- mean(f1_svm_elastic)

# Print the results for L2 model
cat("SVM with L2 Accuracy:", avg_accuracy_svm_l2, "\n")
cat("SVM with L2 Precision:", avg_precision_svm_l2, "\n")
cat("SVM with L2 Recall:", avg_recall_svm_l2, "\n")
cat("SVM with L2 F1 Score:", avg_f1_svm_l2, "\n")
cat("\n")

# Print the results for L1 model
cat("SVM with L1 Accuracy:", avg_accuracy_svm_l1, "\n")
cat("SVM with L1 Precision:", avg_precision_svm_l1, "\n")
cat("SVM with L1 Recall:", avg_recall_svm_l1, "\n")
cat("SVM with L1 F1 Score:", avg_f1_svm_l1, "\n")
cat("\n")

# Print the results for Elastic Net model
cat("SVM with Elastic Net Accuracy:", avg_accuracy_svm_elastic, "\n")
cat("SVM with Elastic Net Precision:", avg_precision_svm_elastic, "\n")
cat("SVM with Elastic Net Recall:", avg_recall_svm_elastic, "\n")
cat("SVM with Elastic Net F1 Score:", avg_f1_svm_elastic, "\n")
