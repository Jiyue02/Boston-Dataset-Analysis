# Boston Dataset Analysis

This repository focuses on the analysis of the Boston dataset using various machine learning models and techniques.

## Objective

The primary goal is to predict the response variable using linear regression and subsequently enhance the model's performance using regularization techniques. The results from these regression models are then compared with results from Support Vector Machine (SVM) and Multilayer Perceptron (MLP) algorithms.

## Workflow

1. **Initial Analysis**: A multiple regression model is built using all predictors from the Boston dataset. This step helps in assessing the relationship of these predictors with the response variable.
2. **Significance Evaluation**: The significance of each predictor is evaluated to determine which of them challenges the null hypothesis.
3. **Regularization Techniques**: The analysis is revisited using three regularization techniques:
   - Lasso Regression
   - Elastic Net Regression
   - Ridge Regression
   The objective is to evaluate the impact of these techniques on the predictive capability of the model.
4. **SVM and MLP Comparison**: Occasionally, regression problems are transformed into classification problems. This is achieved by assigning class labels based on a specific threshold. Both SVM and MLP algorithms are implemented for this purpose, utilizing L2, L1, and Elastic Net regularization.
5. **Cross-Validation**: To ensure the reliability of the results, cross-validation is employed throughout the analysis.

## Dataset

The Boston dataset is selected for this analysis due to its suitability for regression problems. It provides a comprehensive set of predictors that influence the response variable.

## Contributing

Feel free to fork this repository, create a pull request, or raise an issue if you find any discrepancies or have suggestions to improve the analysis.
