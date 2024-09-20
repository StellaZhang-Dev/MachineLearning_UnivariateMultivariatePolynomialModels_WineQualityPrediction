# MachineLearning_AdvancedModeling_WineQualityPrediction

## Project Overview
This project aims to predict the **quality of wine** based on its chemical composition using various **advanced machine learning models**. The primary goal is to develop and evaluate different regression models, including **univariate**, **multivariate**, and **non-linear regression**, to determine which model provides the best performance for predicting wine quality. The dataset used for this project comes from **winequality-red.csv**, which contains a variety of chemical properties and corresponding quality ratings for red wine.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Univariate Linear Regression](#univariate-linear-regression)
4. [Multiple Linear Regression](#multiple-linear-regression)
5. [Non-linear Regression](#non-linear-regression)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Future Work](#future-work)
8. [Conclusion](#conclusion)

## Dataset Overview
The dataset consists of 1599 observations, each representing a sample of red wine. Each observation has 12 attributes, including chemical properties and a quality rating. The target variable is `quality`, which ranges from 0 to 10.

### Attributes:
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality` (target variable)

### Initial Data Summary:
```r
summary(df)
```
The summary provides insights into the distribution of each variable, revealing that most wines have a quality rating between 5 and 7, with alcohol content ranging from 8.4% to 14.9%.

## Data Preprocessing
The following preprocessing steps were applied to prepare the data for modeling:
**Direct file loading using the full path:** We avoid changing the working directory (`setwd()`) and instead load the data using the absolute path.
```r
df <- read_delim("/full/path/to/winequality-red.csv", delim = ";")
```
**Handling missing data:** No missing values were found in the dataset.
**Feature scaling:** Since different variables had different units, **min-max normalization** was applied to ensure all features were on the same scale for model performance:
```r
scaled_df <- as.data.frame(scale(df))
```

## Univariate Linear Regression
Each variable is analyzed independently to observe its effect on the target variable, `quality`.

### Example: Univariate regression for `alcohol`
```r
univariate_model <- lm(quality ~ alcohol, data = df)
summary(univariate_model)
```

### Full univariate regression for all variables:
```r
predictor_vars <- colnames(df)[colnames(df) != "quality"]

for (var in predictor_vars) {
  formula <- as.formula(paste("quality ~", var))
  univariate_model <- lm(formula, data = df)
  cat("\nUnivariate regression for:", var, "\n")
  print(summary(univariate_model))
}
```
This performs a univariate linear regression for each predictor variable, allowing us to assess each variable's impact on wine quality.

## Multiple Linear Regression
In the next step, we apply a multiple linear regression model to include all the predictor variables.

### Model Implementation:
```r
multivariate_model <- lm(quality ~ ., data = df)
summary(multivariate_model)
```
The model includes all the predictor variables, and the summary reveals which variables are statistically significant.

### Key Results:
**Adjusted R-squared:** 0.3561, indicating that 35.61% of the variance in wine quality is explained by the chemical properties.
**Significant variables:** `volatile acidity`, `chlorides`, `sulphates`, `alcohol`, `total sulfur dioxide`, `pH` were found to be significant predictors of wine quality.

## Non-linear Regression
In order to improve model performance, non-linear regression was explored. Specifically, a **polynomial regression** with `alcohol` as the independent variable was applied:
```r
nonlinear_model <- lm(quality ~ poly(alcohol, 2), data = df)
summary(nonlinear_model)
```
### Key Results:
The non-linear model did not significantly improve the fit compared to the linear model (ANOVA p-value = 0.1124).
Therefore, **non-linear regression** was not necessary for this dataset.

## Evaluation Metrics
The performance of the models was evaluated using the following metrics:

**R-squared:** Measures the proportion of variance explained by the model.
**Adjusted R-squared:** Adjusted for the number of predictors.
**Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values.
**P-values:** Determine the significance of each predictor.

## Future Work
**Incorporate more advanced machine learning models:** Models such as **Random Forest**, **Gradient Boosting Machines**, and **Support Vector Machines** could be implemented and compared to the current models.
**Address assumptions violations:** The presence of heteroscedasticity and autocorrelation suggests that models such as **Generalized Linear Models (GLM)** or **Ridge/Lasso regression** could be explored to improve performance.

## Conclusion
This project successfully applied **univariate**, **multivariate**, and non-linear regression techniques to predict wine quality based on its chemical properties. While **alcohol content** and **volatile acidity** were found to be significant predictors, the models were limited by violations of linear regression assumptions. Future work should explore more advanced models to better capture the complex relationships between variables.

