# Install and load necessary packages
required_packages <- c("ggplot2", "dplyr", "GGally", "glmnet", "randomForest", "e1071", "reshape2")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
suppressMessages(lapply(required_packages, require, character.only = TRUE))

# Step 1: Load the dataset
wine_data <- read.csv("winequality-red.csv", sep = ";")
str(wine_data)
summary(wine_data)

# Step 2: Univariate Analysis for each variable

# Function to create summary statistics, histograms, and boxplots
univariate_analysis <- function(data, var) {
  cat("\nSummary of", var, ":\n")
  print(summary(data[[var]]))
  
  # Histogram
  p1 <- ggplot(data, aes_string(x = var)) +
    geom_histogram(binwidth = 0.5, fill = "blue", color = "black") +
    labs(title = paste("Distribution of", var), x = var, y = "Frequency")
  
  # Boxplot
  p2 <- ggplot(data, aes_string(y = var)) +
    geom_boxplot(fill = "orange") +
    labs(title = paste("Boxplot of", var), y = var)
  
  # Show plots
  print(p1)
  print(p2)
}

# List of variables for univariate analysis
variables <- colnames(wine_data)
for (var in variables) {
  univariate_analysis(wine_data, var)
}

# Step 3: Correlation Matrix
correlation_matrix <- cor(wine_data)
print(correlation_matrix)

# Heatmap of correlation matrix
corr_melt <- reshape2::melt(correlation_matrix)
ggplot(corr_melt, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Heatmap of Feature Correlations")

# Step 4: Pairwise Plots to visualize relationships
ggpairs(wine_data)

# Step 5: Multivariate Analysis - Linear Regression Model
linear_model <- lm(quality ~ ., data = wine_data)
summary(linear_model)

# Step 6: Ridge Regression Model
x <- model.matrix(quality ~ . - 1, data = wine_data)
y <- wine_data$quality
ridge_model <- cv.glmnet(as.matrix(x), y, alpha = 0)
print(paste("Best Lambda for Ridge:", ridge_model$lambda.min))

# Ridge regression predictions
ridge_pred <- predict(ridge_model, s = ridge_model$lambda.min, newx = as.matrix(x))
ridge_r_squared <- cor(ridge_pred, y)^2
print(paste("Ridge Regression R-squared:", ridge_r_squared))

# Step 7: Random Forest Model
rf_model <- randomForest(quality ~ ., data = wine_data, importance = TRUE)
print(rf_model)

# Feature Importance from Random Forest
varImpPlot(rf_model)

# Step 8: SVM Model
svm_model <- svm(quality ~ ., data = wine_data)
summary(svm_model)

# Step 9: Model Comparison
# Predict quality for each model
linear_pred <- predict(linear_model, newdata = wine_data)
ridge_pred <- predict(ridge_model, s = ridge_model$lambda.min, newx = as.matrix(x))
rf_pred <- predict(rf_model, newdata = wine_data)
svm_pred <- predict(svm_model, newdata = wine_data)

# Calculate R-squared values
linear_r_squared <- cor(linear_pred, wine_data$quality)^2
ridge_r_squared <- cor(ridge_pred, wine_data$quality)^2
rf_r_squared <- cor(rf_pred, wine_data$quality)^2
svm_r_squared <- cor(svm_pred, wine_data$quality)^2

cat("\nR-squared Values:\n")
cat("Linear Regression R-squared:", linear_r_squared, "\n")
cat("Ridge Regression R-squared:", ridge_r_squared, "\n")
cat("Random Forest R-squared:", rf_r_squared, "\n")
cat("SVM R-squared:", svm_r_squared, "\n")

# Step 10: Final Conclusion - Analyzing the most important factors
cat("\nFinal Analysis of Important Factors from Random Forest Model:\n")
importance_scores <- importance(rf_model)
importance_df <- as.data.frame(importance_scores)
importance_df <- importance_df[order(importance_df$IncNodePurity, decreasing = TRUE), ]
print(importance_df)

# Visualize the most important factors
ggplot(importance_df, aes(x = reorder(row.names(importance_df), IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Importance of Features for Wine Quality", x = "Feature", y = "Importance (Node Purity)")

# Step 11: Residual Analysis for Linear Model
residuals <- residuals(linear_model)
ggplot(data.frame(residuals), aes(residuals)) +
  geom_histogram(bins = 30, fill = "orange", color = "black") +
  labs(title = "Residuals Distribution", x = "Residuals", y = "Frequency")

ggplot(data.frame(fitted = fitted(linear_model), residuals = residuals), aes(fitted, residuals)) +
  geom_point(color = "blue") +
  geom_smooth(se = FALSE, color = "red") +
  labs(title = "Residuals vs Fitted", x = "Fitted values", y = "Residuals")

# Save results to CSV
write.csv(importance_df, "wine_quality_feature_importance.csv")

