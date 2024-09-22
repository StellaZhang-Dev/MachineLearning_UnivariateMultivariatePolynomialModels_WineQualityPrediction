# Set the working directory
setwd("~/R Project/r_work/SLU Statistics B/WineQualityProject")

# Load necessary packages
required_packages <- c("ggplot2", "dplyr", "GGally", "caret", "randomForest", "glmnet")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages)) install.packages(new_packages)
lapply(required_packages, require, character.only = TRUE)

# Load the data
wine_data <- read.csv("winequality-red.csv", sep = ";")
if (is.null(wine_data) || nrow(wine_data) == 0) {
  stop("Data not loaded correctly or the file is empty.")
}
print("Data loaded successfully")

# Univariate analysis for each variable
print("Univariate analysis for each variable:")
summary(wine_data)

# Multivariate analysis (correlation matrix)
print("Multivariate analysis:")
correlation_matrix <- cor(wine_data)
print(correlation_matrix)

# Train-test split
set.seed(123)
train_index <- createDataPartition(wine_data$quality, p = 0.8, list = FALSE)
train_data <- wine_data[train_index, ]
test_data <- wine_data[-train_index, ]

# Linear Regression model
linear_model <- lm(quality ~ ., data = train_data)
print(summary(linear_model))

# Random Forest model
rf_model <- randomForest(quality ~ ., data = train_data)
print(rf_model)

# Model evaluation
linear_pred <- predict(linear_model, newdata = test_data)
rf_pred <- predict(rf_model, newdata = test_data)

print("Linear model R-squared:")
print(cor(linear_pred, test_data$quality)^2)

print("Random Forest model R-squared:")
print(cor(rf_pred, test_data$quality)^2)

# Save predictions (optional)
write.csv(linear_pred, "linear_predictions.csv")
write.csv(rf_pred, "rf_predictions.csv")
