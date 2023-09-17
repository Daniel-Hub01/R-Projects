################################################################################
# COURSE: STATISTICAL LEARNING AND DATA MINING
# CHALLENGE TITLE: REGRESSION CHALLENGE
# TASKS: 
# * To develop two different approaches (parametric & non-parametric) to solve the regression challenge.
# NAME OF AUTHOR (MAIA):
# 1. DANIEL TWENEBOAH ANYIMADU
################################################################################


# Remove all stored global variables
rm(list = ls())


# Installing and loading required dependencies
# install.packages("FNN")    
# install.packages("stats")  
# install.packages("dplyr") 
# install.packages("GGally")
# install.packages("ggcorrplot")
library(readr)          # For reading CSV files
library(FNN)            # For implementing KNN algorithm
library(caret)          # For evaluating model performance
library(tidyverse)      # For data manipulation and visualization
library(stats)          # For statistical functions
library(dplyr)          # For data manipulation
library(GGally)         # For creating visualizations
library(ggplot2)        # For creating plots
library(ggcorrplot)     # For creating correlation plots


################################################################################
# EXPLORATORY DATA ANALYSIS (EDA)
# Read the train CSV file
################################################################################
train_data <- read.csv("C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_2/train_ch.csv")
train_data <- subset(train_data, select = -1) # Drop the X(#1) column 


# Structure and summary statistics of the training data
str(train_data)
summary(train_data)


# Scatter plot for variables(v1:v9) against target(Y)
par(mfrow = c(3, 3))  # Set the layout to 3 rows and 3 columns
for (i in 1:9) {plot(train_data[, paste0("v", i)], train_data$Y, 
                xlab = paste0("v", i), ylab = "Y", 
                main = paste0("Scatter plot: v", i, " vs Y"))}


# Correlation plot
correlation_matrix <- cor(train_data) # Calculate & plot correlation matrix:ggpairs(train_data) ~ Highly collinear(v1~v5; v1~v7; v5~v7; v3~Y)
ggcorrplot(correlation_matrix, type = "lower", hc.order = TRUE,
           lab = TRUE, lab_size = 3, method = "circle",
           title = "Correlation Plot: Variables vs Y")


################################################################################
# TRAIN TEST SPLIT
# Create a validation dataset
################################################################################
set.seed(123)
validation_index <- createDataPartition(train_data$Y, p = 0.80, list = FALSE) # a list of 80% of the rows for training
train_data <- train_data[validation_index,]        # 80% for training
validation <- train_data[-validation_index,]       # 20% for validation


################################################################################
# FEATURE ENGINEERING (PREPROCESSING)
# Apply transformations (polynomial, logarithmic)
################################################################################
train_data <- train_data %>% mutate(v3_sq = v3^2, v3_log = log(v3 + 1)) # v5_v7_interaction = v5 * 7
validation <- validation %>% mutate(v3_sq = v3^2, v3_log = log(v3 + 1))


# Feature Selection
# selected_features <- c("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",  "Y")
selected_features <- c("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v3_sq", "v3_log",  "Y")
train_data <- train_data[, selected_features]
validation <- train_data[, selected_features]


################################################################################
# MACHINE LEARNING (MODEL TRAINING)
# (1) Fitting parametric model [Linear Regression Model] to the training set 
# Fit a linear regression model to the training data
################################################################################
set.seed(123)
lm_model <- lm(Y ~ ., data = train_data, trControl = trainControl(method = 'cv', number = 10))
lm_model
summary(lm_model) # plot(lm_model); attributes(lm_model)

# Plot the linear regression model
# plot(train_data$Y, col = "blue", pch = 16, main = "Linear Regression Model", xlab = "Observations", ylab = "Y")
# abline(lm_model, col = "red", lwd = 2)


# (2) Fitting non-parametric model [KNN] to the training set 
# Tune the KNN algorithm to find the optimal value of k
set.seed(123)
tuneGrid <- expand.grid(k = seq(1, 59, by  = 2))  # Example range of k values to test
knn_model <- train(Y ~ ., data = train_data, method = 'knn',
                   trControl = trainControl(method = 'cv', number = 10),
                   tuneGrid = tuneGrid
)
knn_model
results <- knn_model$results
results |> ggplot(aes(x = k, y = RMSE)) + geom_point() + geom_line()


################################################################################
# PERFORMANCE EVALUATION
# Evaluate lm_model on validation data
################################################################################
lm_preds <- predict(lm_model, validation)
lm_rmse <- sqrt(mean((lm_preds - validation$Y)^2))


# Evaluate knn_model on validation data
knn_preds <- predict(knn_model, validation)
knn_rmse <- sqrt(mean((knn_preds - validation$Y)^2))

# Plot KNN model fitted on the training dataset
# plot(train_data$Y, col = "blue", main = "KNN Model Fitted on Training Dataset", xlab = "Observation", ylab = "Target Variable (Y)")
# points(train_data$Y ~ knn_preds, col = "red")
# legend("topright", legend = c("Actual Y", "KNN Predictions"), col = c("blue", "red"), pch = 1)


# Print RMSE values
cat("Linear Regression Model RMSE on validation data:", lm_rmse, "\n")
cat("KNN Model RMSE on validation data:", knn_rmse, "\n")


# Compare RMSE values
if (lm_rmse < knn_rmse) {
  cat("Linear Regression Model performs better on the validation data.\n")
} else if (knn_rmse < lm_rmse) {
  cat("KNN Model performs better on the validation data.\n")
} else {
  cat("Both models have the same RMSE on the validation data.\n")
}


# Create a data frame to store the RMSE values
rmse_data <- data.frame(Model = c("Linear Regression", "KNN"), RMSE = c(lm_rmse, knn_rmse))
ggplot(rmse_data, aes(x = Model, y = RMSE)) + geom_bar(stat = "identity", fill = "steelblue") +labs(title = "Comparison of RMSE", x = "Model", y = "RMSE") + theme_minimal()





################################################################################
# PREDICTIVE MODEL (TEST DATA)
################################################################################
test_data <- read.csv("C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_2/test_completo_ch.csv")
test_data <- subset(test_data, select = -1) # Drop the X(#1) column

# Feature Engineering
test_data <- test_data %>% mutate(v3_sq = v3^2, v3_log = log(v3 + 1))

# Feature Selection
# selected_features1 <- c("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v3_sq", "v3_log")
test_data <- test_data[, selected_features]

# Predict using the trained models
pred_lm <- predict(lm_model, test_data) 
pred_knn <- predict(knn_model, test_data)

# test_preds <- data.frame(pred_knn, pred_lm)
# test_preds
# write.csv(test_preds, file = "C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_2/0075713_ANYIMADU.csv", row.names = FALSE)

lm_mse_test <- mean((pred_lm - test_data$Y)^2)
knn_mse_test <- mean((pred_knn - test_data$Y)^2)

# Print MSE values
cat("Linear Regression Model MSE on test data:", lm_mse_test, "\n")
cat("KNN Model MSE on test data:", knn_mse_test, "\n")