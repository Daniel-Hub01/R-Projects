################################################################################
# COURSE: STATISTICAL LEARNING AND DATA MINING
# CHALLENGE TITLE: CLASSIFICATION CHALLENGE ON ALZHEIMER'S DISEASE USING MRIs AND GENE EXPRESSION DATA
# TASKS: 
# * To train/develop models for early diagnosis of Alzheimer's disease.
# * To classify patients into different stages: Controls (CTL), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD).
# NAME OF AUTHOR (MAIA):
# 1. DANIEL TWENEBOAH ANYIMADU
################################################################################


# Remove all stored global variables
rm(list = ls())


# Installing and loading required dependencies
# install.packages("ggvis")
# install.packages("corrplot")
# install.packages("caret")
# install.packages("gmodels")
# install.packages("class")
# install.packages("caTools")
library(ggvis)          # for making scatterplots 
library(corrplot)       # for making correlation plots 
library(caret)          # for normalizing data
library(gmodels)
library(class)
library(caTools)
library(pROC)
library(MASS)


################################################################################
#  EXPLORATORY DATA ANALYSIS (EDA) & FEATURE ENGINEERING
# Read the MCICTL CSV file
################################################################################
MCICTL <- read.csv("C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_1/MCICTL/MCICTLtrain.csv")
MCICTL <- subset(MCICTL, select = -1) # Drop the ID column


# LABEL ENCODING: Converting the target (Label) to (MCI:1 and CTL:0)
MCICTL$Label <- ifelse(MCICTL$Label == "MCI", 1, 0)
MCICTL$Label <- factor(MCICTL$Label)  # encode labels using a two level factor
head(MCICTL)                          # visualize the first 6 rows


################################################################################
# TRAIN TEST SPLIT
# Create a validation dataset
################################################################################
set.seed(1234)                           # to obtain the same sequence of random numbers
validation_index <- createDataPartition(y = MCICTL$Label, p = 0.80, list = FALSE) # a list of 80% of the rows for training
training <- MCICTL[validation_index,]    # 80% for training 
validation <- MCICTL[-validation_index,] # 20% for validation


################################################################################
# FEATURE ENGINEERING: SCALING, DIMENSIONALITY REDUCTION
# Normalization
################################################################################
ss <- preProcess(training[, -ncol(training)], method = "range", range = c(0, 1))
training_norm <- predict(ss, training)
validation_norm <- predict(ss, validation)


# Applying PCA on normalized data
ss_pca <- preProcess(training_norm[, -ncol(training)], method = "pca", pcaComp = 70)
training_pca <- predict(ss_pca, training_norm)
validation_pca <- predict(ss_pca, validation_norm)


# FEATURE SELECTION: Extract the PCA rotation matrix
pca_rot_mat <- ss_pca$rotation
top10_cols <- head(order(-abs(pca_rot_mat[, 1])), 70) # Get the indices of the top 10 columns used by PCA
feat_used <- colnames(training_norm)[top10_cols]      # Print the actual column names of the top 10 columns used by PCA

# RESAMPLING METHOD: Run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"


################################################################################
# MODEL TRAINING
# (1) Fitting nonlinear algorithms [CART] to the training set for prediction
################################################################################
set.seed(7) 
fit.cart <- train(Label~., data = training_pca, method = "rpart", metric = metric, trControl = control)

# (2) Fitting nonlinear algorithms [K-NN] to the training set for prediction 
set.seed(7) 
fit.knn <- train(Label~., data = training_pca, method = "knn", metric = metric, trControl = control)

# (3) Fitting advanced algorithms [SVM] to the training set for prediction 
set.seed(7) 
fit.svm <- train(Label~., data = training_pca, method = "svmRadial", metric = metric, trControl = control)

# (4) Fitting advanced algorithms [Random Forest] to the training set for prediction 
set.seed(7) 
fit.rf <- train(Label~., data = training_pca, method = "rf", metric = metric, trControl = control)

# (5) Fitting linear algorithm [Logistic Regression] to the training set for prediction
set.seed(7) 
fit.lr <- train(Label ~ ., data = training_pca, method = "glm", family = "binomial", metric = metric, trControl = control)

# (6) Fitting linear algorithm [Linear Discriminant Analysis] to the training set for prediction
set.seed(7) 
fit.lda <- train(Label ~ ., data = training_pca, method = "lda", metric = metric, trControl = control)


# Summary of accuracy of models
results <- resamples(list(cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf, lr=fit.lr, lda=fit.lda))
summary(results)
dotplot(results) # Compare accuracy of models


print(fit.svm) # Summary of best model
# Estimate skill of rf on the validation dataset
predictions <- predict(fit.svm, validation_pca)
confusionMatrix(predictions, validation_pca$Label)


################################################################################
# EVALUATION METRICS
# Compute performance metrics on the training and validation dataset
# Define function to compute evaluation metrics
################################################################################
TP <- sum(predictions == 1 & validation_pca$Label == 1)
TN <- sum(predictions == 0 & validation_pca$Label == 0)
FP <- sum(predictions == 1 & validation_pca$Label == 0)
FN <- sum(predictions == 0 & validation_pca$Label == 1)


accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
auc <- roc(validation_pca$Label, as.numeric(predictions))$auc
MCC <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
balanced_accuracy <- (sensitivity + specificity) / 2


# Encoding the target (labels) and feature (predictions) as factors
labels <- factor(as.character(validation_pca$Label), levels = c("0", "1"))
predictions <- factor(as.character(predictions), levels = c("0", "1"))


# Compute ROC and AUC
roc_obj <- roc(labels, as.numeric(predictions))
auc <- auc(roc_obj)


# Create a data frame with the metrics
metrics <- data.frame(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity, Precision = precision, 
                      F1_Score = f1_score, AUC = auc, MCC = MCC, Balanced_Accuracy = balanced_accuracy)


# Print the metrics table
print(metrics)


################################################################################
# PREDICTIVE MODEL (TEST DATA)
################################################################################
MCICTL_test <- read.csv("C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_1/MCICTL/MCICTLtest.csv")      # loading the test dataset
MCICTLtest_wl <- read.csv("C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_1/MCICTL/MCICTLtest_wl.csv") # loading the test dataset with label
test_ids <- MCICTL_test$ID
testwl_ids <- MCICTLtest_wl$ID
ADCTL_test <- subset(MCICTL_test, select = -1)      # Drop the ID column
MCICTLtest_wl <- subset(MCICTLtest_wl, select = -1) # Drop the ID column


# Converting the target (Label) feature to (MCI:1 and CTL:0)
MCICTLtest_wl$Label <- ifelse(MCICTLtest_wl$Label == "MCI", 1, 0)
MCICTLtest_wl$Label <- factor(MCICTLtest_wl$Label)  # encode labels using a two level factor


test_norm <- predict(ss, MCICTLtest_wl) # Normalization using the same scaling parameters as the training dataset
test_pca <- predict(ss_pca, test_norm)  # PCA using the same principal components as the training dataset


# Test set prediction using the best trained model (in this case, rf)
test_preds <- predict(fit.svm, test_pca)
# test_probs <- predict(fit.svm, test_pca, type = "prob")
# colnames(test_probs) <- c("CTL:0", "MCI:1")
# test_preds <- data.frame(test_ids, test_preds, test_probs)
# test_preds

# write.csv(test_preds, file = "C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_1/MCICTL/0075713_ANYIMADU_MCICTLres.csv", row.names = FALSE)
# write.csv(feat_used, file = "C:/Users/dtany/Desktop/MAIA_MASTERS_STUDIES/SEM_2_(ITALY)/Statistical_Learning/Challenge_1/MCICTL/0075713_ANYIMADU_MCICTLfeat.csv", row.names = FALSE)


################################################################################
# EVALUATION METRICS (TEST DATA)
################################################################################
TP_t <- sum(test_preds == 1 & test_pca$Label == 1)
TN_t <- sum(test_preds == 0 & test_pca$Label == 0)
FP_t <- sum(test_preds == 1 & test_pca$Label == 0)
FN_t <- sum(test_preds == 0 & test_pca$Label == 1)

accuracy_t <- (TP_t + TN_t) / (TP_t + TN_t + FP_t + FN_t)
sensitivity_t <- TP_t / (TP_t + FN_t)
specificity_t <- TN_t / (TN_t + FP_t)
precision_t <- TP_t / (TP_t + FP_t)
f1_score_t <- 2 * (precision_t * sensitivity_t) / (precision_t + sensitivity_t)
auc_t <- roc(test_pca$Label, as.numeric(test_preds))$auc
MCC_t <- (TP_t * TN_t - FP_t * FN_t) / sqrt((TP_t + FP_t) * (TP_t + FN_t) * (TN_t + FP_t) * (TN_t + FN_t))
balanced_accuracy_t <- (sensitivity_t + specificity_t) / 2

# Encoding the target (labels) and feature (predictions) as factors
labels_t <- factor(as.character(test_pca$Label), levels = c("0", "1"))
predictions_t <- factor(as.character(test_preds), levels = c("0", "1"))

# Compute ROC and AUC
roc_obj_t <- roc(labels_t, as.numeric(test_preds))
auc_t <- auc(roc_obj_t)

# Create a data frame with the metrics
metrics <- data.frame(Accuracy = accuracy_t, Sensitivity = sensitivity_t, Specificity = specificity_t, Precision = precision_t, 
                      F1_Score = f1_score_t, AUC = auc_t, MCC = MCC_t, Balanced_Accuracy = balanced_accuracy_t)

# Print the metrics table
print(metrics)

# Print the AUC and MCC values
cat("MCICTLtest AUC:", auc_t, "\n")
cat("MCICTLtest MCC:", MCC_t, "\n")
