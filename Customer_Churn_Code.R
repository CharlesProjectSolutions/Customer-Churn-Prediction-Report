
# Load All Necessary Libraries ---

library(tidyverse)  # Data manipulation and visualization with ggplot2
library(lubridate)  # Date Parsing
library(rsample)    # Train/Test split
library(recipes)    # pre-processing pipeline (one-hot, scaling, etc.)
library(yardstick)  # metrics (ROC-AUC, PR-AUC, etc.)
library(parsnip)    # Model definitions
library(workflows)  # Combine recipe + model
library(ranger)     # fast Random Forest
library(kknn)       # KNN engine
library(glmnet)     # (optional) logistic via glmnet
library(pROC)       # ROC tools - Model Evaluation
library(PRROC)      # PR AUC tools - Model Evaluation 
library(ggplot2)    # plotting
library(patchwork)  # Layout Multiple-plots
library(caret)      # Machine learning framework
library(corrplot)   # Correlation plots
library(gridExtra)  # Multiple plot arrangements
library(ROSE)       # Handling imbalanced data
library(car)        # VIF calculation for multicollinearity
library(reshape2)   # Data reshaping
library(ROCR)         # ROC analysis
library(caTools)      # Data splitting
# Specific Model Packages (some are dependencies of caret, but good to be explicit)
library(gbm)        # Gradient Boosting
library(randomForest) # Random forest
library(xgboost)      # XGBoost
library(nnet)       # For simple neural networks
library(knitr)     # For tables
library(dplyr)
library(BSDA)


# Set seed for reproducibility
# Why: Ensures our results can be replicated
set.seed(123)


# =============================================================================
# STEP 2: DATA LOADING AND INITIAL EXPLORATION
# Why: Understanding the data structure is the foundation of any ML project
# =============================================================================


# Read the data

transaction_history   <- read_csv(file.choose(), show_col_types = FALSE)
customer_service      <- read_csv(file.choose(), show_col_types = FALSE)
customer_activity     <- read_csv(file.choose(), show_col_types = FALSE)
customer_demographics <- read_csv(file.choose(), show_col_types = FALSE)
churn                 <- read_csv(file.choose(), show_col_types = FALSE)


# =============================================================================
# STEP 3: INITIAL INSPECTION & DATA QUALITY CHECK
# Why: Identify missing values, data types, and potential issues etc.
# =============================================================================
# 

# Inspection  on Transaction History

# Dataset Dimension
cat("Rows:", nrow(transaction_history), "\n")
cat("Columns:", ncol(transaction_history), "\n")

# Variable Names & Data Types
sapply(transaction_history, class) # Covert CustomerID, TransactionID, TransactionDate & ProductCategory to its proper data type

# First 10 rows of transaction_history Data
head(transaction_history, 10)

# transaction_history Data Frame Structure
str(transaction_history)

# Summary Statistics
print(summary(transaction_history))

# Check for missing values
missing_counts <- colSums(is.na(transaction_history))
print(missing_counts)

# Check for Duplicates or duplicate values
sum(duplicated(transaction_history))   # Number of duplicate rows
transaction_history[duplicated(transaction_history), ]  # View duplicate rows

# Check unique values for categorical variables
cat("ProductCategory:", paste(unique(transaction_history$ProductCategory), collapse=", "), "\n")



# Inspection on Customer Service Data

# Dataset Dimension
cat("Rows:", nrow(customer_service), "\n")
cat("Columns:", ncol(customer_service), "\n")

# Variable Names & Data Types
sapply(customer_service, class)

# First 10 rows of customer_service Data
head(customer_service, 10)

# customer_service Data Frame Structure
str(customer_service)

# Summary Statistics
print(summary(customer_service))

# Check for missing values
missing_counts <- colSums(is.na(customer_service))
print(missing_counts)

# Check for Duplicates or duplicate values
sum(duplicated(customer_service))   # Number of duplicate rows
customer_service[duplicated(customer_service), ]  # View duplicate rows

# Check unique values for categorical variables
cat("ResolutionStatus:", paste(unique(customer_service$ResolutionStatus), collapse=", "), "\n")



# Inspection on Customer Activity Data

# Dataset Dimensions
cat("Number of Rows:", nrow(customer_activity), "\n")
cat("Number of Columns:", ncol(customer_activity), "\n")

# Variable Names & Data Types
sapply(customer_activity, class)

# First 10 rows of customer_activity Data
head(customer_activity, 10)

# customer_activity Data Frame Structure
str(customer_activity)

# Summary Statistics
print(summary(customer_activity))


# Check for missing values
missing_counts <- colSums(is.na(customer_activity))
print(missing_counts)

# Check unique values for categorical variables
cat("LoginFrequency:", paste(unique(customer_activity$LoginFrequency), collapse=", "), "\n")
cat("ServiceUsage:", paste(unique(customer_activity$ServiceUsage), collapse=", "), "\n")

# Check for Duplicates or duplicate values
sum(duplicated(customer_activity))   # Number of duplicate rows
customer_activity[duplicated(customer_activity), ]  # View duplicate rows



# Inspection on Customer Demographics Data

# Customer Demographics Table Dimensions
cat("Number of Rows:", nrow(customer_demographics), "\n")
cat("Number of Columns:", ncol(customer_demographics), "\n")

# Variable Names & Data Types
sapply(customer_demographics, class)

# First 10 rows of customer_demographics Data
head(customer_demographics, 10)

# customer_demographics Data Frame Structure
str(customer_demographics)

# Summary Statistics
print(summary(customer_demographics))


# Check for missing values
missing_counts <- colSums(is.na(customer_demographics))
print(missing_counts)

# Check unique values for categorical variables
cat("Gender:", paste(unique(customer_demographics$Gender), collapse=", "), "\n")
cat("MaritalStatus:", paste(unique(customer_demographics$MaritalStatus), collapse=", "), "\n")
cat("IncomeLevel:", paste(unique(customer_demographics$IncomeLevel), collapse=", "), "\n")



# Check for Duplicates or duplicate values
sum(duplicated(customer_demographics))   # Number of duplicate rows
customer_demographics[duplicated(customer_demographics), ]  # View duplicate rows



# Inspection on churn Data

cat("Number of Rows:", nrow(churn), "\n")
cat("Number of Columns:", ncol(churn), "\n")

# Variable Names & Data Types
sapply(churn, class)

# First 10 rows of churn Data
head(churn, 10)

# churn Data Frame Structure
str(churn)

# Summary Statistics
print(summary(churn))


# Check for missing values
missing_counts <- colSums(is.na(churn)) # Count missing per column
mean(is.na(churn$ChurnStatus)) # % missing in the column "Churn Status"
print(missing_counts)


# Check unique values for categorical variables
cat("ChurnStatus:", paste(unique(churn$ChurnStatus), collapse=", "), "\n")

# Check for Duplicates or duplicate values
sum(duplicated(churn))   # Number of duplicate rows
churn[duplicated(churn), ]  # View duplicate rows


# Summarizing Response Variable (distribution)
class_counts <- table(churn$ChurnStatus)
class_props <- prop.table(class_counts)
minority_class <- names(which.min(class_counts))
minority_pct <- min(class_props) * 100
imbalance_ratio <- max(class_counts) / min(class_counts)


cat("Current Dataset Distribution:\n")
print(class_counts)
cat("\nProportions:\n")
print(round(class_props, 4))

cat("\nKey Metrics:\n")
cat("- Minority class:", minority_class, "\n")
cat("- Minority percentage:", round(minority_pct, 2), "%\n")
cat("- Imbalance ratio:", round(imbalance_ratio, 2), ":1\n")


# Determine current dataset's imbalance category
if(minority_pct >= 40) {
  severity <- "BALANCED"
  color <- "green"
} else if(minority_pct >= 25) {
  severity <- "MILD IMBALANCE"
  color <- "yellow"
} else if(minority_pct >= 10) {
  severity <- "MODERATE IMBALANCE"
  color <- "orange"
} else if(minority_pct >= 5) {
  severity <- "SEVERE IMBALANCE"
  color <- "red"
} else {
  severity <- "EXTREME IMBALANCE"
  color <- "darkred"
}


cat("YOUR DATASET STATUS:", severity, "\n")
cat("Minority class percentage:", round(minority_pct, 2), "%\n")
cat("Imbalance ratio:", round(imbalance_ratio, 2), ":1\n")


# Dataset is imbalanced (more non-churners), around 20% churn rate.
# Which will require balancing strategies (SMOTE, class weights, etc.) in modeling.
# WHAT IS CLASS IMBALANCE? Class imbalance occurs when one class (majority) significantly outnumbers the class (minority)
# in your dataset. 




# ========================================================================
#                        STEP 4: EXPLORATORY DATA ANALYSIS (EDA)

# What: Analyzing distributions, relationships, and patterns
# Why: Understanding the data helps us make informed modeling decisions
# ========================================================================


counts <- table(churn$ChurnStatus)


# Plot 1: Histogram of MonthlyCharges
par(mfrow=c(3,2), mar=c(4,4,2,1))

# Churn Distribution
barplot(counts, main = "Churn Distribution", xlab = "Churn Status (0=No, 1=Yes)", ylab = "Count", col = c("seagreen", "coral"))

# Age Distribution of Customers
hist(customer_demographics$Age, breaks = 20, # Controls the number of bins main = "Age Distribution of Customers", xlab = "Age", 
     ylab = "Frequency", col = "lightblue", border = "white", freq = TRUE) # Ensures y-axis is frequency


# Data Preparation (R equivalent of the merge)
merged_demo_churn <- customer_demographics %>% inner_join(churn, by = "CustomerID")

counts_table <- table(merged_demo_churn$IncomeLevel, merged_demo_churn$ChurnStatus)
counts_table_transposed <- t(counts_table)

# Churn by Income Level
barplot(counts_table_transposed,
        main = "Churn by Income Level",
        xlab = "Income Level",
        ylab = "Count",
        beside = TRUE,
        col = c("seagreen", "coral"),
        names.arg = colnames(counts_table_transposed),
        # Increase the x-axis limit slightly to make room for the legend
        xlim = c(0, ncol(counts_table_transposed) * 4), # Adjusted for wider plot area
        legend.text = rownames(counts_table_transposed),
        # Position the legend to the right of the plot area
        args.legend = list(title = "Churn Status", x = "right", inset = 0.05)) # 'inset' helps positioning



# Total Amount Spent vs Churn
spend_per_customer <- transaction_history %>% group_by(CustomerID) %>% summarise(AmountSpent = sum(AmountSpent))

spend_churn <- spend_per_customer %>% inner_join(churn, by = "CustomerID")

# Use the formula interface: AmountSpent ~ ChurnStatus
boxplot(AmountSpent ~ ChurnStatus, data = spend_churn, main = "Total Amount Spent vs Churn", xlab = "Churn Status", 
        ylab = "Total Amount Spent", col = c("seagreen", "coral"), names = c("0", "1"))

# Login Frequency vs Churn (Box Plot)

# Data Preparation (R equivalent of the merge)
activity_churn <- customer_activity %>% inner_join(churn, by = "CustomerID")


# Base R
# Use the formula interface: LoginFrequency ~ ChurnStatus
boxplot(LoginFrequency ~ ChurnStatus, data = activity_churn, main = "Login Frequency vs Churn", xlab = "Churn Status", 
        ylab = "Login Frequency", col = c("seagreen", "coral"), names = c("0", "1"))



# Data Preparation (R equivalent of the merge)
service_churn <- customer_service %>% inner_join(churn, by = "CustomerID")

# Base R
# Requires calculating the table of counts first
counts_table_service <- table(service_churn$ResolutionStatus, service_churn$ChurnStatus)
count_table_transposed <- t(counts_table_service)

barplot(count_table_transposed, main = "Churn by Customer Service Resolution", xlab = "Resolution Status",
        ylab = "Count",
        beside = TRUE, # Makes bars side-by-side (grouped)
        col = c("seagreen", "coral"), legend.text = rownames(counts_table_service), 
        args.legend = list(title = "Churn Status", x = "topright"))


table(service_churn$ResolutionStatus, service_churn$ChurnStatus)
table(activity_churn$ServiceUsage, activity_churn$ChurnStatus)
table(merged_demo_churn$IncomeLevel, merged_demo_churn$ChurnStatus)
favstats <- favstats(LoginFrequency ~ ChurnStatus, data = activity_churn)
favstats


favstats <- favstats(spend_churn$AmountSpent ~ spend_churn$ChurnStatus, data = spend_churn)
view(favstats)



max(customer_demographics$Age)
min(customer_demographics$Age)




# ========================================================================
# STEP 5: DATA CLEANING AND PREPROCESSING
# What: Handling missing values, data types, and column names
# Why: Clean data is essential for accurate modeling
# ========================================================================



# Ensure date columns are parsed consistently & Correctly (coerce invalids to NA)
transaction_history <- transaction_history %>% mutate(TransactionDate = as.Date(TransactionDate, format = "%m/%d/%Y"))
customer_service <- customer_service %>% mutate(InteractionDate = as.Date(InteractionDate))
customer_activity <- customer_activity %>% mutate(LastLoginDate = as.Date(LastLoginDate))




class(transaction_history$TransactionDate)
class(transaction_history$TransactionDate)
class(customer_service$InteractionDate)




# ============================================================================
# STEP 5: Per-Customer Feature Engineering (No Leakage)
# ============================================================================


# Transactions: aggregate spend, frequency, product count
tfeat <- transaction_history %>% group_by(CustomerID) %>%
  summarise(TotalSpend = sum(AmountSpent, na.rm = TRUE), AvgSpend = mean(AmountSpent, na.rm = TRUE), 
            NumTransactions = n(), # count rows per customer
            UniqueProductCategories = n_distinct(ProductCategory)) %>% ungroup()


head(tfeat, 6)

transaction_history %>% filter(CustomerID == 2)

tfeat %>% filter(CustomerID == 2)

# Service: interaction volume and outcomes
sfeat <- customer_service %>% group_by(CustomerID) %>%
  summarise(NumInteractions = n(), NumResolved = sum(ResolutionStatus == "Resolved", na.rm = TRUE), 
            NumUnresolved = sum(ResolutionStatus == "Unresolved", na.rm = TRUE)) %>% 
  mutate(ResolutionRate = if_else(NumInteractions > 0, NumResolved / NumInteractions, 0)) %>% ungroup()

head(customer_service)
head(sfeat, 5)


# Activity: recent and engagement
max_login <- max(customer_activity$LastLoginDate, na.rm = TRUE)
max_login

afeat <- customer_activity %>% mutate(DaysSinceLastLogin = as.numeric(max_login - LastLoginDate)) %>% 
  select(CustomerID, LoginFrequency, ServiceUsage, DaysSinceLastLogin) %>% ungroup()

head(afeat)
str(afeat)



# ============================================================================
# STEP 3: Merge to a master customer table
# ============================================================================


master <- customer_demographics %>%
  left_join(tfeat, by = "CustomerID") %>%
  left_join(sfeat, by = "CustomerID") %>%
  left_join(afeat, by = "CustomerID") %>%
  left_join(churn, by = "CustomerID")

head(master)
str(master)
master <- as.data.frame(master)

# Checking for Missing Values
colSums(is.na(master)) # Count missing per column


# Fill engineered NAs with neutral values (just to keeps the rows usable)
master <- master %>% 
  mutate(TotalSpend = replace_na(TotalSpend, 0), AvgSpend = replace_na(AvgSpend, 0), 
         NumTransactions = replace_na(NumTransactions, 0), 
         UniqueProductCategories = replace_na(UniqueProductCategories, 0), NumInteractions = replace_na(NumInteractions, 0), 
         NumResolved = replace_na(NumResolved, 0), NumUnresolved = replace_na(NumUnresolved, 0), 
         ResolutionRate = replace_na(ResolutionRate, 0), LoginFrequency = replace_na(LoginFrequency, 0), 
         DaysSinceLastLogin = replace_na(DaysSinceLastLogin, max(DaysSinceLastLogin, na.rm = TRUE))) %>%
  
  # Making sure the target is a factor with levels "0","1" (yardstick expectations)
  mutate(ChurnStatus = factor(ChurnStatus, levels = c(0,1), labels = c("0","1")))

# -----------------------------
head(master)
str(master) 

colnames(master)


# Checking for Missing Values
colSums(is.na(master)) # Count missing per column

#   write_csv(master, "C:/Users/Charles/Downloads/R Project/Customer_Churn_Data.csv")








# DATA LOADING AND INITIAL EXPLORATION

df <- read_csv(file.choose(), show_col_types = FALSE)

# First 10 rows of the df
head(df, 10)

# Last 10 rows of the df
tail(df, 10)

# Looking at variable names & data types
sapply(df, class)

# Checking for Missing Values
colSums(is.na(df)) # Count missing per column

# Check for Duplicates or duplicate values
sum(duplicated(df))   # Number of duplicate rows


# Convert all categorical variables to factors for classification
df <- df %>% mutate(Gender = as.factor(Gender), MaritalStatus = as.factor(MaritalStatus), IncomeLevel = as.factor(IncomeLevel), 
                    ServiceUsage = as.factor(ServiceUsage), 
                    ChurnStatus = factor(ChurnStatus, levels = c(0, 1), labels = c("No", "Yes")))



# ==============================================================================
#                                FEATURE ENGINEERING
# ==============================================================================
# Creating domain-specific features captures business logic and non-linear relationships that improve model performance.


# # 1. Customer Value Score - combines spending and engagement: WHY: High-value customers deserve different retention strategies
# 2. Engagement Score - balances activity frequency with recency: WHY: Recent and frequent engagement indicates loyalty
# 3. Support Efficiency - quality of support experience: WHY: Poor support experiences often lead to churn
# 4. Has Unresolved Issues - binary flag for pending problems: WHY: Unresolved issues are strong churn predictors
# 5. Transaction Frequency - transactions per active day: WHY: Measures intensity of platform usage
# 6. Product Diversity Score - variety in product usage: WHY: Customers using multiple products are stickier
# 7. Activity Level - categorical recency segmentation: WHY: Different interventions for different activity levels
# 8. Customer Lifetime Value Proxy - estimated long-term value: WHY: Helps prioritize retention efforts
# 9. Risk Score - preliminary risk assessment: WHY: Quick identification of at-risk customers


df_eng <- df %>% mutate(CustomerValueScore = TotalSpend * 0.4 + NumTransactions * 50 * 0.3 + UniqueProductCategories * 100 * 0.3, 
                        EngagementScore = scale(LoginFrequency) + scale(NumTransactions), 
                        SupportEfficiency = ifelse(NumInteractions > 0, ResolutionRate, 1), 
                        HasUnresolvedIssues = as.numeric(NumUnresolved > 0), 
                        TransactionFrequency = NumTransactions / (DaysSinceLastLogin + 1),
                        ProductDiversityScore = UniqueProductCategories / (NumTransactions + 1), 
                        ActivityLevel = cut(DaysSinceLastLogin, breaks = c(-1, 30, 90, 180, Inf), labels = c("Very_Active", 
                                                                                                             "Active", "Moderate", 
                                                                                                             "Inactive")),
                        AgeGroup = cut(Age, breaks = c(17, 24, 34, 44, 54, 64, Inf), labels = c("18-24", "25-34", "35-44", 
                                                                                                "45-54", "55-64", "65+")),
                        AvgTransactionsPerLogin = ifelse(LoginFrequency > 0, NumTransactions / LoginFrequency, 0),
                        IsHighlyInactive = factor(ifelse(DaysSinceLastLogin > 90, "Yes", "No"))
                         )



# Binary flag for inactivity


# Ratio feature for engagement efficiency



# List new features
new_features <- c("CustomerValueScore", "EngagementScore", "SupportEfficiency", "HasUnresolvedIssues", "TransactionFrequency", 
                  "ProductDiversityScore", "ActivityLevel", "CLV_Proxy", "AgeGroup", "AvgTransactionsPerLogin", "IsHighlyInactive")

cat(sprintf("✓ Created %d new engineered features:\n", length(new_features)))
for (feat in new_features) {
  cat(sprintf("  • %s\n", feat))
  }

head(df)
head(df_eng)
sapply(df_eng, class)



# ==============================================================================
#                                Multicollinearity Handling
# ==============================================================================
# Multicollinearity can destabilize models and make interpretation difficult. 
# We check for highly correlated predictors to avoid redundancy. 
# TotalSpend, NumTransactions, etc. are highly correlated (r > 0.7). We will remove these highly correlated features
# Removing highly correlated features improves model performance and interpretability.

df_final <- df_eng %>% mutate(CustomerID = as.character(CustomerID))

# Select only numeric features for correlation analysis
numeric_features <- df_final %>% select(where(is.numeric))

# Calculate correlation matrix
correlation_matrix <- cor(numeric_features)

# Visualize correlation matrix
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", tl.cex = 0.8, tl.col = "black")


# Find highly correlated features (absolute correlation > 0.8)
high_corr <- findCorrelation(correlation_matrix, cutoff = 0.8)
cat("\nHighly correlated features to remove:\n")
print(names(numeric_features)[high_corr])

# Side by side Correlation Matrix for All Numerical Columns
numerical_df <- df_final[sapply(df_final, is.numeric)]

# Calculate the correlation matrix
cor_matrix <- cor(numerical_df, use = "pairwise.complete.obs")

# Print the matrix
print(cor_matrix)

# Remove highly correlated features
if(length(high_corr) > 0) {df_final <- df_final %>% select(-all_of(names(numeric_features)[high_corr]))}

# Remove redundant, identifier, and original columns used for engineering
df_final <- df_final %>% select(-CustomerID, -DaysSinceLastLogin, -Age)


# 2. Model Training Pipeline

## A stratified 80/20 split is performed to ensure the proportion of churners is the same in both the training and testing sets.

set.seed(42) # for reproducibility
train_index <- createDataPartition(df_final$ChurnStatus, p = 0.8, list = FALSE)
train_data <- df_final[train_index, ]
test_data <- df_final[-train_index, ]

# Verify proportions
prop.table(table(train_data$ChurnStatus))
prop.table(table(test_data$ChurnStatus))


## Preprocessing Recipe & Class Imbalance
# We define a preprocessing recipe using `caret`. This includes one-hot encoding for categorical variables and **SMOTE** 
# for handling
# class imbalance directly within the cross-validation process.


# Define the training control with 10-fold Cross-Validation
# We use twoClassSummary to get ROC, Sens (Recall), and Spec (Specificity)
# SMOTE is applied during resampling to prevent data leakage

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary, 
                     sampling = "smote", # Apply SMOTE 
                     verboseIter = FALSE)



## Train 5 Algorithms
# Training five candidate models using the same cross-validation and preprocessing methodology. We optimize for the ROC metric.


# For reproducibility
set.seed(42)

# 1. Logistic Regression
model_lr <- train(ChurnStatus ~ ., data = train_data, 
                  method = "glm", family = "binomial", trControl = ctrl, 
                  metric = "ROC", preProcess = c("center", "scale")
                  )

# 2. Random Forest
model_rf <- train(ChurnStatus ~ ., data = train_data, 
                  method = "rf", trControl = ctrl, metric = "ROC", 
                  preProcess = c("center", "scale"), tuneGrid = expand.grid(.mtry = c(5, 10, 15)) # Tune mtry
                  )

# 3. Gradient Boosting Machine (GBM)
model_gbm <- train(ChurnStatus ~ ., data = train_data, 
                   method = "gbm", trControl = ctrl, 
                   metric = "ROC", preProcess = c("center", "scale"), verbose = FALSE
                   )

# 4. XGBoost
model_xgb <- train(ChurnStatus ~ ., data = train_data, 
                   method = "xgbTree", trControl = ctrl, 
                   metric = "ROC", preProcess = c("center", "scale")
                   )


# 5. Neural Network (Single-layer)
model_nnet <- train(ChurnStatus ~ ., data = train_data, 
                    method = "nnet", trControl = ctrl, 
                    metric = "ROC", preProcess = c("center", "scale"), trace = FALSE
                    )



# Compare model performance based on CV results
results <- resamples(list(LogisticRegression = model_lr, 
                          RandomForest = model_rf, 
                          GBM = model_gbm, 
                          XGBoost = model_xgb, 
                          NeuralNet = model_nnet
                          ))

summary(results)
dotplot(results)



# 3. Model Evaluation on Test Set

#         These models are evaluated on the unseen test data to assess their real-world performance.

## Performance Metrics Table: We calculate Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, and PR-AUC for each model.


# Function to get all metrics

get_metrics <- function(model, test_data) {
  predictions_class <- predict(model, newdata = test_data)
  predictions_prob <- predict(model, newdata = test_data, type = "prob")$Yes
  
  cm <- confusionMatrix(predictions_class, test_data$ChurnStatus, positive = "Yes")
  
  roc_obj <- roc(test_data$ChurnStatus, predictions_prob)
  pr_obj <- pr.curve(scores.class0 = predictions_prob[test_data$ChurnStatus == "Yes"],
                     scores.class1 = predictions_prob[test_data$ChurnStatus == "No"],
                     curve = FALSE)
  
  return(c(
    cm$overall["Accuracy"],
    cm$byClass["Precision"],
    cm$byClass["Recall"],
    cm$byClass["Specificity"],
    cm$byClass["F1"],
    ROC_AUC = auc(roc_obj),
    PR_AUC = pr_obj$auc.integral
  ))
}

# Get metrics for all models
metrics_lr <- get_metrics(model_lr, test_data)
metrics_rf <- get_metrics(model_rf, test_data)
metrics_gbm <- get_metrics(model_gbm, test_data)
metrics_xgb <- get_metrics(model_xgb, test_data)
metrics_nnet <- get_metrics(model_nnet, test_data)



# Combine into a data frame
metrics_summary <- data.frame(Model = c("Logistic Regression", "Random Forest", "GBM", "XGBoost", "Neural Network"), 
                              rbind(metrics_lr, metrics_rf, metrics_gbm, metrics_xgb, metrics_nnet)
                              )


# Print as a nice table
kable(metrics_summary, digits = 3, caption = "Model Performance Comparison on Test Set")

# Conclusion: The Neural Network model is the champion, showing the best overall performance, especially in Recall (Sensitivity) 
# and ROC_AUC scores.



## ROC and PR Curves

# Get probabilities for all models
probs_lr <- predict(model_lr, test_data, type = "prob")$Yes
probs_rf <- predict(model_rf, test_data, type = "prob")$Yes
probs_gbm <- predict(model_gbm, test_data, type = "prob")$Yes
probs_xgb <- predict(model_xgb, test_data, type = "prob")$Yes
probs_nnet <- predict(model_nnet, test_data, type = "prob")$Yes


# --- ROC CURVE PLOTTING (using pROC library) ---

# Create ROC objects for each model
# The roc() function is robust and handles factors well.
# It needs: roc(actual_labels, predicted_probabilities)

roc_lr <- roc(test_data$ChurnStatus, probs_lr)
roc_rf <- roc(test_data$ChurnStatus, probs_rf)
roc_gbm <- roc(test_data$ChurnStatus, probs_gbm)
roc_xgb <- roc(test_data$ChurnStatus, probs_xgb)
roc_nnet <- roc(test_data$ChurnStatus, probs_nnet)

# Plot the ROC curves
plot(roc_lr, col = "red", main = "ROC Curves for All Models", legacy.axes = TRUE)
lines(roc_rf, col = "blue")
lines(roc_gbm, col = "green")
lines(roc_xgb, col = "purple", lwd = 2.5) # Make champion model thicker
lines(roc_nnet, col = "orange")

legend("bottomright", 
       legend = c(paste("XGBoost (AUC =", round(auc(roc_xgb), 2), ")"),
                  paste("GBM (AUC =", round(auc(roc_gbm), 2), ")"),
                  paste("Random Forest (AUC =", round(auc(roc_rf), 2), ")"),
                  paste("Neural Net (AUC =", round(auc(roc_nnet), 2), ")"),
                  paste("Logistic Reg (AUC =", round(auc(roc_lr), 2), ")")),
       col = c("purple", "green", "blue", "orange", "red"), lwd = c(2.5, 1, 1, 1, 1))



# Plot PR Curves
pr_lr <- pr.curve(scores.class0 = probs_lr[test_data$ChurnStatus == "Yes"], 
                  scores.class1 = probs_lr[test_data$ChurnStatus == "No"], curve = TRUE)
pr_rf <- pr.curve(scores.class0 = probs_rf[test_data$ChurnStatus == "Yes"], 
                  scores.class1 = probs_rf[test_data$ChurnStatus == "No"], curve = TRUE)
pr_gbm <- pr.curve(scores.class0 = probs_gbm[test_data$ChurnStatus == "Yes"], 
                   scores.class1 = probs_gbm[test_data$ChurnStatus == "No"], curve = TRUE)
pr_xgb <- pr.curve(scores.class0 = probs_xgb[test_data$ChurnStatus == "Yes"], 
                   scores.class1 = probs_xgb[test_data$ChurnStatus == "No"], curve = TRUE)
pr_nnet <- pr.curve(scores.class0 = probs_nnet[test_data$ChurnStatus == "Yes"], 
                    scores.class1 = probs_nnet[test_data$ChurnStatus == "No"], curve = TRUE)

plot(pr_lr, col = "red", main = "Precision-Recall Curves", auc.main = FALSE)
plot(pr_rf, col = "blue", add = TRUE)
plot(pr_gbm, col = "green", add = TRUE)
plot(pr_xgb, col = "purple", add = TRUE, lwd = 2)
plot(pr_nnet, col = "orange", add = TRUE)
legend("topright", legend = c("Logistic Reg", "Random Forest", "GBM", "XGBoost (Champion)", "Neural Net"), 
       col = c("red", "blue", "green", "purple", "orange"), lwd = c(2.5, 1, 1, 1, 1))


# # 4. Champion Model Interpretation

## Confusion Matrix: The confusion matrix for the Neural Network model shows its performance in absolute numbers.

# nnet_preds <- predict(model_nnet, test_data)
# cm_nnet <- confusionMatrix(nnet_preds, test_data$ChurnStatus, positive = "Yes")
# print(cm_nnet)

# --- Create a more readable confusion matrix output ---

# 1. Create confusion matrix object
nnet_preds <- predict(model_nnet, test_data)
cm_nnet <- confusionMatrix(nnet_preds, test_data$ChurnStatus, positive = "Yes")

cm_nnet

# Create a function to print it nicely
print_descriptive_cm <- function(cm) {
  # Extract the 4 cells from the confusion matrix table
  tn <- cm$table[1, 1] # True Negative
  fp <- cm$table[1, 2] # False Positive
  fn <- cm$table[2, 1] # False Negative
  tp <- cm$table[2, 2] # True Positive
  
# Use cat() to print formatted text
  cat("Confusion Matrix (Descriptive)\n")
  cat("------------------------------------------------------------------------\n")
  cat(sprintf("%-20s %-30s %-30s\n", "", "Actual", "Actual"))
  cat(sprintf("%-20s %-30s %-30s\n", "Prediction: No", 
              paste(tn, "(True Negatives)"), 
              paste(fn, "(False Negatives)")))
  cat(sprintf("%-20s %-30s %-30s\n", "Prediction: Yes", 
              paste(fp, "(False Positives)"), 
              paste(tp, "(True Positives)")))
  cat("------------------------------------------------------------------------\n")
}

# Call the function to print your result
print_descriptive_cm(cm_nnet)



# 2. Extract the matrix table and convert it to a tidy data frame
cm_table <- as.data.frame(cm_nnet$table)

# 3. Create the descriptive labels for each cell
plot_data <- cm_table %>%
  mutate(
    Labels = case_when(
      Prediction == "No" & Reference == "No"  ~ "True Negative",
      Prediction == "Yes" & Reference == "Yes" ~ "True Positive",
      Prediction == "No" & Reference == "Yes"  ~ "False Negative (Miss)",
      Prediction == "Yes" & Reference == "No"  ~ "False Positive (Mistake)"
    ),
    # Combine the count and the label for plotting
    TileText = paste(Freq, Labels, sep = "\n")
  )

# 4. Reverse the order of the Prediction factor to have "Yes" at the top
plot_data$Prediction <- factor(plot_data$Prediction, levels = c("Yes", "No"))

# 5. Build Confusion matrix ggplot heatmap
ggplot(data = plot_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  # Create the tiles for the heatmap
  geom_tile(color = "white", lwd = 1.5) +
  
  # Add the text labels inside the tiles
  geom_text(aes(label = TileText), vjust = 0.5, size = 5, color = "white", fontface = "bold") +
  
  # Define a color scale (e.g., from light blue to dark blue)
  # Lower counts will be lighter, higher counts will be darker.
  scale_fill_gradient(low = "#6495ED", high = "#00008B") +
  
  # Add titles and labels for clarity
  labs(
    title = "Confusion Matrix Heatmap (Neural Network)",
    subtitle = "Performance on the Unseen Test Set",
    x = "Actual Customer Status (Reference)",
    y = "Model's Prediction"
  ) +
  
  # Apply a clean theme and remove the unnecessary color legend
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    legend.position = "none" # The text in the tiles makes the legend redundant
  )





## 4. Feature Importance


# Get and plot feature importance
importance <- varImp(model_nnet, scale = FALSE)
plot(importance, top = 10, main = "Top 10 Churn Drivers (Neural Network)")

# Prepare feature importance for CSV export
feature_importance_df <- importance$importance %>% as.data.frame() %>% rownames_to_column("Feature") %>% arrange(desc(Overall))

head(feature_importance_df)


# Threshold Tuning
# We analyze how performance metrics change at different probability thresholds to potentially optimize for higher recall.

thresholds <- seq(0.1, 0.9, by = 0.05)
threshold_metrics <- map_df(thresholds, function(thresh) {
  preds <- factor(ifelse(probs_nnet >= thresh, "Yes", "No"), levels = c("No", "Yes"))
  
  # Using tryCatch to handle cases where a threshold results in no predictions for a class
  cm_res <- tryCatch({
    cm <- confusionMatrix(preds, test_data$ChurnStatus, positive = "Yes")
    tibble(Threshold = thresh, Precision = cm$byClass["Precision"], Recall = cm$byClass["Recall"], F1 = cm$byClass["F1"]
           )
  }, error = function(e) {
    tibble(Threshold = thresh, Precision = NA, Recall = NA, F1 = NA)
  })
  
  return(cm_res)
})

kable(threshold_metrics, digits = 3, caption = "Performance Across Thresholds")


# Insight: Lowering the threshold to 0.4 could increase Recall significantly while maintaining acceptable Precision, i mean if the 
#     business wants to cast a wider net for retention campaigns. For this report, we will proceed with the default 0.5 threshold.




# Final Deliverables (with New Business-Facing Segments)


# --- Create a single master prediction file for the entire dataset ---
predictions_all_customers <- df_eng %>% 
                      # Re-create engineered features for the full dataset
  mutate(HasUnresolvedIssues = factor(ifelse(NumUnresolved > 0, "Yes", "No")), 
         IsHighlyInactive = factor(ifelse(DaysSinceLastLogin > 90, "Yes", "No")), 
         AvgTransactionsPerLogin = ifelse(LoginFrequency > 0, NumTransactions / LoginFrequency, 0), 
         AgeGroup = cut(Age, breaks = c(17, 24, 34, 44, 54, 64, Inf), labels = c("18-24", "25-34", "35-44", "45-54", 
                                                                                 "55-64", "65+")),
    # Predict churn probability
    Churn_Probability = predict(model_nnet, newdata = ., type = "prob")$Yes,
    # Create Risk Segments
    RiskSegment = case_when(
      Churn_Probability >= 0.65 ~ "Critical",
      Churn_Probability >= 0.50 ~ "High",
      Churn_Probability >= 0.30 ~ "Medium",
      TRUE ~ "Low"
    ),
    RiskSegment = factor(RiskSegment, levels = c("Critical", "High", "Medium", "Low"))
  )

# --- Deliverable 1: customer_segments.csv ---
customer_segments_report <- predictions_all_customers %>% group_by(RiskSegment) %>%
  summarise(CustomerCount = n(), AvgChurnProbability = mean(Churn_Probability),
            ActualChurnRate = mean(ChurnStatus == "Yes") # Calculate actual churn rate
            ) %>% arrange(factor(RiskSegment, levels = c("Critical", "High", "Medium", "Low")))

write_csv(customer_segments_report, "C:/Users/Charles/Downloads/R Project/My First R Project/customer_segments.csv")
kable(customer_segments_report, digits = 3, caption = "Customer Risk Segment Analysis")

sapply(df_eng, class)


# --- Deliverable 2: top_risk_customers.csv ---
top_risk_customers <- predictions_all_customers %>% filter(RiskSegment == "High") %>% 
  select(CustomerID, NumResolved, HasUnresolvedIssues, NumInteractions, IncomeLevel, ActivityLevel, IsHighlyInactive,
         TransactionFrequency, DaysSinceLastLogin, Age, Churn_Probability) %>% arrange(desc(Churn_Probability))

# Calculate and add actual churn rate for this specific group as a summary
actual_churn_rate_top_risk <- customer_segments_report %>% filter(RiskSegment == "High") %>% pull(ActualChurnRate)

print(paste("Actual Churn Rate for High Risk Customers:", scales::percent(actual_churn_rate_top_risk)))
write_csv(top_risk_customers, "C:/Users/Charles/Downloads/R Project/My First R Project/top_risk_customers.csv")




# --- Deliverable 3: predictions_all_customers.csv ---
write_csv(predictions_all_customers %>% select(CustomerID, Churn_Probability, RiskSegment, ChurnStatus), 
          "predictions_all_customers.csv")

# --- Deliverable 4: feature_importance.csv ---
write_csv(feature_importance_df_v2, "feature_importance.csv")

print("All deliverables, including segmentation report, have been generated.")





# --- Generate the List of 26 Correctly Identified At-Risk Customers ---

# Get the predictions and probabilities for the test set from our champion model

nnet_probs <- predict(model_nnet, test_data, type = "prob")$Yes

# Add the predictions and original CustomerID back to the test data
# We need to re-add CustomerID to join back original features like Age and DaysSinceLastLogin
test_data_with_preds <- test_data %>% mutate(CustomerID = as.numeric(rownames(test_data)), # Get original row number as CustomerID 
                                             Predicted_Churn = nnet_preds, Churn_Probability = nnet_probs
                                             )

# Filter for the True Positives
# These are customers where the Actual Churn was "Yes" AND our Prediction was also "Yes"
true_positives_df <- test_data_with_preds %>% filter(ChurnStatus == "Yes" & Predicted_Churn == "Yes")

# Select the desired columns and join back original data for the final report
# We will select the following columns Since 'ActivityLevel' and 'TransactionFrequency' 
# are concepts, we will use 'LoginFrequency' and 'NumTransactions' as their direct measures.
final_at_risk_list <- true_positives_df %>%
  # Join with the original dataframe to get back 'Age' and 'DaysSinceLastLogin'
  left_join(df %>% select(CustomerID, Age, DaysSinceLastLogin, NumTransactions), by = "CustomerID") %>%
  # Select and reorder the final columns for the report
  select(
    CustomerID,
    Churn_Probability,
    NumResolved,
    HasUnresolvedIssues,
    NumInteractions,
    IncomeLevel,
    IsHighlyInactive,
    NumTransactions,  # Using as 'TransactionFrequency'
    DaysSinceLastLogin,
    Age
  ) %>%
  # Arrange by highest probability to prioritize outreach
  arrange(desc(Churn_Probability))

# Print the final list as a polished table
kable(final_at_risk_list, digits = 3, caption = "List of 26 High-Risk Customers Correctly Identified by the Model (True Positives)")

write_csv(final_at_risk_list, "C:/Users/Charles/Downloads/R Project/My First R Project/top_at_risk_customers.csv")



# 6. Business Recommendations & ROI

## Recommendations
# 1. Overhaul Customer Support Feedback: Immediately address any customer with an unresolved ticket (`NumUnresolved` > 0).
# 2. Launch Re-Engagement Campaign: Target customers with `DaysSinceLastLogin` > 60 days with incentives.
# 3. Create High-Risk Watch-list: 
# 4. Use the model's predictions to assign customers with >75% churn probability to a dedicated retention team.



## CalCalculate ROI based on the below parameters

# Assumptions
arpc <- 1500 # Average Revenue Per Customer
num_customers_total <- nrow(df)
top_risk_pct <- 0.10
intervention_cost <- 50
intervention_success_rate <- 0.30

# Calculate ROI
num_targeted <- ceiling(num_customers_total * top_risk_pct)

# We need to get the actual churn rate of this top 10% group
full_predictions_df <- df_eng %>% mutate(Churn_Probability = predict(model_nnet, newdata = ., type = "prob")$Yes)

top_risk_df <- full_predictions_df %>% arrange(desc(Churn_Probability)) %>% dplyr::slice(1:num_targeted)

# Estimated churners in this group based on their average probability
customers_expected_to_churn <- sum(top_risk_df$Churn_Probability)
customers_saved <- floor(customers_expected_to_churn * intervention_success_rate)
revenue_saved <- customers_saved * arpc
total_intervention_cost <- num_targeted * intervention_cost
net_savings <- revenue_saved - total_intervention_cost
roi <- net_savings / total_intervention_cost

roi_summary <- tibble(
  Metric = c("Customers Targeted", "Customers Saved", "Total Revenue Saved", "Total Intervention Cost", "Net Savings", "ROI"),
  Value = c(num_targeted, customers_saved, revenue_saved, total_intervention_cost, net_savings, roi)
)

kable(roi_summary, caption = "Estimated ROI of Intervention Campaign", format.args = list(big.mark = ","))


















