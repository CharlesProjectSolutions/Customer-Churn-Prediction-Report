# 1. Context:
- **Client:** Financial services provider
- **Problem:** A financial services provider is experiencing high churn among personal and small business customers
- **Goal:** Build a reliable classification model that predicts the probability of a customer churning and recommend interventions to reduce attrition
 
 --- 

# 2. Scope:
1. Data preprocessing:
   - Data Overview
   - Exploratory Data Analysis (EDA)
   - Handle missing values, duplicates, and outliers
   - Feature engineering
2. Multicollinearity handling (VIF, drop redundant features)
3. Class imbalance treatment:
   - Apply SMOTE, class weights, and threshold tuning
4. Model training:
   - Use stratified train/test split and cross-validation
   - Train 5 algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, Neural Network
5. Model evaluation:
   - Metrics: accuracy, precision, recall, specificity, F1-score, ROC-AUC, PR-AUC
   - Visualizations: ROC curves, PR curves, confusion matrices, feature importance plots
6. Interpretation:
   - Identify top features driving churn
   - Provide business recommendations tied to these drivers
7. ROI analysis:
   - Estimate cost vs. benefit of interventions
   * Calculate ROI for retention strategies
     
 --- 
 
# 3. Deliverables:
- predictions_all_customers.csv (CustomerID, ChurnProbability, PredictedChurn)
- Model report (executive summary with metrics, visuals, recommendations, ROI)
  
---

# 1. Data Preprocessing & Feature Engineering
Source Files Consolidated:
   - Transaction_history.csv (5054 obs. with 5 variables)
   - Customer_service.csv (1002 obs. with 5 variables)
   - Customer_activity.csv (1000 obs. with 4 variables)
   - Customer_demographics.csv (1000 obs. with 5 variables)
   - Churn.csv (1000 obs. with 2 variables).

  
   - Data Model
         <img width="33000" height="2500" alt="image" src="https://github.com/user-attachments/assets/7cca7e56-342d-46e3-8b18-a300e7837ff5" />

   - 5 fives prepared & consolidated into one single source of truth (Customer_Churn_Data.csv).
   - Size: 1,000 customers with 17 features
   - Feature Categories: Numerical Features (11), Categorical Features (4)


#    Exploratory Data Analysis
   - **Missing & Duplicate Values:** The dataset was complete with no missing & duplicate values.
   - **Feature Engineering:**
        - **AgeGroup:** The Age feature was binned into categories (18-24, 25-34, 35-44, 45-54, 55-64, 65+) to capture non-linear relationships.
        - **EngagementScore:** A composite score was created by combining LoginFrequency and NumTransactions to represent overall customer engagement.
        - **Categorical Encoding:** Features like Gender, MaritalStatus, IncomeLevel, and ServiceUsage were converted into numerical format using one-hot encoding.

   - **Visualize Distributions**
   <img width="2000" height="1200" alt="EDA" src="https://github.com/user-attachments/assets/994bd48e-e57a-49ba-9574-3b59b579c616" />

### Key Findings
   - **Class Imbalance:** The target variable ChurnStatus was imbalanced, with churned customers representing only 20% of the dataset, which will require balancing strategies (SMOTE, class weights, etc.) in modeling. To address the issue of class imbalance, SMOTE was applied during resampling to prevent data leakage 
   - **Age Distribution:** Customers are fairly evenly distributed across ages 18 – 69, and no obvious skew.
   - **Churn by Income Level:** Churn is present across all income levels. Slightly higher churn counts in Low income groups (Could suggest income sensitivity plays a role in churn).
   - **Login Frequency vs Churn:** Churners show lower median login frequency compared to retained customers. It suggests engagement is a strong predictor of retention.
   - **Customer Service Resolution vs Churn:** The churn rate is almost identical between resolved (20.5%) and unresolved (20.7%) cases. **Resolution Status** alone may not be a strong churn predictor. So, it might not be whether an issue was resolved or unresolved. It may be how long it took, how many interactions were addressed, or how many issues were unresolved over time.


     **Visualize Correlations & Identify Multicollinearity**
     
     <img width="2000" height="1200" alt="Correlation" src="https://github.com/user-attachments/assets/c3ce3106-ce53-42df-80e9-aec9e2a03306" />

 - High correlation found between a couple of features. For example: TotalSpend, AvgSpend and NumTransactions etc.
 - Drop highly correlated (e.g. TotalSpend will be dropped in favor of the more granular AvgSpend and NumTransactions) features to avoid **redundancy**.
 
 --- 

# 2. Model Training & Evaluation

Five machine learning models were trained and evaluated. The data was split into 80% for training and 20% for testing. The models were evaluated on their ability to correctly identify churners (Recall) and their overall predictive power (ROC-AUC and PR-AUC).

###    Final Model Performance
<img width="2000" height="1500" alt="Model Performance Comparison On Test Set" src="https://github.com/user-attachments/assets/6520e695-1237-444d-81c0-7d9de850c4e2" />

**Conclusion:** After multiple iterations, **Neural Network** was selected as the model champion despite having the lowest accuracy but particularly having the highest Recall and AUC score of 0.52.

*   **Why?** Since the primary business goal is to **identify as many potential churners as possible**. Recall is therefore the most critical metric here.
*   **Neural Network**'s Recall of **0.650** means it successfully finds at least 65% of all true churners. The other models, with a Recall of only 5 to 10%, are practically useless for this business objective.

---

# 3. Champion Model Deep Dive (**Neural Network**) 
The following visuals provide a detailed look at the final model's performance.

*   **Confusion Matrix**

This shows the model's predictions versus the actual outcomes. Out of 40 customers who actually churned in the test set, our model correctly identified 26 of them.

<img width="1000" height="500" alt="Rplot" src="https://github.com/user-attachments/assets/2dfb0d73-dd4d-4016-8c30-ab77f66e51e6" />

*   **ROC Curves**
These curves illustrate the model's ability to distinguish between churning and non-churning customers across all probability thresholds.

<img width="1500" height="1000" alt="ROC curves" src="https://github.com/user-attachments/assets/04cd2e7d-5900-496e-93e4-da7c3a50e122" />


---


# 3. Key Drivers of Churn (**Feature Importance**) 
Understanding why a customer churns is critical. The model identified the following factors as the most significant predictors of churn:

<img width="1500" height="1000" alt="Top 10 Churn Drivers" src="https://github.com/user-attachments/assets/bd7dc8eb-0e89-46df-b6b0-c4549dea57f4" />

*   **NumUnresolved:** The absolute number of unresolved support tickets is a major friction point. Even one or two unresolved issues significantly increase churn probability.
*   **HasUnresolvedIssues:** Customers whose support tickets are not resolved (or have a low resolution rate) are extremely likely to churn. This points directly to dissatisfaction with customer service.


---
*   **Diagnosis: What Could Have Gone Wrong?**

These results are a classic symptom of a machine learning pipeline where the class imbalance may not have been handled correctly. The tree-based models (RF, GBM, XGBoost) are notoriously sensitive to imbalanced data & have defaulted to predicting the majority class.

The fact that SMOTE was included in the trainControl but the results are still this poor suggests that it may not have been applied effectively or that the underlying features have very weak predictive power.

**My Recommendations & Next Steps Would Be:**
1. Re-evaluate the SMOTE Implementation:
     - Double-check that the caret trainControl is correctly configured. Sometimes a simple syntax error can cause the sampling step to be skipped.
    - Try an alternative to SMOTE. In trainControl, change sampling = "smote" to sampling = "up" (upsampling) or sampling = "down" (downsampling) and retrain the models to see if the results improve.

2. Conduct Deeper Feature Engineering:
 - The near-random AUC scores suggest that the features themselves might not have predictive power. I would go back and create more powerful features. For example:
 - Interaction Terms: Does Age combined with IncomeLevel have an effect?
 - Ratio Features: What is the ratio of NumTransactions to LoginFrequency?
 - Create binary flags like HasUnresolvedTickets (1 if NumUnresolved > 0, else 0) or IsHighlyInactive (1 if DaysSinceLastLogin > 90, else 0)

