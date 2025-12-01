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
- top_risk_customers.csv (Top 10% highest churn probability)
- feature_importance.csv (Ranked drivers of churn)
- threshold_metrics.csv (Performance across thresholds)
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

   - Master File: Consolidated into a single source of truth (Customer_Churn_Data.csv).
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

   - **Visualize Correlations & Identify Multicollinearity**
     
     <img width="2000" height="1200" alt="Correlation" src="https://github.com/user-attachments/assets/c3ce3106-ce53-42df-80e9-aec9e2a03306" />


### Key Findings
   - **Class Imbalance:** The target variable ChurnStatus was imbalanced, with churned customers representing only 20% of the dataset, which will require balancing strategies (SMOTE, class weights, etc.) in modeling.
   - **Age Distribution:** Customers are fairly evenly distributed across ages 16 – 69. No obvious skew, so age can be used directly (maybe binned into groups for interpretability).
   - **Churn by Income Level:** Churn is present across all income levels. Slightly higher churn counts in Low income groups (Could suggest income sensitivity plays a role in churn).
   - **Login Frequency vs Churn:** Churners show lower median login frequency compared to retained customers. It suggests engagement is a strong predictor of retention.
   - **Customer Service Resolution vs Churn:** The churn rate is almost identical between resolved (20.5%) and unresolved (20.7%) cases. **Resolution Status** alone may not be a strong churn predictor. So, it might not be whether an issue was resolved or unresolved. It may be how long it took, how many interactions were addressed, or how many issues were unresolved over time.
    - **Multicollinearity:** A high correlation was found between TotalSpend, AvgSpend and NumTransactions etc. To avoid redundancy, TotalSpend was dropped in favor of the more granular AvgSpend and NumTransactions features 
 
 --- 

# 2. Model Training & Evaluation

Five machine learning models were trained and evaluated. The data was split into 80% for training and 20% for testing. The models were evaluated on their ability to correctly identify churners (Recall) and their overall predictive power (ROC-AUC and PR-AUC).

###    Final Model Performance & Justification
<img width="2000" height="1500" alt="Model Performance Comparison On Test Set" src="https://github.com/user-attachments/assets/6520e695-1237-444d-81c0-7d9de850c4e2" />

**Conclusion:** After multiple iterations, **Neural Network** was selected as the model champion despite having the lowest accuracy but particularly having the highest Recall and AUC score of 0.52.

*   **Why?** Since the primary business goal is to **identify as many potential churners as possible**. Recall is therefore the most critical metric here.
*   **Neural Network**'s Recall of **0.650** means it successfully finds at least 65% of all true churners. The other models, with a Recall of only 5 to 10%, are practically useless for the business objective.

---

# 3. Champion Model Deep Dive (**Neural Network**) 
The following visuals provide a detailed look at the final model's performance.

*   **Confusion Matrix**

This shows the model's predictions versus the actual outcomes. Out of 40 customers who actually churned in the test set, our model correctly identified 26 of them.



---

