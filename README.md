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

# Data Overview
1. Source Files Consolidated:
   - Transaction_history.csv (5054 obs. with 5 variables),
   - Customer_service.csv (1002 obs. with 5 variables),
   - Customer_activity.csv (1000 obs. with 4 variables),
   - Customer_demographics.csv (1000 obs. with 5 variables),
   - Churn.csv (1000 obs. with 2 variables).

  
   - Data Model
         <img width="33000" height="2500" alt="image" src="https://github.com/user-attachments/assets/7cca7e56-342d-46e3-8b18-a300e7837ff5" />

         
   - Master File: Consolidated into a single source of truth (Customer_Churn_Data.csv).
   - Size: 1,000 customers with 17 features
   - Target Variable: ChurnStatus (Binary: 0=Not Churned, 1=Churned)
   - Class Imbalance: 79.6% Not Churned, 20.4% Churned (3.9:1 ratio)
   - Feature Categories: Numerical Features (11), Categorical Features (4)


#    Exploratory Data Analysis (EDA) 

   <img width="20000" height="1500" alt="EDA" src="https://github.com/user-attachments/assets/994bd48e-e57a-49ba-9574-3b59b579c616" />

### Key Findings
   - **Churn Distribution:** Around 20% churn rate. Dataset is imbalanced (more non-churners), which will require balancing strategies (SMOTE, class weights, etc.) in modeling.
   - **Age Distribution:** Customers are fairly evenly distributed across ages 16 – 69. No obvious skew, so age can be used directly (maybe binned into groups for interpretability).
   - **Churn by Income Level:** Churn is present across all income levels. Slightly higher churn counts in Low income groups (Could suggest income sensitivity plays a role in churn).
   - **Login Frequency vs Churn:** Churners show lower median login frequency compared to retained customers. It suggests engagement is a strong predictor of retention.
   - **Customer Service Resolution vs Churn:** The churn rate is almost identical between resolved (20.5%) and unresolved (20.7%) cases. So, **Resolution Status** alone may not be a strong churn predictor.
         - Hypothesis: It might not be whether an issue was resolved or unresolved. It may be how long it took, how many interactions were needed, or how many issues were unresolved over time.
 
 
 --- 

**Final Model Performance:**

**Model Justification:**
**Neural Network** was selected as the model champion despite having the lowest accuracy.
<img width="2000" height="1500" alt="Model Performance Comparison On Test Set" src="https://github.com/user-attachments/assets/6520e695-1237-444d-81c0-7d9de850c4e2" />

*   **Why?** Well, the primary business goal is to **identify as many potential churners as possible**. Recall is therefore the most critical metric. **Neural Network**'s Recall of **0.652** means it successfully finds at least 65% of all true churners. The other models, with a Recall of only 5%, are practically useless for this business objective. This is a classic case of choosing the right tool for the job, not just the one with the highest score on a generic metric.

*   **Business Impact:** By using the model to create customer risk segments ("Critical", "High", "Medium", "Low"), the retention team can focus their budget and efforts on the small group of customers who are most likely to churn, dramatically increasing the efficiency and ROI of their campaigns.

---



