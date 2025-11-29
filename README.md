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
         <img width="2000" height="1500" alt="image" src="https://github.com/user-attachments/assets/7cca7e56-342d-46e3-8b18-a300e7837ff5" />

         
   - Master File: Consolidated into a single source of truth (Customer_Churn_Data.csv).
   - Size: 1,000 customers with 17 features
   - Target Variable: ChurnStatus (Binary: 0=Not Churned, 1=Churned)
   - Class Imbalance: 79.6% Not Churned, 20.4% Churned (3.9:1 ratio)
   - Feature Categories: Numerical Features (11), Categorical Features (4)
     
 --- 




**Final Model Performance:**


**Model Justification:**
Despite having the lowest accuracy, **Neural Network** was selected as the model champion.

*   **Why?** The primary business goal is to **identify as many potential churners as possible**. Recall is therefore the most critical metric. **Neural Network**'s Recall of **65%** means it successfully finds at least 65% of all true churners. The other models, with a Recall of only 5%, are practically useless for this business objective. This is a classic case of choosing the right tool for the job, not just the one with the highest score on a generic metric.

*   **Business Impact:** By using the model to create customer risk segments ("Critical", "High", "Medium", "Low"), the retention team can focus their budget and efforts on the small group of customers who are most likely to churn, dramatically increasing the efficiency and ROI of their campaigns.

---



