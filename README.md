# 1. Context:
- **Client:** Financial services provider
- **Problem:** A financial services provider is experiencing high churn among personal and small business customers
- **Goal:** Build a reliable classification model that predicts the probability of a customer churning and recommend interventions to reduce attrition

---

# 2. Data Overview
1. Source Files Consolidated:
   - Transaction_history.csv
   - Customer_service.csv
   - Customer_activity.csv
   - Customer_demographics.csv
   - Churn.csv: Target Variable: ChurnStatus (0 = retained, 1 = churned).

   - Data Model
         <img width="1910" height="942" alt="Churn Analysis Data Model" src="https://github.com/user-attachments/assets/c714ad6c-23c5-49f8-91e9-ab7a1690d499" />             
   - Master File: Consolidated into a single source of truth (Customer_Churn_Data.csv).
     
 --- 

# 3. Scope:
1. Data preprocessing:
   - Perform Exploratory Data Analysis (EDA)
   - Handle missing values, duplicates, and outliers
   - Feature engineering (ratios, engagement metrics)
     *   `HasUnresolvedIssues`: A binary flag (Yes/No) to explicitly signal poor service experiences.
     *   `IsHighlyInactive`: A binary flag for customers with `DaysSinceLastLogin` > 90.
     *   `AvgTransactionsPerLogin`: A ratio feature to measure engagement efficiency.
     *   `AgeGroup`: Binned the `Age` variable to capture non-linear trends.
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
 
# 4. Deliverables:
- predictions_all_customers.csv (CustomerID, ChurnProbability, PredictedChurn)
- top_risk_customers.csv (Top 10% highest churn probability)
- feature_importance.csv (Ranked drivers of churn)
- threshold_metrics.csv (Performance across thresholds)
- Model report (executive summary with metrics, visuals, recommendations, ROI)


---


**Final Model Performance:**


**Model Justification:**
Despite having the lowest accuracy, **Neural Network** was selected as the model champion.

*   **Why?** The primary business goal is to **identify as many potential churners as possible**. Recall is therefore the most critical metric. **Neural Network**'s Recall of **65%** means it successfully finds at least 65% of all true churners. The other models, with a Recall of only 5%, are practically useless for this business objective. This is a classic case of choosing the right tool for the job, not just the one with the highest score on a generic metric.

---

## 3. Business Insights & Recommendations

Using the champion Logistic Regression model, the following actionable insights were derived:

*   **Insight 1: Poor Service is the Biggest Driver of Churn.** The most important features were directly related to customer service interactions (`ResolutionRate`, `HasUnresolvedIssues`).
    *   **Recommendation:** Overhaul the customer support feedback loop. Implement an immediate follow-up protocol for any customer with an unresolved ticket.

*   **Insight 2: Disengagement is a Clear Warning Sign.** The next most important features were related to customer inactivity (`DaysSinceLastLogin`, `LoginFrequency`).
    *   **Recommendation:** Launch a proactive re-engagement campaign targeting customers flagged by the model as "Highly Inactive."

*   **Business Impact:** By using the model to create customer risk segments ("Critical", "High", "Medium", "Low"), the retention team can focus their budget and efforts on the small group of customers who are most likely to churn, dramatically increasing the efficiency and ROI of their campaigns.
