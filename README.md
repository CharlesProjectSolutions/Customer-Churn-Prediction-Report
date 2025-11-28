# Context:
- **Client:** Financial services provider
- **Problem:** A financial services provider is experiencing high churn among personal and small business customers
- **Goal:** Build a reliable classification model that predicts the probability of a customer churning and recommend interventions to reduce attrition

---

Scope:
1. Data preprocessing:
   - Handle missing values, duplicates, and outliers
   - Feature engineering (ratios, engagement metrics)
   - Multicollinearity handling (VIF, drop redundant features)
2. Class imbalance treatment:
   - Apply SMOTE, class weights, and threshold tuning
3. Model training:
   - Train 5 algorithms: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM
   - Use stratified train/test split and cross-validation
4. Model evaluation:
   - Metrics: accuracy, precision, recall, specificity, F1-score, ROC-AUC, PR-AUC
   - Visualizations: ROC curves, PR curves, confusion matrices, feature importance plots
5. Interpretation:
   - Identify top features driving churn
   - Provide business recommendations tied to these drivers
6. ROI analysis:
   - Estimate cost vs. benefit of interventions
   - Calculate ROI for retention strategies

# Deliverables:
- predictions_all_customers.csv (CustomerID, ChurnProbability, PredictedChurn)
- top_risk_customers.csv (Top 10% highest churn probability)
- feature_importance.csv (Ranked drivers of churn)
- threshold_metrics.csv (Performance across thresholds)
- Model report (executive summary with metrics, visuals, recommendations, ROI)


---

## 2. Methodology & Pipeline

This project followed an iterative, multi-stage methodology, moving from a naive first attempt to a robust final solution.

### Stage 1: Data Wrangling and Feature Engineering
*   **Initial Cleanup:** Loaded the `.csv` data, converted the target variable `ChurnStatus` to a factor.
*   **Feature Creation:** Engineered new, high-impact features to better capture customer behavior, including:
    *   `HasUnresolvedIssues`: A binary flag (Yes/No) to explicitly signal poor service experiences.
    *   `IsHighlyInactive`: A binary flag for customers with `DaysSinceLastLogin` > 90.
    *   `AvgTransactionsPerLogin`: A ratio feature to measure engagement efficiency.
    *   `AgeGroup`: Binned the `Age` variable to capture non-linear trends.
*   **Data Reduction:** Handled multicollinearity by removing the `TotalSpend` feature, which was highly correlated with `NumTransactions`.

### Stage 2: Data Modeling (An Iterative Process)
This project's core lesson was in its iterative modeling approach.

*   **Attempt #1 (Standard Pipeline & Failure):**
    *   **Method:** A standard pipeline was built using `caret` with 10-fold cross-validation and **SMOTE** to handle class imbalance.
    *   **Result:** This approach failed. Tree-based models like Random Forest and XGBoost achieved high accuracy (~78%) but had a **Recall of only 5%** and an ROC-AUC near 0.5.
    *   **Diagnosis:** The models were not learning; they were simply predicting the majority class ("No Churn"). SMOTE was proving ineffective for this particular dataset.

*   **Attempt #2 (Robust Pipeline & Success):**
    *   **Method:** The pipeline was rebuilt to use **direct class weighting**, a more powerful technique. The weight for the positive class was calculated as `(count of 'No') / (count of 'Yes')` and applied directly during model training.
    *   **Result:** This was successful. The models began to learn the patterns of the minority class, producing meaningful and varied results.

### Stage 3: Compare and Justify Model Selection

After the robust pipeline was built, the models were evaluated on the unseen test set.

**Final Model Performance:**


**Model Justification:**
Despite having the lowest accuracy, **Logistic Regression was selected as the champion model.**

*   **Why?** The primary business goal is to **identify as many potential churners as possible**. Recall is therefore the most critical metric. Logistic Regression's Recall of **47.5%** means it successfully finds almost half of all true churners. The other models, with a Recall of only 5%, are practically useless for this business objective. This is a classic case of choosing the right tool for the job, not just the one with the highest score on a generic metric.

---

## 3. Business Insights & Recommendations

Using the champion Logistic Regression model, the following actionable insights were derived:

*   **Insight 1: Poor Service is the Biggest Driver of Churn.** The most important features were directly related to customer service interactions (`ResolutionRate`, `HasUnresolvedIssues`).
    *   **Recommendation:** Overhaul the customer support feedback loop. Implement an immediate follow-up protocol for any customer with an unresolved ticket.

*   **Insight 2: Disengagement is a Clear Warning Sign.** The next most important features were related to customer inactivity (`DaysSinceLastLogin`, `LoginFrequency`).
    *   **Recommendation:** Launch a proactive re-engagement campaign targeting customers flagged by the model as "Highly Inactive."

*   **Business Impact:** By using the model to create customer risk segments ("Critical", "High", "Medium", "Low"), the retention team can focus their budget and efforts on the small group of customers who are most likely to churn, dramatically increasing the efficiency and ROI of their campaigns.
