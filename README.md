# Executive Summary
This report details an end-to-end machine learning pipeline built to predict customer churn for a financial services provider. Faced with high customer attrition, the goal was to develop a model that could accurately identify at-risk customers and provide actionable insights for retention campaigns.

After evaluating five algorithms, an **Neural Network** was selected as the champion model, achieving an **ROC-AUC of 0.52** and a **Recall of 65%**. The primary drivers of churn were identified as poor customer service experiences (low `ResolutionRate`, `HasUnresolvedIssues`) and low customer engagement (`ActivityLevel`) etc.

Based on these findings, we recommend targeted interventions for the top 10% of at-risk customers. A conservative ROI analysis projects with an investment of **$5000** could save approximately **$22,000** in retained revenue, yielding a potential **ROI of 4.4x**. All deliverables, including customer risk lists and feature importance, have been generated.. The project highlights the importance of aligning model evaluation with business goals and demonstrates a robust, iterative approach to data science.

---

## 1. Project Goal & Business Problem

*   **Goal:** To build a reliable classification model that predicts the probability of a customer churning.
*   **Business Problem:** A financial services provider is experiencing high customer attrition. They need a data-driven tool to identify at-risk customers *before* they leave, enabling proactive retention campaigns.
*   **Challenge:** The dataset is highly imbalanced, with churners representing a small minority. This poses a significant challenge for standard modeling techniques.

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
