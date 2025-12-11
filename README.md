# E-Commerce Fraud Detection

## Project Overview
In this project, I built a machine learning pipeline to detect fraudulent transactions in a highly imbalanced e-commerce dataset (2% fraud rate). 

**Goal:** Maximize the detection of fraud (Recall) while minimizing financial loss from false positives.

## Business Impact
By optimizing the classification threshold using the Precision-Recall curve, this model is projected to save **~$467,530** on the test dataset compared to a baseline of no fraud detection.

* **Baseline Cost:** ~$992,000 (Fraud losses)
* **Model Cost:** ~$524,470 (Operational costs + Remaining fraud)
* **ROI:** ~89%

## Tech Stack
* **Python:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)
* **Modeling:** Logistic Regression vs. Random Forest vs. **XGBoost (Champion)**
* **Techniques:** SMOTE Pipeline, GridSearch Cross-Validation, Threshold Tuning.

## Key Insights
1.  **Time Matters:** Fraud rates spike at 11 PM and 1 AM (Late night activity).
2.  **Location Mismatch:** Transactions where the user country differs from the bank country are **7-8x** more likely to be fraud.
3.  **Spending Behavior:** Sudden deviations from a user's average spending (`amount_vs_avg`) were strong indicators of fraud.

## How to Run
1.  Clone the repo
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the notebook: `Ecommerce_Fraud_Detection_XGBoost.ipynb`
