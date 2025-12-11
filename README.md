# E-Commerce Fraud Detection System

## Project Overview
Fraud detection is a critical challenge in e-commerce, where the cost of missed fraud (chargebacks, lost goods) must be balanced against the cost of false alarms (customer friction).

This project builds an end-to-end Machine Learning pipeline to detect fraudulent transactions in a **highly imbalanced dataset** (~2% fraud). By optimizing the classification threshold using the Precision-Recall curve, the final model focuses on maximizing **Net Financial Savings** rather than just technical accuracy.

## Business Impact Summary
The final model was evaluated against a baseline of "No Fraud Detection" (letting all transactions pass).

| Metric | Value |
| :--- | :--- |
| **Champion Model** | **XGBoost Classifier** |
| **Baseline Fraud Loss** | ~$992,000 |
| **Projected Net Savings** | **~$467,530** (on Test Data) |
| **Cost Reduction** | **47.1%** |

---

## Data & Exploratory Analysis
**Dataset:** [Kaggle E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)
* **Size:** ~300k Transactions
* **Imbalance:** 2.21% Fraud Rate

### Key Findings
1.  **The "Midnight" Spike:** Fraud rates spike significantly at **11 PM (23:00)** and early morning hours (**1 AM - 2 AM**).
2.  **Location Mismatch:** Transactions where the User Country differed from the Bank Country were **7-8x** more likely to be fraud (>11% risk vs 1.5% baseline).
3.  **Spending Anomalies:** Fraudulent transactions often showed extreme deviations from a user's historical average spending.

---

## Methodology

### 1. Feature Engineering
We moved beyond raw data to create behavioral features:
* `amount_vs_avg`: Ratio of current transaction amount to user's historical average.
* `time_since_last_tx`: Velocity check (seconds since previous transaction).
* `is_high_risk_hour`: Binary flag for the 11 PM - 2 AM window.
* `location_mismatch`: Binary flag for cross-border discrepancies.

### 2. Handling Imbalance (The Right Way)
* **Split First:** Data was split into Train/Test *before* any resampling to prevent data leakage.
* **Pipeline:** Used `imblearn.pipeline` to apply **SMOTE (Synthetic Minority Over-sampling Technique)** *only* within cross-validation training folds.

### 3. Model Selection & Tuning
Three models were compared using `StratifiedKFold` Cross-Validation:
* **Logistic Regression** (Baseline)
* **Random Forest**
* **XGBoost** (Winner)

**Hyperparameter Tuning:** Performed via `GridSearchCV` to optimize `n_estimators`, `max_depth`, and `scale_pos_weight`.

### 4. Threshold Optimization
Standard models classify at a 0.5 probability. In fraud, this is rarely optimal.
* We utilized the **Precision-Recall Curve** to identify a custom decision threshold (~0.65).
* This shift sacrificed some Recall to drastically increase Precision, minimizing expensive False Positives (blocking good users).

---

## Results Visualization

### Feature Importance (XGBoost)
The model found that **Payment Method (`channel_web`)** and **Location Mismatch** were the strongest predictors, validating our EDA findings.

### Confusion Matrix (Test Set)
* **True Negatives (Legit):** High accuracy in passing good customers.
* **True Positives (Caught Fraud):** The model successfully flagged high-value fraud cases while maintaining a manageable False Positive rate.

---

## Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/ecommerce-fraud-detection.git](https://github.com/yourusername/ecommerce-fraud-detection.git)
    cd ecommerce-fraud-detection
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis**
    Open `Ecommerce_Fraud_Detection.ipynb` in Jupyter Lab or VS Code.
    *Note: The script automatically downloads the dataset using the Kaggle API.*

---

## Contact
* Name: **Pham Thanh Dat**
* LinkedIn: [[LinkedIn]](https://www.linkedin.com/in/dat-pham-840797313/)
