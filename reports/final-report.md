
# Portfolio Report — Regression & Classification

## Overview
This portfolio evaluates two supervised learning problems using real datasets. For **regression**, we predict house sale prices from King County (Seattle area). For **classification**, we predict whether a firm is bankrupt given financial ratios. For each task we train **two algorithms**, perform basic hyperparameter tuning, and compare results with appropriate metrics. Reproducibility is ensured via scikit‑learn Pipelines that encapsulate preprocessing and modeling.

---
## Task 1 — Regression: King County House Prices

### Dataset
- **Source:** Kaggle public dataset “kc_house_data” (houses sold in King County, USA). The target is `price` (continuous).
- **Size:** ~21k rows, 19 features (e.g., bedrooms, bathrooms, living area, lot size, floors, condition, grade, waterfront, view, zipcode, lat/long). A date column contains the sale timestamp; `id` serves as an identifier.
- **Cleaning decisions:** Remove `id`. Convert `date` to `sale_year` and `sale_month` to capture seasonality while avoiding timestamp leakage. Median imputation for numeric features; most-frequent for categoricals.

### Method
- **Split:** 80/20 train/test split with `random_state=42`.
- **Preprocessing:** Numeric → median imputation + standardization. Categorical → most‑frequent imputation + one‑hot encoding (OHE). Applied through a `ColumnTransformer`.
- **Models:**
  1. **Ridge Regression** (linear model with L2 regularization). Tuning: `alpha ∈ {0.1, 1, 10, 100}`.
  2. **Random Forest Regressor** (nonlinear ensemble). Tuning: `n_estimators ∈ {300, 600}`, `max_depth ∈ {None,12,20}`, `min_samples_split ∈ {2,5}`.
- **Scoring & Validation:** 5‑fold cross‑validation on the training set with **RMSE** as the objective; final evaluation on the held‑out test set with **RMSE** and **R²**.
- **Feature importance:** Extract impurity‑based importance from the forest and visualize the top features.

### Results (summary)
- **Ridge** typically achieves strong baseline performance due to the many near‑linear relations (e.g., sqft_living and price), but cannot capture nonlinearities or interactions like location × grade.
- **Random Forest** usually improves RMSE and R² by modeling nonlinear effects and interactions (e.g., sharp price jumps at higher `grade` and `view` levels). In most runs, RF outperforms Ridge by a material margin.
- **Top features:** `sqft_living`, `grade`, `view`, `bathrooms`, `waterfront`, and geographic coordinates often dominate feature importance; `sale_month` captures mild seasonality.

### Discussion
- **Why RF > Ridge:** House prices exhibit nonlinear relationships and thresholds (e.g., premium for waterfront or certain views). RF captures these with tree splits; Ridge cannot without manual feature engineering.
- **Limitations:** No inflation adjustment across years; potential spatial leakage if very close neighbors end up across splits; no log‑transform of skewed targets (optional extension). Appraisal features (`grade`, `condition`) can be subjective.
- **Improvements:** Try **Gradient Boosting** or **XGBoost**, add engineered features (`price_per_sqft`, distance‑to‑center), and consider log‑transforming `price` to stabilize variance.

---
## Task 2 — Classification: Company Bankruptcy

### Dataset
- **Source:** Taiwanese Company Bankruptcy dataset. **Target**: `Y` (1 = bankrupt, 0 = non‑bankrupt). Features `X1..X95` include profitability, leverage, liquidity, turnover, and cash flow ratios (see documentation).
- **Size & characteristics:** High‑dimensional tabular data; **class imbalance** (bankrupt firms are a small minority).
- **Cleaning:** Features are numeric; we standardize after median imputation. No text features.

### Method
- **Split:** Stratified 80/20 to preserve class ratio.
- **Preprocessing:** Numeric imputation (median) + scaling; categorical OHE (none/rare).
- **Models:**
  1. **Logistic Regression** with **class_weight** set to the *balanced* weights computed from the training set; tuning `C ∈ {0.1, 1, 10}`.
  2. **Random Forest Classifier** with `class_weight='balanced'`; tuning `n_estimators ∈ {400, 700}`, `max_depth ∈ {None, 12, 20}`, `min_samples_split ∈ {2,5}`.
- **Metrics:** Accuracy is misleading with imbalance. We emphasize **ROC‑AUC**, and report **precision, recall, F1** at the default threshold along with **confusion matrices** and **precision‑recall curves**.

### Results (expected patterns)
- **Baseline:** Logistic regression provides a well‑calibrated decision boundary and good interpretability. With proper scaling and class weights, it delivers competitive ROC‑AUC.
- **Random Forest:** Often delivers a higher ROC‑AUC by capturing nonlinear interactions among ratios (e.g., leverage × liquidity). However, precision at default threshold can be low; PR curve helps pick an operating point depending on business costs (missing a bankruptcy vs. false alarms).
- **Important features (RF):** leverage ratios (Debt/Equity, Liability/Assets), profitability (`ROA`, `Net Income to TA`), and cash‑flow‑based measures typically rank highly, aligning with finance theory.

### Discussion
- **Handling Imbalance:** We used class weighting. As extensions, try **SMOTE** on the training set or adjust decision thresholds for a target recall (e.g., >80%). Cost‑sensitive evaluation is recommended.
- **Limitations:** Financial ratios can be correlated; tree ensembles handle this reasonably, but linear models may need regularization paths or feature selection. Temporal drift may affect generalization if training/test split ignores time order.
- **Improvements:** Add **Gradient Boosted Trees** (e.g., XGBoost/LightGBM), perform **feature selection** with mutual information, calibrate probabilities with **Platt/Isotonic**, and tune the threshold to a specified precision/recall target.

---
## Consolidated Comparison

| Task | Best Model (typical) | Why it wins | Primary Metric(s) |
|---|---|---|---|
| House Prices (regression) | Random Forest Regressor | Captures nonlinearities, interactions, and thresholds | Lower RMSE, higher R² |
| Bankruptcy (classification) | Random Forest or Logistic (depending on class ratio) | RF for nonlinearity; Logistic for calibrated probabilities and simplicity | Higher ROC‑AUC; PR curve area |

**Interpretability vs. accuracy:** Linear models are easier to explain and deploy; forests typically gain accuracy but reduce interpretability. Use feature importance and partial dependence (future work) to communicate drivers.

---
## Reproducibility & How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Open the notebooks: `01_regression_house_prices.ipynb` and `02_classification_bankruptcy.ipynb`.
3. Run all cells. Artifacts (predictions, feature importances, plots) are saved to `reports/`.
4. For grading, include this report and exported charts.

---
## References
- scikit‑learn User Guide & API docs (pipelines, model selection, preprocessing).
- Kaggle King County House Prices dataset page.
- Company Bankruptcy (Taiwan) dataset documentation.
