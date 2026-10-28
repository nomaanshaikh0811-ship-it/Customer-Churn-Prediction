# Presentation Outline: Customer Churn Prediction & Retention Segmentation
## 6-Slide Executive Deck for Practical 10 Lab Examination

---

### Slide 1: Title & Executive Summary
- **Title:** Customer Churn Prediction and Retention Segmentation for Telecom & Subscription Services
- **Student Details:** TY Computer Engineering — Machine Learning Lab (Practical 10)
- **Problem Statement:** Telecom operators suffer high revenue leakage from subscriber churn. Acquiring new customers costs 5x-7x more than retaining existing ones.
- **Goal:** Develop an end-to-end ML solution to:
  1. Accurately predict churn risk using tuned supervised classifiers.
  2. Segment at-risk subscribers using unsupervised clustering for personalized retention actions.
  3. Deploy a production-ready interactive diagnostic tool in Streamlit.

---

### Slide 2: Dataset, Preprocessing & Feature Engineering
- **Dataset:** IBM Telco Churn (7,043 rows, 21 attributes).
- **Data Cleaning:** Imputed 11 blank strings in `TotalCharges` with 0.0 (where tenure=0) and handled categorical features.
- **Domain Feature Engineering:**
  - `tenure_group`: Discretized tenure into 5 lifecycle brackets (0-12m to >60m).
  - `avg_monthly_spend`: Historical spend intensity (`TotalCharges / (tenure + 1)`).
  - `num_services`: Total count of active value-added subscriptions.
  - `has_support_services`: Compound flag for Tech Support and Online Security.
- **Class Imbalance Strategy:** Mitigated 73:27 imbalance using SMOTE and class-weight balancing inside an anti-leakage `Pipeline`.

---

### Slide 3: Supervised Model Evaluation & Benchmarks
- **Algorithms Compared:**
  1. *Logistic Regression:* Baseline interpretable linear model.
  2. *Support Vector Machine (SVM):* RBF kernel capturing non-linear boundaries.
  3. *k-Nearest Neighbors (k-NN):* Sensitivity analysis across $k \in [3, 21]$.
  4. *Gradient Boosting:* Sequential ensemble boosting minimizing loss residuals.
- **Primary Optimization Metric:** Recall on Churn Class (minimizing False Negatives) + ROC-AUC.
- **Key Performance Results:**
  - Gradient Boosting achieved best ROC-AUC (~0.85) and highest balanced F1 score.
  - Logistic Regression + SMOTE provided top churn recall (~80%).
  - k-NN exhibited optimal performance around $k=11$ to $15$.

---

### Slide 4: Explainability & Churn Drivers (SHAP)
- **Top 5 Factors Driving Customer Churn:**
  1. *Contract Type:* Month-to-month contracts have a 4x higher churn rate than 1- or 2-year contracts.
  2. *Tenure:* Churn risk drops exponentially after the first 12 months.
  3. *Internet Service:* Fiber optic subscribers without tech support show highest attrition due to price/service sensitivity.
  4. *Payment Method:* Electronic check users have significantly higher churn than autopay users.
  5. *Value-Added Services:* Customers with Online Security & Tech Support show 65% lower churn.
- **Explainability Tools:** SHAP Summary plots, Tree Feature Importances, and Logistic Regression Odds Ratios.

---

### Slide 5: Unsupervised Retention Segmentation (K-Means & PCA)
- **Cohort:** Filtered high-risk customers ($P(\text{churn}) \ge 0.50$).
- **Clustering Technique:** K-Means validated using Elbow Curve & Silhouette score ($k=3$), cross-compared with GMM.
- **Actionable Micro-Segments:**
  - **Segment 1 — High-Spend Newcomers:** Short tenure (< 6 months), high monthly charges ($85+), Fiber optic, month-to-month contract.
  - **Segment 2 — Budget-Conscious Streamers:** Moderate tenure, basic internet, streaming add-ons, price sensitive.
  - **Segment 3 — Legacy Dissatisfied Long-Termers:** High total spend, older accounts, unbundled services, lack of digital tech support.
- **Visuals:** 2D PCA cluster projection showing distinct separation of behavioral personas.

---

### Slide 6: Business Impact, Retention Strategy & Deployment
- **Targeted Retention Strategies:**
  - *High-Spend Newcomers:* 20% discount on 1-year contract lock-in + free 60-day tech support onboarding.
  - *Budget-Conscious Streamers:* Entertainment bundle saver discount + autopay \$5 credit.
  - *Legacy Dissatisfied:* Dedicated account manager check-in + complimentary router upgrade.
- **Financial Simulation:**
  - Projected Net Revenue Saved: $\mathbf{>\$50,000 / \text{year}}$ for an average 1,000-subscriber at-risk cohort at 25% intervention success rate.
- **Live Streamlit App:**
  - Real-time customer churn probability & risk classification (Low / Medium / High).
  - Segment assignment and tailored recommendation display.
