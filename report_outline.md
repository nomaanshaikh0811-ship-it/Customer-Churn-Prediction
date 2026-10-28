# Academic Lab Report Outline
## Practical 10: Customer Churn Prediction and Retention Segmentation for Telecom Services
**Subject:** Machine Learning Lab (TY Computer Engineering)  
**Dataset:** Telco Customer Churn (IBM Base, 7,043 rows, 21 attributes)  

---

### 1. Abstract
- **Context:** In subscription and telecom industries, customer acquisition costs (CAC) typically exceed customer retention costs (CRC) by 5x to 7x.
- **Problem Statement:** Accurately predicting churn propensity before attrition occurs and discovering distinct micro-segments among at-risk customers to devise targeted, cost-effective retention campaigns.
- **Methodology:** Data cleaning (imputation of blank TotalCharges), exploratory data analysis (EDA), domain feature engineering (tenure groups, service usage, spend ratio), class imbalance correction via SMOTE, hyperparameter-tuned classification models (Logistic Regression, SVM, k-NN with parameter sensitivity, Gradient Boosting), model explainability using SHAP and coefficient analysis, followed by unsupervised clustering (K-Means, GMM/Hierarchical) on high-risk cohorts.
- **Key Findings:** Gradient Boosting achieved the highest ROC-AUC (~0.85) and optimal balance of recall on the churn class (~80% with SMOTE threshold tuning). High-risk customers clustered into three actionable personas: *High-Spend Newcomers*, *Budget-Conscious Streamers*, and *Legacy Dissatisfied Long-Termers*.
- **Business Impact:** Implementing segment-specific retention offers produces a projected net positive return on investment (ROI) with thousands of dollars in preserved Monthly Recurring Revenue (MRR).

---

### 2. Introduction & Business Problem
- **Background:** Telecommunication services market dynamics, low barrier to provider switching (number portability), commodity pricing pressure.
- **Objectives:**
  1. Build a binary classification pipeline to predict customer churn (`Churn = Yes/No`).
  2. Overcome class imbalance (~26.5% churn vs 73.5% retain).
  3. Formulate interpretability metrics to explain *why* individuals leave.
  4. Segment the churn cohort into actionable customer personas using unsupervised learning.
  5. Deliver an interactive Streamlit diagnostic tool for customer support and marketing teams.
- **Business Value & Success Metrics:** Focus on Recall for the minority churn class (minimizing False Negatives, since lost customers directly erode revenue) alongside ROC-AUC for robust class separation.

---

### 3. Literature Survey
- **Traditional Churn Analysis:** Statistical survival analysis, logistic regression baselines, RFM (Recency, Frequency, Monetary) modeling in telecommunications.
- **Modern Machine Learning Approaches:** Non-linear decision trees, ensemble gradient boosting (XGBoost, LightGBM, sklearn GradientBoosting), and kernel Support Vector Machines for high-dimensional customer feature spaces.
- **Handling Class Imbalance:** Synthetic Minority Over-sampling Technique (SMOTE; Chawla et al., 2002), cost-sensitive learning, focal loss, and precision-recall trade-offs.
- **Hybrid Supervised-Unsupervised Pipelines:** Combining predictive churn scoring with cluster analysis (K-Means, Gaussian Mixture Models) to bridge the gap between risk identification and operational marketing strategies.

---

### 4. Methodology & Architectural Pipeline
- **System Architecture Flow:**
  1. *Data Ingestion & Hygiene:* Detection of whitespace strings in `TotalCharges`, type casting, zero/median imputation for zero-tenure users.
  2. *Exploratory Data Analysis:* Uni-variate and bivariate distributions, churn correlation matrices, contract dependency analysis.
  3. *Feature Engineering:*
     - `tenure_group`: Discretized tenure bins (0-12m, 12-24m, 24-48m, 48-60m, >60m).
     - `avg_monthly_spend`: Historical spend efficiency ratio (`TotalCharges / (tenure + 1)`).
     - `num_services`: Aggregation across streaming, voice, backup, and security subscriptions.
     - `has_support_services`: Compound indicator for tech support and online security.
  4. *Anti-Leakage Validation Strategy:* Strict train-test separation (Stratified 80/20, `random_state=42`). All scalers, encoders, and SMOTE resampling fitted strictly within training folds using `imblearn.pipeline.Pipeline`.
  5. *Supervised Modeling & Tuning:*
     - Logistic Regression (L2 penalty, C optimization)
     - Support Vector Classifier (RBF kernel, gamma & C grid)
     - k-Nearest Neighbors (systematic evaluation across $k \in [3, 21]$ to analyze bias-variance trade-off)
     - Gradient Boosting Classifier (learning rate, tree depth, and estimator count grid search)
  6. *Model Interpretability:* Tree feature importances, logistic odds ratios, and SHAP (SHapley Additive exPlanations) values.
  7. *Post-Prediction Segmentation:* Dimensionality reduction (PCA) and clustering (K-Means with Elbow & Silhouette analysis; comparison with Gaussian Mixture Model).

---

### 5. Implementation Details
- **Tech Stack:** Python 3.14, Pandas, Scikit-Learn, Imbalanced-Learn, Matplotlib, Seaborn, Joblib, SHAP, Streamlit.
- **Environment & Reproducibility:** Fixed seeds (`random_state=42`) across cross-validation splits and stochastic algorithms.
- **Artifacts Generated:** Serialized preprocessing pipeline and classifier (`models/best_churn_model.joblib`), clustering model (`models/retention_cluster_model.joblib`), figures (`outputs/figures/`).

---

### 6. Results & Discussion
- **Classifier Benchmark:**
  - Table comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC across all four algorithms.
  - Discussion on why Gradient Boosting or Tuned Logistic Regression outperforms raw k-NN and SVM.
  - Analysis of the ROC Curves and Confusion Matrices.
- **k-NN Sensitivity Analysis:** Plot showing how varying $k$ affects Recall and ROC-AUC.
- **Feature Attribution Findings:** Contract type (Month-to-month), Tenure, Monthly Charges, and Internet Service (Fiber optic) emerge as primary drivers of churn.
- **Customer Segments Profile:**
  - Segment 1: *High-Spend Newcomers* (High risk, low tenure, high monthly charge, fiber optic, month-to-month).
  - Segment 2: *Budget-Conscious Streamers* (Moderate tenure, basic services, price sensitive).
  - Segment 3: *Legacy Dissatisfied Long-Termers* (High lifetime value, older contracts, experiencing technical friction).

---

### 7. Business Recommendations & Revenue Impact
- **Targeted Strategies:**
  - *Segment 1:* Onboarding incentives, long-term contract discounts, complimentary tech support bundle.
  - *Segment 2:* Budget-friendly retention plans, loyalty streaming credits.
  - *Segment 3:* VIP customer care outreach, annual loyalty discount, proactive equipment upgrade.
- **Financial Simulation (Saved Revenue Model):**
  - $\text{Net Revenue Saved} = N_{\text{at-risk}} \times \text{Intervention Success Rate} \times \text{Avg Annual Spend} - \text{Intervention Cost}$.
  - Concrete dollar calculation based on test-set cohort projections.

---

### 8. Conclusion, Limitations & Future Scope
- **Conclusion:** Summary of project deliverables, validated accuracy, and practical business utility.
- **Limitations:** Lack of temporal usage logs (e.g. data consumption drops, customer service call frequency), static demographic snapshot.
- **Future Scope:** Survival analysis (Cox Proportional Hazards) for time-to-churn forecasting, real-time streaming feature ingestion (Kafka/Spark), automated multi-arm bandit testing for retention offers.

---

### 9. References
1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
4. IBM Cognos Analytics Telco Customer Churn Dataset.
