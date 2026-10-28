# Customer Churn Prediction & Retention Segmentation for Telecom Services

> **Academic ML Lab Mini-Project (Practical 10) — Third Year (TY) Computer Engineering**  
> *A Machine Learning pipeline combining Supervised Churn Classification and Unsupervised Behavioral Micro-Segmentation.*

---

## 📌 Executive Summary

In subscription and telecommunications industries, **Customer Acquisition Cost (CAC)** is typically **5x to 7x higher** than **Customer Retention Cost (CRC)**. When customers switch providers (churn), recurring revenues erode rapidly.

This project delivers an end-to-end Machine Learning system that:
1. **Predicts** whether an individual customer will churn using tuned supervised algorithms (Logistic Regression, SVM, k-NN, and Gradient Boosting).
2. **Mitigates Class Imbalance** (~73.5% retain vs ~26.5% churn) via Synthetic Minority Over-sampling Technique (**SMOTE**) strictly within cross-validation folds.
3. **Explains Predictions** using **SHAP (SHapley Additive exPlanations)**, decision tree feature importances, and logistic regression log-odds.
4. **Segments At-Risk Customers** using **K-Means clustering** (validated via Elbow and Silhouette analysis) and **PCA 2D projection** into actionable personas.
5. **Prescribes Targeted Retention Strategies** and models financial ROI (projecting **$130,522.21 net saved / 276.0% ROI** in the optimistic scenario and **$58,617.23 net saved / 169.4% ROI** in the realistic precision-adjusted scenario on 825 active at-risk customers).
6. **Deploys an Interactive Diagnostic Web App** powered by **Streamlit**.

---

## 📂 Project Directory Structure

```plaintext
MlProject/
├── .venv/                         # Virtual environment (Python 3.14)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (7,043 rows, 21 cols)
├── notebooks/
│   └── churn_project.ipynb        # Complete lab notebook executed top-to-bottom
├── models/
│   ├── best_churn_model.joblib    # Serialized Gradient Boosting + SMOTE Pipeline
│   └── retention_cluster_model.joblib # Serialized K-Means + PCA Segmentation
├── outputs/
│   └── figures/                   # 16 High-resolution generated visual plots
│       ├── churn_distribution.png
│       ├── churn_vs_contract.png
│       ├── churn_vs_tenure.png
│       ├── churn_vs_monthly_charges.png
│       ├── churn_vs_internet_service.png
│       ├── churn_vs_payment_method.png
│       ├── correlation_heatmap.png
│       ├── knn_k_effect.png
│       ├── confusion_matrices.png
│       ├── roc_curves_comparison.png
│       ├── threshold_tuning.png
│       ├── gb_feature_importance.png
│       ├── lr_coefficients.png
│       ├── shap_summary.png
│       ├── kmeans_elbow_silhouette.png
│       └── pca_clusters.png
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Cleaning, encoding, scaling, train-test split
│   ├── feature_engineering.py     # Feature creation functions
│   ├── models.py                  # Model pipelines, grid search & evaluation
│   ├── segmentation.py            # K-Means, GMM, PCA & ROI calculator
│   └── utils.py                   # Plotting helpers & feature extractors
├── app.py                         # Streamlit interactive web dashboard
├── requirements.txt               # Project dependencies
├── report_outline.md              # Academic lab report structure & literature survey
├── viva_questions.md              # 15 likely viva questions with comprehensive answers
├── presentation_outline.md        # 6-slide executive deck outline
└── README.md                      # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git (optional)

### 2. Activate Virtual Environment
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A: Run the Streamlit Web Application
Launch the interactive web diagnostic dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.  
You can enter customer demographics, contract details, and services to view:
- Churn Probability Meter (%)
- Risk Classification Badge (**LOW / MEDIUM / HIGH RISK**)
- Assigned Behavioral Persona
- Prescribed Next-Best Retention Action

### Option B: Run or View the Jupyter Notebook
To view or re-run the complete interactive notebook:
```bash
jupyter notebook notebooks/churn_project.ipynb
```
Or execute all 41 cells non-interactively from terminal:
```bash
python run_notebook.py
```

---

## 📊 Experimental Results & Model Benchmarks

Evaluation performed on an unseen holdout test set (Stratified 20% split, $N=1,409$):

| Model | Accuracy | Precision | Recall (Churn Class) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (+ SMOTE)** | 74.17% | 0.5086 | 78.88% | 0.6184 | **0.8443** |
| **Gradient Boosting (+ SMOTE)** | **77.93%** | **0.5714** | 67.38% | 0.6184 | 0.8410 |
| **Support Vector Machine (+ SMOTE)** | 75.80% | 0.5304 | 77.01% | **0.6281** | 0.8277 |
| **k-NN ($K=81$, + SMOTE)** | 69.27% | 0.4592 | **88.77%** | 0.6053 | 0.8353 |

### 🏆 Best Model Justification & Threshold Tuning
- **Empirical Metric Breakdown:**
  - **Logistic Regression (+ SMOTE)** achieved the highest **ROC-AUC (0.8443)**, demonstrating superior global ranking ability across decision thresholds.
  - **Support Vector Machine (+ SMOTE)** achieved the highest default **F1-Score (0.6281)**.
  - **Gradient Boosting (+ SMOTE)** achieved the highest **Accuracy (77.93%)** and highest **Precision (0.5714)** among all models, minimizing wasteful campaign outreach to non-churners.
  - **k-NN ($K=81$, + SMOTE):** An extended hyperparameter search across $K = 3..81$ (odd values) evaluated strictly by 5-fold cross-validation on the training set identified $K=81$ as optimal (Mean CV ROC-AUC: 0.8370). It achieved the highest raw baseline **Recall (88.77%)** on the test set. *(Note: 5-fold CV ROC-AUC plateaus around K=73–81 [0.8359 to 0.8370], so K=81 is near the edge of the search range.)*
- **Decision Threshold Tuning for Deployed Model:**
  The decision threshold was tuned strictly on the training set using 5-fold cross-validation (`cross_val_predict`), completely preventing holdout test set leakage. Holdout test set metrics were evaluated and reported once at the selected threshold:
  - At default cutoff ($0.50$): Recall is 67.38%, Precision is 57.14%, and F1 is 0.6184.
  - At optimal tuned cutoff (**$0.40$**): **Precision drops slightly** from 57.14% to **52.79%** as **recall rises significantly** from 67.38% to **75.94% (+8.56 percentage points)**, achieving the model's peak **F1-Score of 0.6228** (Accuracy: 75.59%).
- **Deployment Rationale:** **Gradient Boosting with a tuned operational threshold of 0.40** is deployed in production (`app.py`). It effectively models complex feature interactions without distance-scaling distortion, maintains superior precision to avoid costly false alarms, achieves ~76% churn recall at the tuned threshold, and provides full interpretability via **SHAP**.

---

## 🔍 Key Findings & Churn Drivers (SHAP & Feature Importance)

From SHAP summary analysis and Tree Feature Importances:
1. **Contract Type (Month-to-Month):** Strongest single predictor of churn. Customers on month-to-month contracts churn at over 4x the rate of annual subscribers.
2. **Customer Tenure:** Churn risk drops exponentially after the first 12 months. Early customer onboarding is the most critical window.
3. **Internet Service (Fiber Optic) & High Monthly Charges:** Fiber optic users with monthly charges between \$70–\$100 experience the highest churn, often driven by price sensitivity and technical support friction.
4. **Support Services:** Subscribers with **Online Security** or **Tech Support** churn at less than half the rate of unassisted users.
5. **Payment Method:** Customers using **Electronic check** churn significantly more than those using automated ACH bank transfer or credit card.

---

## 👥 Unsupervised Customer Segmentation (Active At-Risk Customers)

Retention campaigns only target currently active subscribers (**`Churn == 'No'`**). Scoring all 5,174 active customers using the deployed model pipeline isolates **825 active at-risk subscribers** ($P(\text{Churn}) \ge 0.50$, representing 15.9% of active accounts). 

Clustering with **K-Means ($k=3$)** was chosen for **operational interpretability**: silhouette score mathematically favors $k=2$ (0.4627 vs 0.4352 for $k=3$, which merely splits accounts by high/low tenure), while Gaussian Mixture Model (GMM) BIC reaches its minimum at $k=5$ (-4,740.2, over-fragmenting campaign operations into micro-segments). Choosing $k=3$ cleanly isolates 3 distinct, highly actionable strategic personas.

A verification check confirms that the average monthly spend across the 825 customers in the cluster table exactly matches the ROI calculation at **$71.93/month** ($863.21/year).

| Cluster Persona | Cohort Size (% of at-risk) | Avg Tenure | Avg Monthly Spend | Avg Services | Empirical Characteristics | Targeted Retention Action |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **High-Spend Fiber Newcomers** | 389 (47.2%) | 11.3 mos | $81.75 | 3.94 | 93.1% Fiber optic, 99.0% Month-to-month, moderate streaming (~40%) | 15% discount on 1-year contract lock-in, free 60-day tech support onboarding, proactive network speed verification |
| **Budget-Conscious Newcomers** | 272 (33.0%) | 2.7 mos | $42.48 | 2.06 | Basic DSL/telephony, 42.6% mailed check, 100% Month-to-month, <9% streaming (*not streamers*) | Dedicated onboarding check-in call from customer care, $5/month discount for paperless autopay enrollment (ACH/Credit Card), 1-year contract upgrade offer with entry-tier price-lock protection (no entertainment bundles) |
| **Premium Long-Term Streamers** | 164 (19.9%) | **40.1 mos** | **$97.49** | **5.83** | **87.2% StreamingTV, 85.4% StreamingMovies**, 99.4% Fiber optic, 97.0% Month-to-month | Dedicated VIP account care outreach, complimentary Wi-Fi 6 next-gen router upgrade, $30 annual loyalty account credit |

---

## 💰 Financial ROI Simulation: Optimistic vs. Realistic Scenarios

We model the estimated economic impact of executing targeted retention campaigns on the **825 active at-risk subscribers** ($P(\text{Churn}) \ge 0.50$):
- **Target Active At-Risk Cohort:** 825 active subscribers
- **Average Spend:** \$71.93 / month (\$863.21 / year) — verified identical between ROI model and cluster table
- **Outreach Campaign Operational Cost:** \$25.00 per at-risk customer contacted (\$20,625.00 total)
- **Retention Discount Cost:** 15% annual promotional discount applied to retained users
- **Retention Success Rate:** 25% of contacted churners accept the personalized offer and stay

### Side-by-Side Scenario Comparison

| Campaign Metric | Optimistic Scenario | Realistic Scenario (Precision-Adjusted) |
| :--- | :---: | :---: |
| **Target Active At-Risk Cohort** | 825 customers | 825 customers |
| **Model Precision (True Churner %)** | 100.0% (assumed) | **52.79%** |
| **True Churners in Target Cohort** | 825 customers | 435 customers |
| **Retention Success Rate** | 25% | 25% |
| **Customers Truly Retained** | **206 customers** | **108 customers** ($825 \times 0.5279 \times 0.25$) |
| **Average Annual Spend per User** | $863.21 | $863.21 |
| **Gross Annual Revenue Preserved** | $177,820.25 | $93,226.15 |
| **Outreach Cost ($25/customer)** | $20,625.00 | $20,625.00 |
| **Retention Discount Cost (15%)** | $26,673.04 | $13,983.92 |
| **Total Retention Campaign Cost** | $47,298.04 | $34,608.92 |
| **Net Annual Revenue Saved** | **$130,522.21** | **$58,617.23** |
| **Campaign ROI (Illustrative Estimate)** | **276.0%** | **169.4%** |

*Note: In the realistic scenario, outreach must still contact all 825 flagged subscribers ($20,625 outreach cost), but retention discounts are disbursed only to the 108 customers truly prevented from churning.*

---

## ⚠️ Limitations & Future Scope

### Limitations
1. **IBM Sample Dataset:** The dataset is a cross-sectional snapshot from the IBM Telco Customer Churn sample dataset. It lacks real-time time-series telemetry (e.g., daily network latency curves, data throttling events, dropped call logs, or customer service ticket escalation history).
2. **Probability Scoring on All Active Rows:** Churn propensity was estimated across all active customer records in the database. In production environments, propensity scoring should run against streaming holdouts or rolling out-of-fold windows to guard against in-sample optimism, backed by continuous probability recalibration.
3. **Illustrative ROI Model:** The financial ROI model operates under illustrative assumptions (uniform 25% offer acceptance and flat 15% discount). In practice, customer price elasticity and willingness to renew differ significantly across personas and channels, requiring live randomized controlled A/B testing with untreated control groups.
4. **Cohort Selection vs. Deployment Threshold:** The at-risk customer cohort is selected at probability $\ge 0.50$ while the deployed decision threshold is $0.40$, and the realistic ROI uses the test precision at $0.40$.

### Future Scope
1. **Survival Analysis:** Implementing Cox Proportional Hazards or Kaplan-Meier estimators to predict the exact *expected time-to-churn* rather than a static binary probability.
2. **Real-Time Streaming Pipelines:** Integrating feature stores (e.g. Feast) with Apache Kafka and Spark for sub-second churn scoring during live call center or chat interactions.
3. **Multi-Armed Bandits & Uplift Modeling:** Deploying contextual multi-armed bandits to optimize personalized retention offers dynamically, maximizing Incremental Customer Lifetime Value (CLV).

---

## 🎓 Academic Lab Examination Deliverables

This workspace contains all required materials for evaluation:
- [notebooks/churn_project.ipynb](notebooks/churn_project.ipynb): Executed Jupyter notebook with all markdown analyses, code cells, and diagnostic outputs.
- [report_outline.md](report_outline.md): Comprehensive 9-section academic lab report structure with citations.
- [viva_questions.md](viva_questions.md): 15 likely viva questions with detailed answers.
- [presentation_outline.md](presentation_outline.md): 6-slide presentation deck.
- [app.py](app.py): Interactive Streamlit diagnostic web app.
