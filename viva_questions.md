# Machine Learning Lab Viva Questions & Answers
## Subject: Customer Churn Prediction and Retention Segmentation (Practical 10)

---

### Q1. What is customer churn, and why is it important in business?
**Answer:** Customer churn is the rate at which customers discontinue their subscription or service with a company over a given timeframe. It is vital because acquiring a new customer costs 5 to 7 times more than retaining an existing one. Reducing churn directly protects recurring revenue and customer lifetime value (CLV).

---

### Q2. Why is raw Accuracy a misleading metric for churn prediction?
**Answer:** The dataset is imbalanced (~73.5% retain, ~26.5% churn). A naive model that simply predicts "No Churn" for every customer would achieve 73.5% accuracy but catch 0 churners (0% recall). Therefore, metrics like **Recall**, **Precision**, **F1-Score**, and **ROC-AUC** are much more informative.

---

### Q3. Why is Recall on the Churn class prioritized over Precision in this problem?
**Answer:** In telecom retention, a **False Negative** (failing to identify an actual churner) means losing the customer and their entire lifetime revenue. A **False Positive** (offering a discount/call to a customer who wasn't going to leave) only costs a minor promotion expense. Thus, minimizing False Negatives (maximizing Recall) is prioritized.

---

### Q4. What data quality issue was present in the `TotalCharges` column, and how was it handled?
**Answer:** In 11 rows where `tenure == 0` (brand-new customers), the `TotalCharges` field contained whitespace strings (`" "`). This prevented numerical parsing. It was handled by converting whitespace strings to `NaN`, followed by replacing them with `0.0` (consistent with zero tenure) and casting the column to `float64`.

---

### Q5. What is data leakage, and how did you prevent it during preprocessing?
**Answer:** Data leakage occurs when information from outside the training dataset (such as the test set) is used to fit the model or preprocessors, leading to overly optimistic evaluation metrics. To prevent leakage:
1. The train-test split was performed first (Stratified 80/20).
2. All encoders, imputers, scalers, and SMOTE resampling were fitted **strictly on the training data** using `imblearn.pipeline.Pipeline`.

---

### Q6. What is SMOTE, and when should it be applied?
**Answer:** SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples of the minority class by interpolating feature values between nearest neighbors of minority samples. It must be applied **only to the training data** inside cross-validation folds, never to the test or validation datasets.

---

### Q7. What feature engineering steps did you implement, and why?
**Answer:**
1. `tenure_group`: Grouped customer tenure into discrete intervals (e.g. 0-12m, 13-24m, etc.) to capture non-linear customer lifecycle risk.
2. `avg_monthly_spend`: `TotalCharges / (tenure + 1)` to measure historical spend velocity relative to current monthly charges.
3. `num_services`: Sum of active services (internet, security, streaming, backup, phone) as higher service stickiness reduces churn.
4. `has_support_services`: Flag indicating presence of Tech Support or Online Security, which are known to enhance customer retention.

---

### Q8. What four supervised algorithms were evaluated, and how do their mechanics differ?
**Answer:**
1. **Logistic Regression:** Linear probabilistic baseline modeling log-odds via sigmoid function.
2. **Support Vector Machine (SVM):** Finds an optimal separating hyperplane using an RBF kernel to project features into higher dimensions.
3. **k-Nearest Neighbors (k-NN):** Instance-based non-parametric classifier predicting based on majority vote of the $k$ nearest Euclidean points.
4. **Gradient Boosting:** Ensemble of shallow decision trees built sequentially, where each tree corrects the residual errors of prior trees.

---

### Q9. How does the choice of $k$ affect the k-NN model's performance?
**Answer:**
- Very small $k$ (e.g., $k=1$ or $k=3$) suffers from **high variance / overfitting**, becoming overly sensitive to noise and outliers.
- Large $k$ (e.g., $k \ge 21$) increases smoothing and causes **high bias / underfitting**, diluting minority class boundaries.
- Hyperparameter tuning helps locate the sweet spot (typically $k=7$ to $15$) balancing sensitivity and generalization.

---

### Q10. What is the ROC curve, and what does the ROC-AUC score represent?
**Answer:** The Receiver Operating Characteristic (ROC) curve plots True Positive Rate (Recall) vs False Positive Rate at all classification thresholds. The Area Under the Curve (ROC-AUC) represents the probability that the classifier will rank a randomly chosen positive (churner) higher than a randomly chosen negative (non-churner). A score of 0.5 is random guessing, while 1.0 is perfect discrimination.

---

### Q11. What is SHAP, and how does it explain model predictions?
**Answer:** SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It calculates the marginal contribution of each feature to the model's prediction for an individual instance or globally across the dataset, ensuring fair and consistent feature attribution regardless of model complexity.

---

### Q12. What were the top drivers of customer churn identified by the models?
**Answer:**
1. **Contract Type:** Month-to-month contracts have the highest churn propensity.
2. **Tenure:** Shorter tenure strongly correlates with higher churn.
3. **Monthly Charges & Fiber Optic:** High monthly bills, particularly on fiber optic plans without tech support, exhibit high churn.
4. **Payment Method:** Electronic check payments show significantly higher churn compared to automatic bank transfers or credit cards.

---

### Q13. Why did you use unsupervised clustering after supervised classification?
**Answer:** Supervised classification only flags *who* is at risk (binary probability). Unsupervised clustering groups those at-risk customers into distinct behavioral segments (e.g., price-sensitive streamers vs dissatisfied high-spenders), allowing marketing teams to design tailored, differentiated retention campaigns rather than generic blanket offers.

---

### Q14. How did you choose the optimal number of clusters for K-Means?
**Answer:** We evaluated two complementary methods:
1. **Elbow Method:** Plotting Inertia (Within-Cluster Sum of Squares) against $k$ and looking for the bend where marginal inertia drop diminishes.
2. **Silhouette Score:** Measuring how similar an object is to its own cluster compared to other clusters (range -1 to +1); the highest silhouette score indicates well-separated clusters.

---

### Q15. How do you calculate the estimated business revenue saved by this ML solution?
**Answer:** Using the ROI formula:
$$\text{Net Revenue Saved} = (N_{\text{at-risk}} \times \text{Intervention Success Rate} \times \text{Avg Monthly Charge} \times 12) - \text{Campaign Cost}$$
For example, if 300 high-risk customers are identified, and targeted offers save 25% of them (75 customers) with an average annual spend of \$850 each, the gross preserved revenue is \$63,750. Subtracting campaign costs (\$10,000) yields a net annual savings of \$53,750.
