"""
Supervised Model Training, Hyperparameter Tuning & Evaluation
Customer Churn Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def get_model_pipelines(preprocessor, use_smote: bool = True, random_state: int = 42):
    """
    Construct model pipelines with preprocessor, SMOTE (to prevent data leakage),
    and supervised classifiers.
    """
    pipelines = {}

    # 1. Logistic Regression (Baseline)
    if use_smote:
        pipelines["Logistic_Regression"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state))
        ])
    else:
        pipelines["Logistic_Regression"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))
        ])

    # 2. Support Vector Machine
    if use_smote:
        pipelines["SVM"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", SVC(probability=True, random_state=random_state))
        ])
    else:
        pipelines["SVM"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(probability=True, class_weight="balanced", random_state=random_state))
        ])

    # 3. k-Nearest Neighbors
    if use_smote:
        pipelines["k-NN"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", KNeighborsClassifier())
        ])
    else:
        pipelines["k-NN"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier())
        ])

    # 4. Gradient Boosting
    if use_smote:
        pipelines["Gradient_Boosting"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", GradientBoostingClassifier(random_state=random_state))
        ])
    else:
        pipelines["Gradient_Boosting"] = ImbPipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=random_state))
        ])

    return pipelines


def get_hyperparameter_grids():
    """
    Define hyperparameter grids for tuning with cross-validation.
    """
    param_grids = {
        "Logistic_Regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"]
        },
        "SVM": {
            "classifier__C": [0.1, 1.0, 5.0],
            "classifier__kernel": ["rbf"]
        },
        "k-NN": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "classifier__weights": ["uniform", "distance"]
        },
        "Gradient_Boosting": {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 4]
        }
    }
    return param_grids


def evaluate_knn_k_effect(X_train, y_train, X_test, y_test, preprocessor, k_values=None, random_state=42):
    """
    Study the effect of K on k-NN model performance (Recall, Precision, ROC-AUC, Accuracy).
    Returns a DataFrame of results.
    """
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 15, 21, 25]

    results = []
    for k in k_values:
        pipe = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", KNeighborsClassifier(n_neighbors=k))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "k": k,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

    return pd.DataFrame(results)


def evaluate_model_performance(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate fitted models on test set and return summary comparison table.
    """
    records = []
    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        records.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
        })

    df_results = pd.DataFrame(records).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return df_results
