"""
Data Preprocessing and Feature Engineering Module
Customer Churn Prediction and Retention Segmentation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the customer churn CSV dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Strip whitespace
    - Handle blank strings in TotalCharges (replace with 0.0 for tenure=0)
    - Drop customerID
    - Convert SeniorCitizen to categorical object
    """
    df_clean = df.copy()

    # Drop customerID if present
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # Fix TotalCharges containing whitespace strings
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"].astype(str).str.strip(),
            errors="coerce"
        )
        # Missing TotalCharges correspond to customers with tenure = 0
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Convert SeniorCitizen (0/1) to readable categorical
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain features:
    1. tenure_group: Discretized customer lifecycle brackets
    2. avg_monthly_spend: Historical spend efficiency TotalCharges / (tenure + 1)
    3. num_services: Count of active services subscribed
    4. has_support_services: Flag for OnlineSecurity or TechSupport
    """
    df_fe = df.copy()

    # 1. Tenure Group
    bins = [-1, 12, 24, 48, 60, np.inf]
    labels = ["0-12m", "13-24m", "25-48m", "49-60m", ">60m"]
    df_fe["tenure_group"] = pd.cut(df_fe["tenure"], bins=bins, labels=labels)

    # 2. Average Monthly Spend ratio
    df_fe["avg_monthly_spend"] = (df_fe["TotalCharges"] / (df_fe["tenure"] + 1)).round(2)

    # 3. Number of Active Services
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    
    count_services = pd.Series(0, index=df_fe.index)
    for col in service_cols:
        if col in df_fe.columns:
            count_services += (df_fe[col] == "Yes").astype(int)
    
    if "InternetService" in df_fe.columns:
        count_services += (df_fe["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)
        
    df_fe["num_services"] = count_services

    # 4. Support Services Flag (Tech Support or Online Security)
    has_tech = (df_fe.get("TechSupport", pd.Series("No", index=df_fe.index)) == "Yes")
    has_sec = (df_fe.get("OnlineSecurity", pd.Series("No", index=df_fe.index)) == "Yes")
    df_fe["has_support_services"] = (has_tech | has_sec).map({True: "Yes", False: "No"})

    return df_fe


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separate feature matrix X and target vector y.
    Encode Churn as binary: Yes -> 1, No -> 0.
    """
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def get_column_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Scales numeric columns with StandardScaler
    - One-hot encodes categorical columns with drop='first' (or handle_unknown='ignore')
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train-test split to maintain class proportions.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
