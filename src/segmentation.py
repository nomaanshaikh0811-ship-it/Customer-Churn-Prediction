"""
Unsupervised Retention Segmentation Module
K-Means, GMM, PCA Profiling & Business Personas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def find_optimal_clusters(X_scaled: np.ndarray, k_range=range(2, 7)):
    """
    Calculate Inertia (Elbow method) and Silhouette Scores for K-Means across k_range.
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, kmeans.labels_)
        silhouettes.append(sil)

    return list(k_range), inertias, silhouettes


def fit_gmm_comparison(X_scaled: np.ndarray, k_range=range(2, 7)):
    """
    Fit Gaussian Mixture Models (EM) to compute BIC and AIC across k_range.
    """
    bics = []
    aics = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))
        aics.append(gmm.aic(X_scaled))
    return list(k_range), bics, aics


def perform_segmentation(df_high_risk: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Fit K-Means clustering on numeric features of high-risk customers.
    Reduce to 2D using PCA for visualization.
    Returns clustered dataframe and trained models.
    """
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_services"]
    available_cols = [c for c in numeric_cols if c in df_high_risk.columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_high_risk[available_cols])

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA 2D
    pca = PCA(n_components=2, random_state=random_state)
    coords_2d = pca.fit_transform(X_scaled)

    df_result = df_high_risk.copy()
    df_result["Cluster"] = clusters
    df_result["PCA1"] = coords_2d[:, 0]
    df_result["PCA2"] = coords_2d[:, 1]

    return df_result, kmeans, scaler, pca, available_cols


def profile_clusters(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cluster profiles across key behavioral and monetary dimensions.
    """
    metrics = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_services"]
    available_metrics = [m for m in metrics if m in df_clustered.columns]

    profile = df_clustered.groupby("Cluster")[available_metrics].mean().round(2)
    profile["Count"] = df_clustered.groupby("Cluster").size()
    profile["Percentage"] = (profile["Count"] / len(df_clustered) * 100).round(1)

    return profile


def calculate_retention_roi(
    n_at_risk: int,
    avg_monthly_charges: float,
    intervention_success_rate: float = 0.25,
    cost_per_intervention: float = 20.0
):
    """
    Calculate annual revenue saved from retention actions:
    Net Revenue Saved = (Retained Customers * Annual Spend) - (Total Campaign Cost)
    """
    retained_customers = int(n_at_risk * intervention_success_rate)
    annual_revenue_per_user = avg_monthly_charges * 12
    gross_revenue_saved = retained_customers * annual_revenue_per_user
    campaign_cost = n_at_risk * cost_per_intervention
    net_revenue_saved = gross_revenue_saved - campaign_cost
    roi_percentage = (net_revenue_saved / campaign_cost * 100) if campaign_cost > 0 else 0

    return {
        "At_Risk_Cohort_Size": n_at_risk,
        "Retained_Customers": retained_customers,
        "Gross_Revenue_Saved": round(gross_revenue_saved, 2),
        "Campaign_Cost": round(campaign_cost, 2),
        "Net_Revenue_Saved": round(net_revenue_saved, 2),
        "ROI_Percentage": round(roi_percentage, 1)
    }
