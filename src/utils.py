"""
Utility functions for plotting, serialization and metric formatting.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_figure(fig, filename: str, output_dir: str = "outputs/figures", dpi: int = 300):
    """
    Save matplotlib figure to output directory with high resolution and tight layout.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"[Saved Figure] {full_path}")


def get_feature_names_from_preprocessor(preprocessor, X: pd.DataFrame) -> list:
    """
    Extract transformed feature names from ColumnTransformer.
    Works reliably across scikit-learn versions.
    """
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback reconstruction
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        ohe = preprocessor.named_transformers_["cat"]
        cat_encoded_names = ohe.get_feature_names_out(cat_features).tolist()

        return numeric_features + cat_encoded_names
