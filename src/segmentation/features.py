"""Feature selection and scaling for clustering."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from segmentation import config


def cluster_matrix(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Select the clustering features in a fixed order."""
    cols = features if features is not None else config.CLUSTER_FEATURES
    return df[cols].astype(float)


def scale(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score every feature so no unit (like dollars) dominates the distance."""
    scaler = StandardScaler()
    values = scaler.fit_transform(features)
    return pd.DataFrame(values, columns=features.columns, index=features.index), scaler
