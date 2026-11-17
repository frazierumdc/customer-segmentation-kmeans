"""PCA check, choosing k, final K-means fit, and cluster profiles."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from segmentation import config


def pca_cumulative_variance(scaled: pd.DataFrame, seed: int = config.SEED) -> np.ndarray:
    """Cumulative explained-variance ratio for 1..n components."""
    pca = PCA(n_components=scaled.shape[1], random_state=seed).fit(scaled)
    return np.cumsum(pca.explained_variance_ratio_)


def elbow_inertia(
    scaled: pd.DataFrame, ks: Iterable[int] = config.K_RANGE_ELBOW, seed: int = config.SEED
) -> dict[int, float]:
    """Within-cluster sum of squares for each k."""
    return {
        k: float(
            KMeans(n_clusters=k, random_state=seed, n_init="auto", max_iter=1000)
            .fit(scaled)
            .inertia_
        )
        for k in ks
    }


def silhouette_by_k(
    scaled: pd.DataFrame, ks: Iterable[int] = config.K_RANGE_SILHOUETTE, seed: int = config.SEED
) -> dict[int, float]:
    """Mean silhouette score for each k (higher means better-separated clusters)."""
    scores: dict[int, float] = {}
    for k in ks:
        labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(scaled)
        scores[k] = float(silhouette_score(scaled, labels))
    return scores


def best_k(scores: dict[int, float]) -> int:
    return max(scores, key=lambda k: scores[k])


def fit_kmeans(scaled: pd.DataFrame, k: int, seed: int = config.SEED) -> KMeans:
    """Final model, with the settings of the original run."""
    return KMeans(n_clusters=k, random_state=seed, max_iter=1000, n_init="auto").fit(scaled)


def profile(scaled: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean scaled value of every feature per cluster (0 = the average customer)."""
    return scaled.assign(cluster=labels).groupby("cluster").mean()


def name_segments(profiles: pd.DataFrame) -> dict[int, str]:
    """Name clusters by income so labels stay stable even if K-means numbers them differently."""
    order = profiles["Income"].sort_values(ascending=False).index.tolist()
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    return {int(cluster): f"Segment {names[i]}" for i, cluster in enumerate(order)}
