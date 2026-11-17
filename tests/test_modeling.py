import numpy as np
import pandas as pd

from segmentation import config
from segmentation.cleaning import clean
from segmentation.features import cluster_matrix, scale
from segmentation.modeling import (
    best_k,
    fit_kmeans,
    name_segments,
    pca_cumulative_variance,
    profile,
)
from segmentation.pipeline import compare, run


def _scaled(raw):
    df, _ = clean(raw)
    scaled, _ = scale(cluster_matrix(df))
    return scaled


def test_scaling_gives_zero_mean_unit_variance(raw):
    scaled = _scaled(raw)
    assert list(scaled.columns) == config.CLUSTER_FEATURES
    assert np.allclose(scaled.mean(), 0, atol=1e-9)
    assert np.allclose(scaled.std(ddof=0), 1, atol=1e-9)


def test_pca_cumulative_variance_ends_at_one(raw):
    cumvar = pca_cumulative_variance(_scaled(raw))
    assert len(cumvar) == 15
    assert np.all(np.diff(cumvar) >= 0)
    assert abs(cumvar[-1] - 1.0) < 1e-9


def test_kmeans_is_deterministic_with_fixed_seed(raw):
    scaled = _scaled(raw)
    a = fit_kmeans(scaled, 2).labels_
    b = fit_kmeans(scaled, 2).labels_
    assert (a == b).all()


def test_segments_named_by_income():
    profiles = pd.DataFrame({"Income": [-0.5, 0.8]}, index=[0, 1])
    assert name_segments(profiles) == {1: "Segment A", 0: "Segment B"}


def test_profile_shape(raw):
    scaled = _scaled(raw)
    labels = fit_kmeans(scaled, 3).labels_
    assert profile(scaled, labels).shape == (3, 15)


def test_best_k_picks_highest_score():
    assert best_k({2: 0.30, 3: 0.25, 4: 0.20}) == 2


def test_pipeline_runs_end_to_end(raw):
    metrics = run(raw)
    assert metrics["rows"] == len(raw)
    assert sum(metrics["segment_sizes"].values()) == len(raw)
    assert metrics["n_features"] == 15


def test_compare_flags_differences():
    ref = {"_source": "x", "k": 2, "sizes": {"A": 922}, "share": 0.92}
    assert compare({"k": 2, "sizes": {"A": 922}, "share": 0.921}, ref) == []
    assert len(compare({"k": 3, "sizes": {"A": 900}, "share": 0.5}, ref)) == 3
