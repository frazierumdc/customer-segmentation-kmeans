"""End-to-end run: load, clean, scale, check PCA, choose k, fit, profile.

    python -m segmentation.pipeline --data data/raw/marketing_campaign.csv \
        --out results/metrics.json --check results/reference_metrics.json
    python -m segmentation.pipeline --synthetic --out results/demo_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from segmentation import config
from segmentation.cleaning import clean
from segmentation.data import load_customers
from segmentation.features import cluster_matrix, scale
from segmentation.modeling import (
    best_k,
    elbow_inertia,
    fit_kmeans,
    name_segments,
    pca_cumulative_variance,
    profile,
    silhouette_by_k,
)
from segmentation.synthetic import make_customers


def run(raw: pd.DataFrame, k: int | None = None) -> dict[str, Any]:
    """Run the whole analysis and return every number the README reports."""
    df, report = clean(raw)
    scaled, _ = scale(cluster_matrix(df))

    cumvar = pca_cumulative_variance(scaled)
    inertia = elbow_inertia(scaled)
    silhouette = silhouette_by_k(scaled)
    chosen_k = k if k is not None else best_k(silhouette)

    model = fit_kmeans(scaled, chosen_k)
    profiles = profile(scaled, model.labels_)
    names = name_segments(profiles)
    sizes = pd.Series(model.labels_).value_counts()

    return {
        "rows": report.rows,
        "columns": int(raw.shape[1]),
        "duplicate_rows": report.duplicate_rows,
        "missing_income_filled": report.missing_income_filled,
        "income_fill_value": report.income_fill_value,
        "implausible_ages_replaced": report.implausible_ages_replaced,
        "age_fill_value": report.age_fill_value,
        "marital_status_remapped": report.marital_status_remapped,
        "n_features": int(scaled.shape[1]),
        "pca_variance_11_components": round(float(cumvar[config.PCA_COMPONENTS_REPORTED - 1]), 4),
        "elbow_inertia": {str(k_): round(v, 1) for k_, v in inertia.items()},
        "silhouette": {str(k_): round(v, 4) for k_, v in silhouette.items()},
        "k": chosen_k,
        "segment_sizes": {names[int(c)]: int(n) for c, n in sizes.items()},
        "segment_profiles": {
            names[int(c)]: {f: round(float(v), 2) for f, v in row.items()}
            for c, row in profiles.iterrows()
        },
    }


def compare(metrics: dict[str, Any], reference: dict[str, Any], tol: float = 0.015) -> list[str]:
    """List differences between a fresh run and the reference (original) run."""
    problems: list[str] = []
    for key, expected in reference.items():
        if key.startswith("_"):
            continue
        actual = metrics.get(key)
        if isinstance(expected, dict) and isinstance(actual, dict):
            problems += [f"{key}.{p}" for p in compare(actual, expected, tol)]
        elif isinstance(expected, float) and isinstance(actual, int | float):
            if abs(actual - expected) > tol:
                problems.append(f"{key}: expected {expected}, got {actual}")
        elif actual != expected:
            problems.append(f"{key}: expected {expected}, got {actual}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Customer segmentation pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="path to the tab-separated customer file")
    source.add_argument("--synthetic", action="store_true", help="use generated demo data")
    parser.add_argument("--k", type=int, default=None, help="force k (default: best silhouette)")
    parser.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--check", type=Path, help="reference metrics to compare against")
    args = parser.parse_args(argv)

    raw = make_customers() if args.synthetic else load_customers(args.data)
    metrics = run(raw, k=args.k)
    if args.synthetic:
        metrics["_note"] = "Synthetic demo data. These numbers say nothing about real customers."

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("rows", "k", "segment_sizes")}, indent=2))
    print(f"Wrote {args.out}")

    if args.check:
        problems = compare(metrics, json.loads(args.check.read_text()))
        if problems:
            print("Differences from the reference run:\n  " + "\n  ".join(problems))
            return 1
        print("Matches the reference run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
