"""Loading and validating the raw customer file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "ID",
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Dt_Customer",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    "Z_CostContact",
    "Z_Revenue",
    "Response",
]


class SchemaError(ValueError):
    """Raised when the input file does not have the expected columns."""


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing columns: {missing}")


def load_customers(path: str | Path) -> pd.DataFrame:
    """Read the tab-separated customer file and check its columns."""
    df = pd.read_csv(path, sep="\t")
    validate_schema(df)
    return df
