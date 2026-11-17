"""Generate a fake customer file with the real schema.

Used by the tests and by `make demo`, so the code can run end to end without the real
dataset. The numbers it produces mean nothing about real customers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from segmentation.data import REQUIRED_COLUMNS


def make_customers(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    affluent = rng.random(n) < 0.4  # two loose groups so clustering has something to find
    income = np.where(affluent, rng.normal(75_000, 12_000, n), rng.normal(35_000, 10_000, n))
    kids = np.where(affluent, rng.integers(0, 2, n), rng.integers(0, 3, n))
    spend_scale = np.where(affluent, 1.0, 0.2)

    df = pd.DataFrame(
        {
            "ID": np.arange(n),
            "Year_Birth": rng.integers(1945, 2000, n),
            "Education": rng.choice(["Graduation", "PhD", "Master", "2n Cycle", "Basic"], n),
            "Marital_Status": rng.choice(["Married", "Together", "Single", "Divorced"], n),
            "Income": income.round(0),
            "Kidhome": kids,
            "Teenhome": rng.integers(0, 3, n),
            "Dt_Customer": pd.to_datetime("2012-08-01")
            + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
            "Recency": rng.integers(0, 100, n),
        }
    )
    df["Dt_Customer"] = df["Dt_Customer"].dt.strftime("%d-%m-%Y")
    for col, base in [
        ("MntWines", 600),
        ("MntFruits", 50),
        ("MntMeatProducts", 350),
        ("MntFishProducts", 70),
        ("MntSweetProducts", 50),
        ("MntGoldProds", 70),
    ]:
        df[col] = (rng.gamma(2.0, base / 2, n) * spend_scale).round().astype(int)
    df["NumDealsPurchases"] = rng.integers(0, 6, n)
    df["NumWebPurchases"] = (rng.integers(1, 8, n) * (1 + affluent)).astype(int)
    df["NumCatalogPurchases"] = (rng.integers(0, 4, n) * (1 + 2 * affluent)).astype(int)
    df["NumStorePurchases"] = (rng.integers(2, 7, n) * (1 + affluent)).astype(int)
    df["NumWebVisitsMonth"] = np.where(affluent, rng.integers(1, 5, n), rng.integers(4, 10, n))
    for col in ["AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2"]:
        df[col] = (rng.random(n) < 0.07).astype(int)
    df["Complain"] = (rng.random(n) < 0.01).astype(int)
    df["Z_CostContact"] = 3
    df["Z_Revenue"] = 11
    df["Response"] = (rng.random(n) < 0.15).astype(int)

    # Plant the same data problems the real file has.
    df.loc[df.sample(frac=0.01, random_state=seed).index, "Income"] = np.nan
    df.loc[[0, 1], "Year_Birth"] = [1893, 1899]
    df.loc[[2], "Marital_Status"] = "Alone"
    df.loc[[3, 4], "Marital_Status"] = ["YOLO", "Absurd"]
    return df[REQUIRED_COLUMNS]
