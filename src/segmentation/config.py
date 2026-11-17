"""Constants that define the analysis. Changing any of these changes the results."""

from __future__ import annotations

SEED: int = 1

# The original run computed age as REFERENCE_YEAR - Year_Birth. A fixed year keeps
# results reproducible; using "today" would shift every age each January.
REFERENCE_YEAR: int = 2026
MAX_PLAUSIBLE_AGE: int = 100

# Marital-status values that are not real categories in this dataset.
MARITAL_STATUS_TO_SINGLE: tuple[str, ...] = ("Alone",)
MARITAL_STATUS_INVALID: tuple[str, ...] = ("Absurd", "YOLO")

SPENDING_COLUMNS: list[str] = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
CHANNEL_COLUMNS: list[str] = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

# The 15 features used for clustering, in the order of the original run.
# Recency and tenure were left out: they showed almost no correlation with anything else.
CLUSTER_FEATURES: list[str] = [
    "Age",
    "Income",
    *SPENDING_COLUMNS,
    "NumDealsPurchases",
    *CHANNEL_COLUMNS,
    "Kidhome",
    "Teenhome",
]

# Constant columns with no information.
CONSTANT_COLUMNS: list[str] = ["Z_CostContact", "Z_Revenue"]

K_RANGE_ELBOW: range = range(1, 10)
K_RANGE_SILHOUETTE: range = range(2, 10)
PCA_COMPONENTS_REPORTED: int = 11
