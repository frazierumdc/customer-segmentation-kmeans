"""Data cleaning: each rule is a small, testable function, and every fix is counted."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from segmentation import config


@dataclass
class CleaningReport:
    """What the cleaning step changed, so the numbers can be audited."""

    rows: int = 0
    duplicate_rows: int = 0
    missing_income_filled: int = 0
    income_fill_value: float = float("nan")
    implausible_ages_replaced: int = 0
    age_fill_value: int = 0
    marital_status_remapped: dict[str, int] = field(default_factory=dict)
    marital_status_mode: str = ""


def fill_missing_income(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Fill missing income with the median: income has extreme outliers that drag the mean."""
    out = df.copy()
    median = float(out["Income"].median())
    report.missing_income_filled = int(out["Income"].isna().sum())
    report.income_fill_value = median
    out["Income"] = out["Income"].fillna(median)
    return out


def add_age(df: pd.DataFrame, reference_year: int = config.REFERENCE_YEAR) -> pd.DataFrame:
    """Replace birth year with age at a fixed reference year."""
    out = df.copy()
    out["Age"] = reference_year - out["Year_Birth"]
    return out.drop(columns=["Year_Birth"])


def add_tenure_days(df: pd.DataFrame, reference_date: str) -> pd.DataFrame:
    """Days between enrollment and a fixed reference date (exploration only; not clustered)."""
    out = df.copy()
    enrolled = pd.to_datetime(out["Dt_Customer"], format="%d-%m-%Y")
    out["Dt_Customer"] = enrolled
    out["Days_Customer"] = (pd.Timestamp(reference_date) - enrolled).dt.days
    return out


def fix_implausible_ages(
    df: pd.DataFrame, report: CleaningReport, max_age: int = config.MAX_PLAUSIBLE_AGE
) -> pd.DataFrame:
    """Replace impossible ages with the rounded mean age of the valid records.

    Keeping the rows keeps their spending data, which is what the clustering uses most.
    """
    out = df.copy()
    bad = out["Age"] > max_age
    mean_age = round(float(out.loc[~bad, "Age"].mean()))
    report.implausible_ages_replaced = int(bad.sum())
    report.age_fill_value = mean_age
    out.loc[bad, "Age"] = mean_age
    return out


def normalize_marital_status(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Map "Alone" to Single, then map joke values to the most common status."""
    out = df.copy()
    to_single = out["Marital_Status"].isin(config.MARITAL_STATUS_TO_SINGLE)
    report.marital_status_remapped["to Single"] = int(to_single.sum())
    out.loc[to_single, "Marital_Status"] = "Single"

    mode = str(out["Marital_Status"].mode()[0])
    invalid = out["Marital_Status"].isin(config.MARITAL_STATUS_INVALID)
    report.marital_status_remapped[f"to {mode}"] = int(invalid.sum())
    report.marital_status_mode = mode
    out.loc[invalid, "Marital_Status"] = mode
    return out


def clean(
    df: pd.DataFrame, tenure_reference_date: str = "2026-01-01"
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run every cleaning rule in order and return the cleaned frame plus a report."""
    report = CleaningReport(rows=len(df), duplicate_rows=int(df.duplicated().sum()))
    out = fill_missing_income(df, report)
    out = add_tenure_days(out, tenure_reference_date)
    out = add_age(out)
    out = normalize_marital_status(out, report)
    out = fix_implausible_ages(out, report)
    return out, report
