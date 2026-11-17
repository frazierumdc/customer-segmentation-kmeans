import numpy as np
import pandas as pd
import pytest

from segmentation.cleaning import (
    CleaningReport,
    add_age,
    clean,
    fill_missing_income,
    fix_implausible_ages,
    normalize_marital_status,
)
from segmentation.data import SchemaError, validate_schema


def test_income_filled_with_median_not_mean():
    df = pd.DataFrame({"Income": [10.0, 20.0, 30.0, 1_000_000.0, np.nan]})
    report = CleaningReport()
    out = fill_missing_income(df, report)
    assert out["Income"].isna().sum() == 0
    assert out.loc[4, "Income"] == 25.0  # median of the four known values
    assert report.missing_income_filled == 1


def test_age_uses_fixed_reference_year():
    out = add_age(pd.DataFrame({"Year_Birth": [1970, 2000]}), reference_year=2026)
    assert out["Age"].tolist() == [56, 26]
    assert "Year_Birth" not in out.columns


def test_implausible_ages_replaced_with_rounded_valid_mean():
    df = pd.DataFrame({"Age": [40, 50, 61, 133]})
    report = CleaningReport()
    out = fix_implausible_ages(df, report, max_age=100)
    assert out["Age"].tolist() == [40, 50, 61, 50]
    assert report.implausible_ages_replaced == 1
    assert report.age_fill_value == 50


def test_marital_status_mapping():
    statuses = ["Married", "Married", "Single", "Alone", "YOLO", "Absurd", "Together"]
    df = pd.DataFrame({"Marital_Status": statuses})
    report = CleaningReport()
    out = normalize_marital_status(df, report)
    assert out["Marital_Status"].tolist() == [
        "Married", "Married", "Single", "Single", "Married", "Married", "Together"
    ]  # fmt: skip
    assert report.marital_status_remapped == {"to Single": 1, "to Married": 2}


def test_clean_leaves_no_gaps_or_bad_values(raw):
    out, report = clean(raw)
    assert len(out) == len(raw) == report.rows
    assert out["Income"].isna().sum() == 0
    assert out["Age"].max() <= 100
    assert not out["Marital_Status"].isin(["Alone", "YOLO", "Absurd"]).any()
    assert report.implausible_ages_replaced == 2


def test_schema_check_rejects_missing_columns(raw):
    with pytest.raises(SchemaError):
        validate_schema(raw.drop(columns=["Income"]))
