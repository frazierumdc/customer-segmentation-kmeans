import pandas as pd
import pytest

from segmentation.synthetic import make_customers


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_customers(n=300, seed=7)
