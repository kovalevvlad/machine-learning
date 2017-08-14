import pandas as pd
from math_util import rolling_harmonic_mean
import numpy as np


def test_rolling_harmonic_mean():
    hmean = rolling_harmonic_mean(pd.Series([1, 4, 4]), 3)
    expected_result = pd.Series([np.nan, np.nan, 2.])
    assert ((hmean == expected_result) | (hmean.isnull() & expected_result.isnull())).all(), (hmean, expected_result)

    hmean = rolling_harmonic_mean(pd.Series([1, 1, 1]), 3)
    expected_result = pd.Series([np.nan, np.nan, 1.])
    assert ((hmean == expected_result) | (hmean.isnull() & expected_result.isnull())).all(), (hmean, expected_result)
