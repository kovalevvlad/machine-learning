import numpy as np
import pandas as pd

from math_util import assert_no_nulls, assert_no_infs
from pandas_util import safe_reindex


def df_smape(ground_truth, predictions):
    """
    As defined at https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    predictions = safe_reindex(predictions, ground_truth.index)
    assert ground_truth.shape == predictions.shape
    assert (ground_truth.index == predictions.index).all()
    assert (ground_truth.columns == predictions.columns).all()

    assert_no_nulls(predictions)
    assert_no_infs(predictions)
    assert_no_infs(ground_truth)

    width, height = ground_truth.shape
    n = width * height

    denominator = ground_truth + predictions.abs()
    summation_parts = (ground_truth - predictions).abs() / denominator
    # As defined by the rules, whenever both ground_truth and predictions are 0 (hence denominator is 0)
    # we define SMAPE to be 0
    summation_parts = summation_parts.replace(np.inf, 0).fillna(0)

    assert_no_nulls(summation_parts)
    assert_no_infs(summation_parts)

    # Ground truth can be nan since sometimes we use the training timeseries themselves,
    # which do have nans. Hence denominator can be nan. A good way to deal with these is to ignore them by
    # setting the error = 0 in such cases and adjusting n
    adjusted_n = n - denominator.isnull().sum().sum()
    return 100 * 2 * summation_parts.sum().sum() / adjusted_n


def np_smape(ground_truth, predictions):
    """
    As defined at https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    # Assert 1D arrays
    assert len(ground_truth.shape) == 1
    assert len(predictions.shape) == 1
    return df_smape(pd.DataFrame({"data": ground_truth}), pd.DataFrame({"data": predictions}))
