import pandas as pd
from smape import smape
import pytest


def test_zero_for_perfect_prediction():
    predictions = pd.DataFrame({"a": [1., 2., 3.], "b": [1., 2., 3.]})
    score = smape(predictions, predictions)
    assert score == pytest.approx(0.0, abs=0.0001)


def test_example():
    predictions = pd.DataFrame({"a": [1., 2., 3.], "b": [1., 2., 3.]})
    truth = pd.DataFrame({"a": [1., 1., 3.], "b": [1., 2., 3.]})
    score = smape(truth, predictions)
    assert score == pytest.approx(100 * (1. / (3. / 2.) / 6.), abs=0.0001)
