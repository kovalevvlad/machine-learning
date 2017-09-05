import pandas as pd

from keras_predictor import keras_smape
from smape import df_smape, np_smape
import pytest
import numpy as np


def test_zero_for_perfect_prediction():
    predictions = pd.DataFrame({"a": [1., 2., 3.], "b": [1., 2., 3.]})
    score = df_smape(predictions, predictions)
    assert score == pytest.approx(0.0, abs=0.0001)


def test_example():
    predictions = pd.DataFrame({"a": [1., 2., 3.], "b": [1., 2., 3.]})
    truth = pd.DataFrame({"a": [1., 1., 3.], "b": [1., 2., 3.]})
    score = df_smape(truth, predictions)
    assert score == pytest.approx(100 * (1. / (3. / 2.) / 6.), abs=0.0001)


def test_keras_smape():
    truth = np.random.randint(0, 10, size=10000)
    prediction = np.random.randint(0, 10, size=10000)
    expected = np_smape(truth, prediction)
    actual = float(keras_smape(truth, prediction).eval())
    assert actual == pytest.approx(expected, abs=0.01)
