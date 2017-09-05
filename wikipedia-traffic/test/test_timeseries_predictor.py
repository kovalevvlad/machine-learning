import datetime
import pandas as pd
import pytest

from feature_extractor import FeatureExtractor, page_label, date_label, y_label, index_column_names
from pandas_util import safe_reindex
from smape import np_smape
from timeseries_predictor import TimeseriesPredictor
from test_feature_extractor import generate_X
import numpy as np


def test_correct_values_supplied_to_model(disable_parallelism):
    data = generate_X()
    shift_size = 7
    shift_count = 2
    days_to_predict = 60

    class FakeModel:
        def __init__(self):
            self.fit_called = False
            self.predict_call_count = 0

        def fit(self, X, y):
            all_features_and_ys = []
            for i in range(shift_count):
                all_features_and_ys.append(self.expected_X_and_y(i * shift_size))
            features, ys = zip(*all_features_and_ys)
            expected_X = pd.concat(features)
            expected_X = safe_reindex(expected_X, X.index)
            expected_y = pd.concat(ys)
            expected_y = safe_reindex(expected_y, y.index)
            assert (expected_X == X).all().all()
            assert (expected_y == y).all()
            self.fit_called = True

        def expected_X_and_y(self, shift):
            as_of_index = days_to_predict + shift
            training_data = data.iloc[:-as_of_index]
            fe = FeatureExtractor(days_to_predict, disable_parallelism=True)
            expected_features = fe.extract_features(training_data)
            expected_features["shift"] = shift
            new_index_columns = ["shift"] + index_column_names
            expected_features = expected_features.reset_index().set_index(new_index_columns)

            expected_y_df = data.iloc[-as_of_index:-shift] if shift > 0 else data.iloc[-as_of_index:]
            expected_y_df["shift"] = shift
            expected_y = expected_y_df.reset_index().melt(id_vars=[date_label, "shift"], var_name=page_label, value_name=y_label).set_index(new_index_columns)[y_label]

            return expected_features, expected_y

        def predict(self, X):
            # The first call is for training_score estimation - ignore it
            if self.predict_call_count == 1:
                fe = FeatureExtractor(days_to_predict, disable_parallelism=disable_parallelism)
                expected_features = fe.extract_features(data)
                expected_features = safe_reindex(expected_features, X.index)
                assert (X == expected_features).all().all()
            self.predict_call_count += 1
            return np.ones((X.shape[0],))

    model = FakeModel()
    TimeseriesPredictor(days_to_predict, shift_count, shift_size, model, np_smape, disable_parallelism=disable_parallelism).predict(data)
    assert model.fit_called
    assert model.predict_call_count == 2


def test_training_score(disable_parallelism):
    actual_raw_traffic_value = 1.
    X = generate_X(np.tile([actual_raw_traffic_value], 360))

    class FakeModel:
        def __init__(self, prediction_value):
            self.prediction_value = prediction_value

        def fit(self, features, y):
            pass

        def predict(self, features):
            return np.repeat([self.prediction_value], features.shape[0])

    predictor = TimeseriesPredictor(60, 2, 3, FakeModel(actual_raw_traffic_value), np_smape, disable_parallelism=disable_parallelism)
    predictor.predict(X)
    assert predictor.training_score == pytest.approx(0.0, abs=0.0001)

    incorrect_prediction_value = actual_raw_traffic_value + 1.0
    predictor = TimeseriesPredictor(60, 2, 3, FakeModel(incorrect_prediction_value), np_smape, disable_parallelism=disable_parallelism)
    predictor.predict(X)
    expected_error = np_smape(np.array([actual_raw_traffic_value]), np.array([incorrect_prediction_value]))
    assert predictor.training_score == pytest.approx(expected_error, rel=0.0001)
