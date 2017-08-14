import pandas as pd

from feature_extractor import FeatureExtractor, page_label, date_label, y_label, index_column_names
from pandas_util import safe_reindex
from timeseries_predictor import TimeseriesPredictor
from test_feature_extractor import generate_X
import numpy as np


def test_correct_values_supplied_to_model():
    data = generate_X()
    shift_size = 7
    shift_count = 2
    days_to_predict = 60

    class FakeModel:
        def __init__(self):
            self.fit_called = False
            self.predict_called = False

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
            expected_features, norm_log_median = fe.extract_features(training_data)
            expected_features["shift"] = shift
            new_index_columns = ["shift"] + index_column_names
            expected_features = expected_features.reset_index().set_index(new_index_columns)

            y_data = data.iloc[-as_of_index:-shift] if shift > 0 else data.iloc[-as_of_index:]
            expected_y_df = np.log(y_data + 1) - norm_log_median
            expected_y_df["shift"] = shift
            expected_y = expected_y_df.reset_index().melt(id_vars=[date_label, "shift"], var_name=page_label, value_name=y_label).set_index(new_index_columns)[y_label]

            return expected_features, expected_y

        def predict(self, X):
            fe = FeatureExtractor(days_to_predict, disable_parallelism=True)
            expected_features, norm_log_median = fe.extract_features(data)
            expected_features = safe_reindex(expected_features, X.index)
            assert (X == expected_features).all().all()
            self.predict_called = True

    model = FakeModel()
    TimeseriesPredictor(days_to_predict, shift_count, shift_size, model, disable_parallelism=True).predict(data)
    assert model.fit_called
    assert model.predict_called
