import logging
import pandas as pd
from sklearn.metrics import make_scorer

from smape import np_smape
from feature_extractor import FeatureExtractor, page_label, date_label, index_column_names, y_label


feature_cache = dict()


class TimeseriesPredictor(object):
    def __init__(self, days_to_predict, training_shifts, shift_size, model, scorer, one_hot=True, disable_parallelism=False, use_feature_cache=True):
        self.days_to_predict = days_to_predict
        self.training_shifts = training_shifts
        self.shift_size = shift_size
        self.model = model
        self.one_hot = one_hot
        self.disable_parallelism = disable_parallelism
        self.scorer = scorer
        self.use_feature_cache = use_feature_cache

    def predict(self, X, **fit_args):
        cache_key = (tuple(X.index.values), tuple(X.columns.values), self.training_shifts, self.shift_size, self.one_hot)
        if not cache_key in feature_cache.keys():
            feature_extractor = FeatureExtractor(self.days_to_predict,
                                                 one_hot=self.one_hot,
                                                 disable_parallelism=self.disable_parallelism)
            train_X, train_y = self.training_features(X, feature_extractor)
            test_X = feature_extractor.extract_features(X)
            if self.use_feature_cache:
                feature_cache[cache_key] = (train_X, train_y, test_X)
        else:
            train_X, train_y, test_X = feature_cache[cache_key]
            logging.debug("Feature cache hit!")

        self.fitting_result = self.model.fit(train_X, train_y, **fit_args)
        self.training_score = self._training_score(train_X, train_y)
        self.feature_names = train_X.columns
        predictions = self.model.predict(test_X)
        if len(predictions .shape) != 1:
            predictions = predictions.reshape((-1,))
        prediction_label = "prediction"
        prediction_series = pd.Series(predictions, index=test_X.index).to_frame(prediction_label)
        prediction_df = prediction_series.reset_index().pivot(index=date_label, columns=page_label, values=prediction_label)
        assert set(prediction_df.columns) == set(X.columns)
        prediction_df = prediction_df.reindex(columns=X.columns)
        return prediction_df

    def training_features(self, X, feature_extractor):
        train_data = []
        last_date = X.index[-1]
        for shift in range(self.training_shifts):
            asof = last_date - pd.Timedelta(days=shift * self.shift_size)
            features, y = feature_extractor.features_with_y(X[:asof])
            features["shift"] = shift * self.shift_size
            # Making the index unique for testing
            new_index_columns = ["shift"] + index_column_names
            features = features.reset_index().set_index(new_index_columns)
            y = pd.DataFrame({y_label: y, "shift": shift * self.shift_size}).reset_index().set_index(new_index_columns)[y_label]
            train_data.append((features, y))
        train_Xs, transformed_ys = zip(*train_data)
        train_X = pd.concat(train_Xs)
        train_y = pd.concat(transformed_ys)
        return train_X, train_y

    def _training_score(self, train_X, train_y):
        predictions = self.model.predict(train_X)
        if len(predictions.shape) > 1:
            predictions = predictions.reshape((-1, ))
        return np_smape(train_y.values, predictions)
