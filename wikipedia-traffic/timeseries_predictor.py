import logging
import pandas as pd
from sklearn.metrics import make_scorer

from smape import np_smape
from feature_extractor import FeatureExtractor, page_label, date_label, index_column_names, y_label


def transform_smape(normalizing_log_median):
    # Minus sign because scikit maximises the score
    def smape_scorer(log_pred, log_truth):
        return np_smape(FeatureExtractor.inverse_y_transform(log_truth, normalizing_log_median),
                        FeatureExtractor.inverse_y_transform(log_pred, normalizing_log_median))

    return make_scorer(smape_scorer, greater_is_better=False)


feature_cache = dict()


class TimeseriesPredictor(object):
    def __init__(self, days_to_predict, training_shifts, shift_size, model, scorer, one_hot=True, disable_parallelism=False, normalizing_period=42, use_feature_cache=True):
        self.days_to_predict = days_to_predict
        self.training_shifts = training_shifts
        self.shift_size = shift_size
        self.model = model
        self.one_hot = one_hot
        self.disable_parallelism = disable_parallelism
        self.normalizing_period = normalizing_period
        self.scorer = scorer
        self.use_feature_cache = use_feature_cache

    def predict(self, X):
        cache_key = (tuple(X.index.values), tuple(X.columns.values), self.training_shifts, self.shift_size, self.normalizing_period, self.one_hot)
        if not cache_key in feature_cache.keys():
            feature_extractor = FeatureExtractor(self.days_to_predict,
                                                 normalizing_period=self.normalizing_period,
                                                 one_hot=self.one_hot,
                                                 disable_parallelism=self.disable_parallelism)
            train_X, train_y, train_norm_log_median_df = self.training_features(X, feature_extractor)
            test_X, test_normalizing_log_medians = feature_extractor.extract_features(X)
            if self.use_feature_cache:
                feature_cache[cache_key] = (train_X, train_y, train_norm_log_median_df, test_X, test_normalizing_log_medians)
        else:
            train_X, train_y, train_norm_log_median_df, test_X, test_normalizing_log_medians = feature_cache[cache_key]
            logging.debug("Feature cache hit!")

        self.model.fit(train_X, train_y)
        self.training_score = self._training_score(train_X, train_y, train_norm_log_median_df)
        self.feature_names = train_X.columns
        normalized_predictions = self.model.predict(test_X)
        prediction_label = "prediction"
        normalized_predictions = pd.Series(normalized_predictions, index=test_X.index).to_frame(prediction_label)
        normalized_predictions = normalized_predictions.reset_index().pivot(index=date_label, columns=page_label, values=prediction_label)
        assert set(normalized_predictions.columns) == set(X.columns)
        normalized_predictions = normalized_predictions.reindex(columns=X.columns)
        return FeatureExtractor.inverse_y_transform(normalized_predictions, test_normalizing_log_medians)

    def training_features(self, X, feature_extractor):
        train_data = []
        last_date = X.index[-1]
        for shift in range(self.training_shifts):
            asof = last_date - pd.Timedelta(days=shift * self.shift_size)
            features, y, normalizing_log_median = feature_extractor.features_with_y(X[:asof])
            normalizing_log_median_df = normalizing_log_median.to_frame("normalizing_log_median")
            normalizing_log_median_df["shift"] = shift * self.shift_size
            features["shift"] = shift * self.shift_size
            # Making the index unique for testing
            new_index_columns = ["shift"] + index_column_names
            features = features.reset_index().set_index(new_index_columns)
            y = pd.DataFrame({y_label: y, "shift": shift * self.shift_size}).reset_index().set_index(new_index_columns)[y_label]
            train_data.append((features, y, normalizing_log_median_df.reset_index().rename(columns={"index": page_label})))
        train_Xs, transformed_ys, norm_log_medians = zip(*train_data)
        train_X = pd.concat(train_Xs)
        train_y = pd.concat(transformed_ys)
        norm_log_median_df = pd.concat(norm_log_medians)
        return train_X, train_y, norm_log_median_df

    def _training_score(self, train_X, train_y, train_norm_log_median_df):
        norm_log_median_with_index = pd.merge(train_X.reset_index(), train_norm_log_median_df, on=[page_label, "shift"]).set_index(["shift"] + index_column_names)["normalizing_log_median"]
        train_prediction_normalized = self.model.predict(train_X)
        train_prediction_denorm = FeatureExtractor.inverse_y_transform(train_prediction_normalized.T, norm_log_median_with_index).T
        train_y_denorm = FeatureExtractor.inverse_y_transform(train_y.T, norm_log_median_with_index).T
        return np_smape(train_y_denorm, train_prediction_denorm)
