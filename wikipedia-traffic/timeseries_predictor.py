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


class TimeseriesPredictor(object):
    def __init__(self, days_to_predict, training_shifts, shift_size, model, one_hot=True, disable_parallelism=False):
        self.days_to_predict = days_to_predict
        self.training_shifts = training_shifts
        self.shift_size = shift_size
        self.model = model
        self.one_hot = one_hot
        self.disable_parallelism = disable_parallelism

    def predict(self, X):
        train_data = []
        last_date = X.index[-1]

        feature_extractor = FeatureExtractor(self.days_to_predict, one_hot=self.one_hot, disable_parallelism=self.disable_parallelism)
        for shift in range(self.training_shifts):
            asof = last_date - pd.Timedelta(days=shift * self.shift_size)
            features, y, normalizing_log_median = feature_extractor.features_with_y(X[:asof])
            features["shift"] = shift * self.shift_size
            # Making the index unique for testing
            new_index_columns = ["shift"] + index_column_names
            features = features.reset_index().set_index(new_index_columns)
            y = pd.DataFrame({y_label: y, "shift": shift * self.shift_size}).reset_index().set_index(new_index_columns)[y_label]
            train_data.append((features, y, normalizing_log_median))

        train_Xs, transformed_ys, _ = zip(*train_data)
        train_X = pd.concat(train_Xs)
        train_y = pd.concat(transformed_ys)
        self.model.fit(train_X, train_y)

        test_features, test_normalizing_log_medians = feature_extractor.extract_features(X)
        normalized_predictions = self.model.predict(test_features)
        prediction_label = "prediction"
        normalized_predictions = pd.Series(normalized_predictions, index=test_features.index).to_frame(prediction_label)
        normalized_predictions = normalized_predictions.reset_index().pivot(index=date_label, columns=page_label, values=prediction_label)
        assert set(normalized_predictions.columns) == set(X.columns)
        normalized_predictions = normalized_predictions.reindex(columns=X.columns)
        return FeatureExtractor.inverse_y_transform(normalized_predictions, test_normalizing_log_medians)
